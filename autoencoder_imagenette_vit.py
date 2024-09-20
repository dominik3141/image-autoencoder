import os
import tarfile
import urllib.request
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any
import wandb
import io
from PIL import Image
import torchvision
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import math
import gc

# ----------------------------
# Vision Transformer (ViT) Encoder with CLS Token
# ----------------------------


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS Token and Positional Embedding are both learned parameters (initialized to 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # add a cls token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, E)

        # Add Positional Encoding
        x = x + self.pos_embed  # (B, 1 + num_patches, E)

        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)  # (S, B, E)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (S, B, E)

        x = x.transpose(0, 1)  # (B, S, E)

        cls_embedding = x[:, 0]  # (B, E) # we only return the cls embedding
        return cls_embedding


# ----------------------------
# Transformer-Based Decoder
# ----------------------------


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        output_channels: int = 3,
        latent_dim: int = 768,
    ):
        super(TransformerDecoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        assert latent_dim == embed_dim, "Latent dimension must be equal to embed_dim"

        # Positional Encoding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=depth
        )

        # Patch Reconstruction
        self.patch_unembed = nn.Linear(
            embed_dim, patch_size * patch_size * output_channels
        )

    def forward(self, latent: Tensor) -> Tensor:
        batch_size = latent.size(0)

        assert latent.shape == (
            batch_size,
            self.latent_dim,
        ), f"Latent shape must be (B, {self.latent_dim}), got {latent.shape}"

        # Expand latent to match number of patches (as memory for decoder)
        memory = latent.unsqueeze(1).repeat(
            1, self.num_patches, 1
        )  # (B, num_patches, E)

        # Add Positional Encoding
        target = torch.zeros(
            self.num_patches, batch_size, self.embed_dim, device=latent.device
        )  # (S, B, E)
        target = target + self.pos_embed.transpose(0, 1)  # (S, B, E)

        # Transformer Decoder
        x = self.transformer_decoder(target, memory.transpose(0, 1))  # (S, B, E)

        x = x.transpose(0, 1)  # (B, S, E)

        # Patch Reconstruction
        x = self.patch_unembed(x)  # (B, S, patch_size*patch_size*channels)

        # Reshape and rearrange patches into image using tensor operations
        channels = x.size(2) // (self.patch_size * self.patch_size)
        patches_per_side = self.image_size // self.patch_size

        x = x.view(
            batch_size,
            patches_per_side,
            patches_per_side,
            channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        reconstructed_image = x.view(
            batch_size, channels, self.image_size, self.image_size
        )

        return reconstructed_image


# ----------------------------
# Complete Transformer Autoencoder
# ----------------------------


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        latent_dim: int = 768,
    ):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            output_channels=in_channels,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ----------------------------
# Download and Extract Imagenette v2
# ----------------------------


def download_url(url: str, save_path: str):
    """Download a file from a URL with a progress bar."""

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=os.path.basename(save_path)
    ) as t:
        urllib.request.urlretrieve(url, filename=save_path, reporthook=t.update_to)


def safe_extract(
    tar: tarfile.TarFile, path: str = ".", members=None, *, numeric_owner=False
):
    """Safely extract tar files to prevent path traversal vulnerabilities."""
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(member_path)]
        ).startswith(os.path.abspath(path)):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)


def download_and_extract_imagenette(dest_dir: str = "data/imagenette") -> str:
    """
    Downloads and extracts the Imagenette v2 dataset.

    Args:
        dest_dir (str): Directory where the dataset will be stored.

    Returns:
        str: Path to the extracted dataset.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tgz_path = os.path.join(dest_dir, "imagenette2-320.tgz")

    if not os.path.exists(tgz_path):
        print("Downloading Imagenette v2...")
        download_url(url, tgz_path)
    else:
        print("Imagenette v2 archive already exists.")

    extract_path = os.path.join(dest_dir, "imagenette2-320")
    if not os.path.exists(extract_path):
        print("Extracting Imagenette v2...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            safe_extract(tar, path=dest_dir)
        print("Extraction complete.")
    else:
        print("Imagenette v2 already extracted.")

    return extract_path


# ----------------------------
# Define DataLoaders
# ----------------------------


def get_imagenette_dataloaders(
    data_dir: str, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for Imagenette training and validation sets.

    Args:
        data_dir (str): Directory where Imagenette is extracted.
        batch_size (int): Number of images per batch.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Define transformations with data augmentation for training
    train_transform = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
                std=[0.229, 0.224, 0.225],
            ),  # Standard ImageNet stds
        ]
    )

    # Define transformations without augmentation for validation
    val_transform = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Create separate datasets for training and validation
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


# ----------------------------
# Training and Evaluation Functions
# ----------------------------


def get_num_heads(embed_dim: int) -> int:
    return max(1, embed_dim // 8)


def get_fixed_images(
    dataloader: DataLoader,
    num_images: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    fixed_images = next(iter(dataloader))[0][:num_images].to(device)
    return fixed_images


def find_optimal_batch_size(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    initial_batch_size: int = 4,
    max_batch_size: int = 512,
) -> int:
    """
    Finds the largest possible batch size that fits into GPU memory.

    Args:
        model (nn.Module): The model to test.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        initial_batch_size (int): Starting batch size for the search.
        max_batch_size (int): Maximum batch size to try.

    Returns:
        int: Optimal batch size.
    """
    model.train()  # Set model to training mode
    device = next(model.parameters()).device
    batch_size = initial_batch_size

    while batch_size <= max_batch_size:
        try:
            loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
            inputs, _ = next(iter(loader))
            inputs = inputs.to(device)

            outputs = model.encoder(inputs)
            reconstructed = model.decoder(outputs)
            loss = torch.sum(reconstructed)
            loss.backward()

            batch_size *= 2

            del inputs, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory with batch size {batch_size}.")
                batch_size = max(batch_size // 2, initial_batch_size)
                print(f"Optimal batch size: {batch_size}")
                return batch_size
            else:
                raise e

    print(f"Reached max batch size. Optimal batch size: {batch_size}")
    return batch_size


def find_lr(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Finds the optimal learning rate by increasing it exponentially and tracking loss.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        float: Optimal learning rate.
    """
    num_batches = 100
    log_lrs = torch.linspace(math.log(1e-7), math.log(10), num_batches)
    lrs = torch.exp(log_lrs)
    losses = []

    model.train()
    for i, (batch, _) in enumerate(train_loader):
        if i >= num_batches:
            break

        batch = batch.to(device)
        optimizer.param_groups[0]["lr"] = lrs[i].item()

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    best_lr = lrs[losses.index(min(losses))].item()
    return best_lr


def train_autoencoder(
    autoencoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    wandb_run: Optional[Any] = None,
    patience: int = 10,
    min_delta: float = 0.001,
    reconstruction_interval: int = 10,
    fixed_images: Optional[Tensor] = None,
) -> Tuple[nn.Module, float]:
    """
    Trains the transformer autoencoder with early stopping and learning rate scheduling.

    Args:
        autoencoder (nn.Module): The transformer autoencoder model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        num_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.
        wandb_run (Optional[Any]): Weights & Biases run instance.
        patience (int): Early stopping patience.
        min_delta (float): Minimum change to qualify as improvement.
        reconstruction_interval (int): Interval (in epochs) to log reconstructions.
        fixed_images (Optional[Tensor]): Fixed images for visualization.

    Returns:
        Tuple[nn.Module, float]: Trained model and best validation loss.
    """
    autoencoder.to(device)
    autoencoder.train()  # Ensure model is in training mode
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    global_step = 0
    scaler = GradScaler()

    # Find the best learning rate
    print("Finding best learning rate...")
    best_lr = find_lr(autoencoder, train_loader, optimizer, criterion, device)
    print(f"Best learning rate found: {best_lr}")

    # Reset the optimizer with the found learning rate
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=best_lr)

    # Set up the OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer, max_lr=best_lr, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )

    for epoch in range(num_epochs):
        # Training Phase
        autoencoder.train()
        train_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            # Use autocast for mixed precision
            with autocast():
                outputs = autoencoder(images)
                loss = criterion(outputs, images)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss += loss.item()

            if wandb_run:
                wandb_run.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = autoencoder(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Log epoch-level metrics and reconstructions
        if wandb_run:
            log_dict = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            }

            # Visualize reconstructions every 'reconstruction_interval' epochs
            if (epoch + 1) % reconstruction_interval == 0 and fixed_images is not None:
                reconstruction_image = visualize_reconstructions(
                    autoencoder, fixed_images, device
                )
                log_dict["reconstructions"] = wandb.Image(reconstruction_image)

            wandb_run.log(log_dict, step=global_step)

        # Early Stopping Check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model = autoencoder.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"Training complete on {device}")

    # Load Best Model
    if best_model is not None:
        autoencoder.load_state_dict(best_model)

    # Save Model (Optional)
    if wandb_run:
        model_path = f"transformer_autoencoder_{autoencoder.encoder.embed_dim}.pth"
        torch.save(autoencoder.state_dict(), model_path)
        wandb.save(model_path)

    return autoencoder, best_val_loss


def visualize_reconstructions(
    model: nn.Module, fixed_images: Tensor, device: torch.device
) -> Image.Image:
    """
    Generates a visualization of original and reconstructed images.

    Args:
        model (nn.Module): Trained autoencoder model.
        fixed_images (Tensor): Fixed images for visualization.
        device (torch.device): Device to run on.

    Returns:
        Image.Image: PIL Image containing the visualization.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(fixed_images)

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    images = fixed_images * std + mean
    outputs = outputs * std + mean

    # Clamp to [0,1]
    images = torch.clamp(images, 0, 1)
    outputs = torch.clamp(outputs, 0, 1)

    # Create a grid of original images
    grid_original = torchvision.utils.make_grid(
        images.cpu(), nrow=len(images), padding=2
    )
    # Create a grid of reconstructed images
    grid_reconstructed = torchvision.utils.make_grid(
        outputs.cpu(), nrow=len(outputs), padding=2
    )

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(len(images) * 2, 4))
    axs[0].imshow(grid_original.permute(1, 2, 0).numpy())
    axs[0].set_title("Original Images")
    axs[0].axis("off")

    axs[1].imshow(grid_reconstructed.permute(1, 2, 0).numpy())
    axs[1].set_title("Reconstructed Images")
    axs[1].axis("off")

    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Open as PIL Image
    pil_image = Image.open(buf)

    plt.close(fig)  # Close the figure to free up memory

    return pil_image


if __name__ == "__main__":
    # Hyperparameters
    image_size = 320
    patch_size = 16
    in_channels = 3
    embed_dims = [64, 128, 256, 512, 768, 1024]
    depth = 8
    mlp_dim = 1024
    dropout = 0.0  # no dropout for now
    num_epochs = 50
    data_dir = "data/imagenette"
    patience = 10
    min_delta = 0.001
    project_name = "transformer_autoencoder_imagenette"
    run_name_template = "embed_dim_{}"
    num_images_to_reconstruct = 8

    # Download and extract Imagenette v2
    extract_path = download_and_extract_imagenette(dest_dir=data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for embed_dim in embed_dims:
        print(f"\nTraining autoencoder with embedding dimension {embed_dim}")

        num_heads = get_num_heads(embed_dim)
        latent_dim = embed_dim

        # 1. Initialize the model
        model = TransformerAutoencoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            latent_dim=latent_dim,
        ).to(device)

        # 2. Create data loaders with temporary batch size for batch size search
        initial_batch_size = 4
        train_loader_temp, _ = get_imagenette_dataloaders(
            data_dir=extract_path, batch_size=initial_batch_size
        )
        train_dataset_temp = train_loader_temp.dataset

        # 3. Find the optimal batch size using the training dataset
        optimal_batch_size = find_optimal_batch_size(model, train_dataset_temp)
        print(f"Optimal batch size determined: {optimal_batch_size}")

        # 4. Create data loaders with the optimal batch size
        train_loader, val_loader = get_imagenette_dataloaders(
            data_dir=extract_path, batch_size=optimal_batch_size
        )

        # Get fixed images for reconstruction visualization
        fixed_images = get_fixed_images(
            val_loader, num_images=num_images_to_reconstruct, device=device
        )

        # 5. Initialize criterion and optimizer with a default learning rate
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3
        )  # Initial LR will be overridden

        # 6. Define Run Name
        run_name = run_name_template.format(embed_dim)

        # 7. Initialize wandb run
        wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_architecture": str(model),
                "embed_dim": embed_dim,
                "image_size": image_size,
                "patch_size": patch_size,
                "depth": depth,
                "num_heads": num_heads,
                "mlp_dim": mlp_dim,
                "dropout": dropout,
                "num_epochs": num_epochs,
                "batch_size": optimal_batch_size,
                "initial_learning_rate": 1e-3,  # Placeholder; will be updated
                "precision": "mixed",
                "lr_scheduler": "OneCycleLR",
            },
            reinit=True,
        )
        wandb.watch(model, log="all")

        # 8. Train Autoencoder
        trained_model, best_val_loss = train_autoencoder(
            autoencoder=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            wandb_run=wandb_run,
            patience=patience,
            min_delta=min_delta,
            reconstruction_interval=5,
            fixed_images=fixed_images,
        )

        # Store Results
        results.append((embed_dim, best_val_loss))

        # Finish wandb run
        wandb_run.finish()

    # Print Summary of Results
    print("\nTraining Summary:")
    for embed_dim, best_loss in results:
        print(f"Embedding dim: {embed_dim}, Best validation loss: {best_loss:.4f}")
