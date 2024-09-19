import os
import tarfile
import urllib.request
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any
import wandb
import io
from PIL import Image
import torchvision
from torch import Tensor


# ----------------------------
# Vision Transformer (ViT) Encoder
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
        )  # Output: (B, embed_dim, H/patch, W/patch)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        # Add Positional Encoding
        x = x + self.pos_embed  # (B, num_patches, embed_dim)
        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)  # (num_patches, B, embed_dim)
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (num_patches, B, embed_dim)
        x = x.transpose(0, 1)  # (B, num_patches, embed_dim)
        return x  # Latent representation


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
    ):
        super(TransformerDecoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Decoder Layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=depth
        )

        # Patch Reconstruction
        self.patch_unembed = nn.Linear(
            embed_dim, patch_size * patch_size * output_channels
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        # Add Positional Encoding
        x = x + self.pos_embed  # (B, num_patches, embed_dim)
        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)  # (num_patches, B, embed_dim)
        # Transformer Decoder
        x = self.transformer_decoder(x)  # (num_patches, B, embed_dim)
        x = x.transpose(0, 1)  # (B, num_patches, embed_dim)
        # Patch Reconstruction
        x = self.patch_unembed(x)  # (B, num_patches, patch_size*patch_size*channels)
        # Reshape to image
        x = x.view(
            batch_size, self.num_patches, -1
        )  # (B, num_patches, patch_size*patch_size*channels)
        # Rearrange patches into image
        patches_per_row = self.image_size // self.patch_size
        channels = x.size(2) // (self.patch_size * self.patch_size)
        reconstructed_image = torch.zeros(
            batch_size, channels, self.image_size, self.image_size, device=x.device
        )
        for i in range(self.num_patches):
            row = i // patches_per_row
            col = i % patches_per_row
            patch = x[:, i, :].view(
                batch_size, channels, self.patch_size, self.patch_size
            )
            reconstructed_image[
                :,
                :,
                row * self.patch_size : (row + 1) * self.patch_size,
                col * self.patch_size : (col + 1) * self.patch_size,
            ] = patch
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
        )
        self.global_step = 0  # Add this line

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


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
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    else:
        print("Imagenette v2 already extracted.")

    return extract_path


# ----------------------------
# Define DataLoaders
# ----------------------------


def get_imagenette_dataloaders(
    data_dir: str, batch_size: int = 32, train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for Imagenette training and validation sets.

    Args:
        data_dir (str): Directory where Imagenette is extracted.
        batch_size (int): Number of images per batch.
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0).

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
                std=[0.229, 0.224, 0.225],
            ),  # Standard ImageNet stds
        ]
    )

    # Create the full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Calculate sizes
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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
    autoencoder.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    global_step = 0

    for epoch in range(num_epochs):
        # Training Phase
        autoencoder.train()
        train_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if wandb_run:
                wandb_run.log({"train_loss": loss.item()}, step=global_step)

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

    # Update the global_step of the model
    autoencoder.global_step = global_step

    # Save Model (Optional)
    if wandb_run:
        torch.save(
            autoencoder.state_dict(),
            f"transformer_autoencoder_{autoencoder.encoder.embed_dim}.pth",
        )
        wandb_run.save(f"transformer_autoencoder_{autoencoder.encoder.embed_dim}.pth")

    return autoencoder, best_val_loss


def get_fixed_images(
    dataloader: DataLoader,
    num_images: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    fixed_images = next(iter(dataloader))[0][:num_images].to(device)
    return fixed_images


def visualize_reconstructions(
    model: nn.Module, fixed_images: Tensor, device: torch.device
) -> Image.Image:
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


# ----------------------------
# Main Execution Block
# ----------------------------

if __name__ == "__main__":
    # Hyperparameters
    image_size = 320
    patch_size = 16
    in_channels = 3
    embed_dims = [64, 128, 256, 512]  # Experiment with different embedding dimensions
    depth = 6
    num_heads = 8
    mlp_dim = 1024
    dropout = 0.1
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    data_dir = "data/imagenette"  # Destination directory for Imagenette
    patience = 10
    min_delta = 0.001
    project_name = "transformer_autoencoder_imagenette"
    run_name_template = "embed_dim_{}"
    num_images_to_reconstruct = 8

    # Download and extract Imagenette v2
    extract_path = download_and_extract_imagenette(dest_dir=data_dir)

    # Get DataLoaders
    train_loader, val_loader = get_imagenette_dataloaders(
        data_dir=extract_path, batch_size=batch_size, train_ratio=0.8
    )

    # Get fixed images for reconstruction visualization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_images = get_fixed_images(
        val_loader, num_images=num_images_to_reconstruct, device=device
    )

    # Initialize Results
    results = []

    # Train Autoencoder with Different Embedding Dimensions
    for embed_dim in embed_dims:
        print(f"\nTraining autoencoder with embedding dimension {embed_dim}")

        # Initialize Autoencoder
        model = TransformerAutoencoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Define Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Define Run Name
        run_name = run_name_template.format(embed_dim)

        # Initialize wandb run
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
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            },
        )
        wandb.watch(model)

        # Train Autoencoder
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
            reconstruction_interval=10,
            fixed_images=fixed_images,
        )

        # Store Results
        results.append((embed_dim, best_val_loss))

        # Visualize and log final reconstructions
        if fixed_images is not None:
            final_reconstruction_image = visualize_reconstructions(
                trained_model, fixed_images, device
            )
            wandb_run.log(
                {"final_reconstructions": wandb.Image(final_reconstruction_image)},
                step=trained_model.global_step,
            )

        # Finish wandb run
        wandb_run.finish()

    # Print Summary of Results
    print("\nTraining Summary:")
    for embed_dim, best_loss in results:
        print(f"Embedding dim: {embed_dim}, Best validation loss: {best_loss:.4f}")
