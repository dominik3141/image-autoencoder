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
from torch.amp import GradScaler, autocast

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
        use_cls_token: bool = True,  # New parameter
    ):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Output: (B, embed_dim, H/patch, W/patch)

        # CLS Token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.num_patches, embed_dim)
            )
        else:
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
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, E)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, E)

        # Add Positional Encoding
        x = x + self.pos_embed  # (B, 1 + num_patches, E) or (B, num_patches, E)

        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)  # (S, B, E)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (S, B, E)

        x = x.transpose(0, 1)  # (B, S, E)

        if self.use_cls_token:
            cls_embedding = x[:, 0]  # (B, E)
            return cls_embedding  # Global latent representation
        return x  # (B, num_patches, E)


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
        latent_dim: int = 768,  # New parameter
    ):
        super(TransformerDecoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Project latent vector to embed_dim
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Positional Encoding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=depth
        )

        # Patch Reconstruction
        self.patch_unembed = nn.Linear(
            embed_dim, patch_size * patch_size * output_channels
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, latent: Tensor) -> Tensor:
        batch_size = latent.size(0)

        # Project latent vector to embed_dim
        latent = self.latent_proj(latent)  # (B, E)

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
        use_cls_token: bool = True,  # New parameter
        latent_dim: int = 768,  # New parameter
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
            use_cls_token=use_cls_token,
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
        self.global_step = 0  # Tracking training steps

    def forward(self, x):
        latent = self.encoder(x)  # (B, latent_dim)
        reconstructed = self.decoder(latent)  # (B, C, H, W)
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
    # Define transformations with data augmentation
    transform = transforms.Compose(
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

    # Create the full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Calculate sizes
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split the dataset
    generator = torch.Generator().manual_seed(
        1208
    )  # Set a fixed seed for reproducibility
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
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
    scaler = GradScaler()  # For mixed precision

    for epoch in range(num_epochs):
        # Training Phase
        autoencoder.train()
        train_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            # Use autocast for mixed precision
            with autocast(device_type=device.type):
                outputs = autoencoder(images)
                loss = criterion(outputs, images)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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
    embed_dims = [
        64,
        128,
        256,
        512,
        768,
        1024,
    ]  # Experiment with different embedding dimensions
    depth = 6
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

    # Function to determine a suitable number of heads based on embed_dim
    def get_num_heads(embed_dim):
        if embed_dim % 16 == 0 and embed_dim // 16 >= 4:
            return 16
        elif embed_dim % 8 == 0 and embed_dim // 8 >= 4:
            return 8
        elif embed_dim % 4 == 0 and embed_dim // 4 >= 4:
            return 4
        else:
            return 1  # Fallback

    # Train Autoencoder with Different Embedding Dimensions
    for embed_dim in embed_dims:
        print(f"\nTraining autoencoder with embedding dimension {embed_dim}")

        num_heads = get_num_heads(embed_dim)  # Determine suitable number of heads
        latent_dim = embed_dim  # Assuming latent_dim equals embed_dim

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
            use_cls_token=True,  # Enable global latent vector (should never be disabled)
            latent_dim=latent_dim,
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
                "precision": "mixed",
            },
            reinit=True,  # Allow multiple runs in the same script
        )
        wandb.watch(model, log="all")

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
            reconstruction_interval=5,
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
