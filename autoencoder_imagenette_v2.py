import os
import tarfile
import urllib.request
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils as vutils
from typing import Optional, Tuple
import wandb


# ----------------------------
# 1. Download and Extract Imagenette v2
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
# 2. Define DataLoaders
# ----------------------------


def get_imagenette_dataloaders(
    data_dir: str, batch_size: int = 32, train_ratio: float = 0.8
) -> tuple:
    """
    Create DataLoaders for Imagenette training and validation sets.

    Args:
        data_dir (str): Directory where Imagenette is extracted.
        batch_size (int): Number of images per batch.
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0).

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
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
# 3. Define the Autoencoder
# ----------------------------


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 32 x 160 x 160
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 80 x 80
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 40 x 40
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 20 x 20
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 256*20*20 = 102400
            nn.Linear(256 * 20 * 20, latent_dim),  # latent_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 20 * 20),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 20, 20)),  # 256 x 20 x 20
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # 128 x 40 x 40
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 x 80 x 80
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 32 x 160 x 160
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),  # 3 x 320 x 320
            nn.Sigmoid(),  # Ensures the output is between 0 and 1
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# ----------------------------
# 4. Training Function
# ----------------------------


def train_autoencoder(
    autoencoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    wandb_run: Optional[wandb.Run] = None,
    patience: int = 10,
    min_delta: float = 0.001,
) -> Tuple[nn.Module, float]:
    """
    Train the autoencoder model.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        optimizer (Adam): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on.
        wandb_run (Optional[wandb.Run]): WandB run object for logging.
        patience (int): Early stopping patience.
        min_delta (float): Minimum change to qualify as improvement.

    Returns:
        Tuple[nn.Module, float]: Trained model and best validation loss.
    """
    autoencoder.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training Phase
        autoencoder.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            # Optional: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if wandb_run:
                wandb_run.log({"train_loss": loss.item()})

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

                if wandb_run:
                    wandb_run.log({"val_loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                }
            )

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
        torch.save(
            autoencoder.state_dict(),
            f"autoencoder_{autoencoder.encoder.latent_dim}.pth",
        )
        wandb_run.save(f"autoencoder_{autoencoder.encoder.latent_dim}.pth")

    return autoencoder, best_val_loss


# ----------------------------
# 5. Visualization Function (Optional)
# ----------------------------


def visualize_reconstructions(
    autoencoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_images: int = 8,
    filename: str = "reconstructions.png",
    wandb_run: Optional[wandb.Run] = None,
):
    """
    Visualize original and reconstructed images side by side and save to a file and wandb.

    Args:
        autoencoder (nn.Module): Trained autoencoder model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        num_images (int): Number of images to display.
        filename (str): Filename to save the plot.
        wandb_run (Optional[wandb.Run]): WandB run object for logging.
    """
    autoencoder.eval()
    with torch.no_grad():
        # Get a batch of images
        images, _ = next(iter(dataloader))
        images = images.to(device)
        outputs = autoencoder(images)

    # Move images to CPU and denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    images = images.cpu() * std + mean
    outputs = outputs.cpu() * std + mean

    # Clamp to [0,1]
    images = torch.clamp(images, 0, 1)
    outputs = torch.clamp(outputs, 0, 1)

    # Create a grid of original images
    grid_original = vutils.make_grid(images[:num_images], nrow=num_images, padding=2)
    # Create a grid of reconstructed images
    grid_reconstructed = vutils.make_grid(
        outputs[:num_images], nrow=num_images, padding=2
    )

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(num_images * 2, 4))
    axs[0].imshow(grid_original.permute(1, 2, 0).numpy())
    axs[0].set_title("Original Images")
    axs[0].axis("off")

    axs[1].imshow(grid_reconstructed.permute(1, 2, 0).numpy())
    axs[1].set_title("Reconstructed Images")
    axs[1].axis("off")

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    # Log the plot to wandb
    if wandb_run is not None:
        wandb_run.log({"reconstructions": wandb.Image(filename)})

    plt.close(fig)  # Close the figure to free up memory

    print(f"Reconstructions saved to {filename} and logged to wandb")


# ----------------------------
# 6. Main Execution Block
# ----------------------------

if __name__ == "__main__":
    # ----------------------------
    # Hyperparameters
    # ----------------------------
    data_dir = "data/imagenette"  # Destination directory for Imagenette
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    latent_dims = [64, 128, 256]  # Experiment with different latent dimensions
    patience = 10
    min_delta = 0.001
    project_name = "autoencoder_imagenettev2"
    run_name_template = "latent_dim_{}"

    # ----------------------------
    # Step 1: Download and Extract Imagenette v2
    # ----------------------------
    extract_path = download_and_extract_imagenette(dest_dir=data_dir)

    # ----------------------------
    # Step 2: Get DataLoaders
    # ----------------------------
    train_loader, val_loader = get_imagenette_dataloaders(
        data_dir=extract_path, batch_size=batch_size, train_ratio=0.8
    )

    # ----------------------------
    # Step 3: Initialize Results
    # ----------------------------
    results = []

    # ----------------------------
    # Step 4: Train Autoencoder with Different Latent Dimensions
    # ----------------------------
    for latent_dim in latent_dims:
        print(f"\nTraining autoencoder with latent dim {latent_dim}")

        # Initialize Autoencoder
        autoencoder = Autoencoder(latent_dim=latent_dim)

        # Define Optimizer and Loss
        optimizer = Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Define Run Name (Optional)
        run_name = run_name_template.format(latent_dim) if project_name else None

        # Initialize wandb run
        wandb_run = None
        if project_name:
            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model_architecture": str(autoencoder),
                    "latent_dim": latent_dim,
                    "num_epochs": num_epochs,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batch_size": train_loader.batch_size,
                },
            )
            wandb.watch(autoencoder)

        # Train Autoencoder
        trained_model, best_val_loss = train_autoencoder(
            autoencoder=autoencoder,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            wandb_run=wandb_run,
            patience=patience,
            min_delta=min_delta,
        )

        # Store Results
        results.append((latent_dim, best_val_loss))

        # Visualize Reconstructions
        visualize_reconstructions(
            trained_model,
            val_loader,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_images=8,
            filename=f"reconstructions_{latent_dim}.png",
            wandb_run=wandb_run,
        )

        # Finish wandb run
        if wandb_run is not None:
            wandb_run.finish()

    # ----------------------------
    # Step 5: Print Summary of Results
    # ----------------------------
    print("\nTraining Summary:")
    for latent_dim, best_loss in results:
        print(f"Latent dim: {latent_dim}, Best validation loss: {best_loss:.4f}")
