import sqlite3
import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import torch
import wandb
from typing import Optional


# Dataset remains unchanged
class CowImageDataset(Dataset):
    def __init__(self, db_path: str = "cow_images.db", transform=None):
        self.db_path = db_path
        self.transform = transform or transforms.ToTensor()
        self.image_ids = self._load_all_image_ids()

    def _load_all_image_ids(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM CowImages")
        ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT image FROM CowImages WHERE id = ?", (img_id,))
            img_data = cursor.fetchone()[0]
        finally:
            conn.close()

        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor


def get_cow_image_dataloaders(
    db_path: str, batch_size: int, train_ratio: float = 0.8, shuffle: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation sets.

    Args:
        db_path (str): Path to the SQLite database.
        batch_size (int): Number of images per batch.
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0).
        shuffle (bool): Whether to shuffle the data.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    dataset = CowImageDataset(db_path)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Modified Encoder for VAE
class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1
            ),  # Output: 32 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # Output: 64 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # Output: 128 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1
            ),  # Output: 256 x 14 x 14
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Decoder remains similar
class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14), nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 128 x 28 x 28
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 64 x 56 x 56
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 32 x 112 x 112
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 3 x 224 x 224
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder_linear(x)
        x = x.view(-1, 256, 14, 14)
        return self.decoder_conv(x)


# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


# VAE Loss Function
def vae_loss(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """
    VAE loss = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss


def train_vae(
    vae: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: Optimizer,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    latent_dim: Optional[int] = None,
    patience: int = 10,
    min_delta: float = 0.001,
):
    # Initialize wandb
    if project_name:
        wandb.init(project=project_name, name=run_name)
        wandb.watch(vae)

        # save the model architecture
        wandb.config.update({"model_architecture": vae})

        # save the latent space dimensionality
        wandb.config.update({"latent_dim": latent_dim})

        # log other hyperparameters
        wandb.config.update({"num_epochs": num_epochs})
        wandb.config.update({"learning_rate": optimizer.param_groups[0]["lr"]})
        wandb.config.update({"batch_size": train_loader.batch_size})

    vae.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training
        vae.train()
        train_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if project_name:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                    }
                )

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon_images, mu, logvar = vae(images)
                loss = vae_loss(recon_images, images, mu, logvar)
                val_loss += loss.item()

                if project_name:
                    wandb.log(
                        {
                            "val_loss": loss.item(),
                        }
                    )

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if project_name:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                }
            )

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model = vae.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"Training complete on {device}")

    if best_model is not None:
        vae.load_state_dict(best_model)

    if project_name:
        # save the best model weights
        torch.save(vae.state_dict(), f"vae_{latent_dim}.pth")

        # save the model to wandb
        wandb.save(f"vae_{latent_dim}.pth")

        wandb.finish()

    return vae, best_val_loss


if __name__ == "__main__":
    # HYPERPARAMETERS
    db_path = "cow_images.db"
    batch_size = 32
    num_epochs = 100  # Increase this, as early stopping will prevent unnecessary epochs
    learning_rate = 0.001
    latent_dims = [16, 32, 64, 128, 256, 512]
    patience = 10
    min_delta = 0.001

    results = []

    for latent_dim in latent_dims:
        print(f"Training VAE with latent dim {latent_dim}")

        train_loader, val_loader = get_cow_image_dataloaders(db_path, batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = VAE(latent_dim).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

        trained_model, best_val_loss = train_vae(
            vae,
            train_loader,
            val_loader,
            num_epochs,
            optimizer,
            device,
            project_name="cow_vae",
            run_name=f"latent_dim_{latent_dim}",
            latent_dim=latent_dim,
            patience=patience,
            min_delta=min_delta,
        )

        results.append((latent_dim, best_val_loss))

    # Print summary of results
    for latent_dim, best_loss in results:
        print(f"Latent dim: {latent_dim}, Best validation loss: {best_loss:.4f}")
