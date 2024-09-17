import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import List, Optional
from torch.utils.data import DataLoader
from autoencoder_cow import Autoencoder


def compare_reconstructions(
    latent_dims: List[int],
    dataloader: DataLoader,
    num_images: Optional[int] = 4,
    filename: Optional[str] = "reconstruction_comparison.png",
    device: Optional[torch.device] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
):
    """
    Compare the reconstructions of different latent space dimensions.
    """
    with torch.no_grad():
        # Get a batch of images
        images = next(iter(dataloader))
        images = images[:num_images].to(device)

        # Create a list to store reconstructions from each model
        all_reconstructions = []

        for latent_dim in latent_dims:
            # Load the model
            model = Autoencoder(latent_dim).to(device)
            model.load_state_dict(
                torch.load(f"autoencoder_{latent_dim}.pth", map_location=device)
            )
            model.eval()

            # Get reconstructions
            latent = model.encoder(images)
            reconstructions = model.decoder(latent)
            all_reconstructions.append(reconstructions.cpu())

        # Create the comparison grid
        num_models = len(latent_dims)
        comparison = torch.cat([images.cpu()] + all_reconstructions)
        grid = vutils.make_grid(comparison, nrow=num_images, normalize=True, padding=2)

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 3 * (num_models + 1)))
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")

        # Add labels
        num_rows = num_models + 1
        row_height = 1.0 / num_rows

        for i in range(num_rows):
            y_position = 1 - (i + 0.5) * row_height
            if i == 0:
                label = "Original"
            else:
                label = f"Latent dim: {latent_dims[i-1]}"

            ax.text(
                -0.05,
                y_position,
                label,
                va="center",
                ha="right",
                fontsize=10,
                transform=ax.transAxes,
            )

        plt.title("Original Images vs Reconstructions with Different Latent Dimensions")
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Comparison plot saved as {filename}")


# Example usage
if __name__ == "__main__":
    from autoencoder_cow import get_cow_image_dataloaders

    db_path = "cow_images.db"
    batch_size = 32
    latent_dims = [16, 32, 64, 128, 256, 512]

    _, val_loader = get_cow_image_dataloaders(db_path, batch_size)

    compare_reconstructions(latent_dims, val_loader, num_images=4)
