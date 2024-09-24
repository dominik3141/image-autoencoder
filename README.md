# Image Autoencoder Project

This project implements and trains autoencoders and variational autoencoders (VAEs) on image datasets, specifically Imagenette v2 and a custom cow image dataset.

## Project Structure

- `autoencoder_imagenette_v2.py`: Implements and trains an autoencoder on the Imagenette v2 dataset.
- `vae_cow.py`: Implements and trains a variational autoencoder on a custom cow image dataset.
- `autoencoder_cow.py`: Implements and trains an autoencoder on a custom cow image dataset (Note: This script has memory issues and needs optimization).

## Features

- Downloads and preprocesses the Imagenette v2 dataset
- Implements custom datasets for SQLite-based image storage
- Trains autoencoders and VAEs with various latent space dimensions
- Uses early stopping to prevent overfitting
- Logs training progress and results using Weights & Biases (wandb)
- Visualizes original and reconstructed images

