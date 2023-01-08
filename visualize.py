"""
Loads up and visualizes one of the models.
"""

import torch
import matplotlib.pyplot as plt

from vae import VAEMario, load_data

if __name__ == "__main__":
    # Select the model that you would like to run (without the .pt)
    model_name = "example"

    # Loading up the VAE and its weights
    vae = VAEMario()
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))

    # Visualizing the latent space
    _, (ax_grid, ax_latent_codes) = plt.subplots(1, 2, figsize=(7 * 2, 7))

    # Plotting a grid of levels by decoding a grid in latent space
    # and placing the level in the center
    vae.plot_grid(ax=ax_grid)

    # Plotting the encodings
    training_data, test_data = load_data()

    training_encodings = vae.encode(training_data).mean.detach().numpy()
    test_encodings = vae.encode(test_data).mean.detach().numpy()

    ax_latent_codes.scatter(training_encodings[:, 0], training_encodings[:, 1])
    ax_latent_codes.scatter(test_encodings[:, 0], test_encodings[:, 1])

    # Showing
    plt.tight_layout()
    plt.show()
    plt.close()
