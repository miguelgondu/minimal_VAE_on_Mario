"""
This script shows how to:
1. Load the VAE provided.
2. Decode a random set of levels.
3. Concatenate them and play them using the simulator.
"""
from pathlib import Path

import torch

from vae import VAEMario
from simulator import test_level_from_int_tensor


def play_random_levels(n_levels: int = 5):
    """
    Loads the provided VAE, decodes n_levels at random,
    and plays them using the simulator interface
    """
    ROOT_DIR = Path(__file__).parent.resolve()

    model = VAEMario()
    model.load_state_dict(torch.load(ROOT_DIR / "models" / "example.pt"))
    model.eval()

    # The latent dim of the model (2, by default)
    latent_dim = model.z_dim

    # Getting {n_levels} random latent codes.
    random_zs = 3.0 * torch.randn((n_levels, latent_dim))

    # Decoding them to a categorical distribution
    p_x_given_z = model.decode(random_zs)

    # Levels (taking the argmax of the probabilities)
    levels = p_x_given_z.probs.argmax(dim=-1)

    # Concatenating them horizontally
    one_large_level = torch.hstack([lvl for lvl in levels])

    # Playing them.
    telemetrics = test_level_from_int_tensor(
        one_large_level, human_player=True, visualize=True
    )
    print(telemetrics)


if __name__ == "__main__":
    play_random_levels()
