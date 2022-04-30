"""
Loads up and visualizes one of the models.
"""

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae import VAEMario

if __name__ == "__main__":
    model_name = "example"
    vae = VAEMario()
    vae.load_state_dict(t.load(f"./models/{model_name}.pt"))

    _, ax = plt.subplots(1, 1)
    vae.plot_grid(ax=ax)
    plt.show()
    plt.close()
