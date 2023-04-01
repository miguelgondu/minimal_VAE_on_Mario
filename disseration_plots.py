from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from vae import VAEMario
from mario_utils.plotting import plot_level_from_array

ROOT_DIR = Path(__file__).parent.resolve()
CHAP_6_FIGURES_DIR = Path("/Users/migd/Projects/dissertation/Figures/Chapter_6")

encoding = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
}


def plot_some_training_levels():
    """
    Plots some training levels chosen at random.
    """
    plots_dir = CHAP_6_FIGURES_DIR / "training_levels"
    plots_dir.mkdir(exist_ok=True)

    levels = np.load(ROOT_DIR / "data" / "all_levels_onehot.npz")["levels"]
    levels = np.argmax(levels, axis=1)

    some_level_idxs = np.random.choice(np.arange(len(levels)), size=5)

    for i, level_idx in enumerate(some_level_idxs):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plot_level_from_array(ax, levels[level_idx])

        fig.savefig(plots_dir / f"training_level_{i}.jpg", dpi=120, bbox_inches="tight")

        plt.close(fig)


def plot_heatmaps():
    """
    Loads a VAE, samples a latent code, and visualizes all the
    probability heatmaps for all classes.
    """
    plots_dir = CHAP_6_FIGURES_DIR / "heatmap_and_samples"
    plots_dir.mkdir(exist_ok=True)

    vae = VAEMario()
    vae.load_state_dict(torch.load(ROOT_DIR / "models" / "example.pt"))

    z = 3.0 * vae.p_z.sample((1,))
    categorical_dist = vae.decode(z)
    probs = categorical_dist.probs[0]
    argmax_level = probs.argmax(dim=-1).detach().numpy()

    fig_level, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_level_from_array(ax, probs.argmax(dim=-1).detach().numpy())
    fig_level.savefig(plots_dir / "level.jpg", dpi=120, bbox_inches="tight")

    fig_heatmaps = plt.figure(layout="constrained", figsize=(7 * 4, 7 * 3))
    mosaic = [
        ["X", "S", "-", "?"],
        ["Q", "E", "<", ">"],
        ["[", "]", "o", "."],
    ]
    title_maps = {
        "X": "stone",
        "S": "breakable stone",
        "?": "question",
        "Q": "depleted question",
        "E": "goomba",
        "<": "left pipe head",
        ">": "right pipe head",
        "[": "left pipe",
        "]": "right pipe",
        "o": "coin",
    }
    ax_dict = fig_heatmaps.subplot_mosaic(mosaic)
    # fig_heatmaps, axes_heatmaps = plt.subplots(3, 4, figsize=(7 * 4, ))
    for name, ax in ax_dict.items():
        ax.set_title(name, fontsize=32)
        if name == "level":
            plot_level_from_array(ax, argmax_level)
        else:
            plot = ax.imshow(
                probs[:, :, encoding[name]].detach().numpy(), vmin=0.0, vmax=1.0
            )
            plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)

        ax.axis("off")

    fig_heatmaps.savefig(plots_dir / "probabilities.jpg", dpi=120, bbox_inches="tight")

    sampled_levels = categorical_dist.sample((5,)).squeeze(1).detach().numpy()
    for i, sampled_level in enumerate(sampled_levels):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plot_level_from_array(ax, sampled_level)
        fig.savefig(plots_dir / f"sample_{i}.jpg", dpi=120, bbox_inches="tight")

    # plt.show()


def plot_grid():
    vae = VAEMario()
    vae.load_state_dict(torch.load(ROOT_DIR / "models" / "example.pt"))

    plots_dir = CHAP_6_FIGURES_DIR / "grid_construction"
    plots_dir.mkdir(exist_ok=True)

    fig_grid, ax = plt.subplots(1, 1, figsize=(7, 7))
    vae.plot_grid(n_rows=5, n_cols=5, ax=ax)
    ax.axis("off")
    fig_grid.savefig(plots_dir / "original_grid.jpg", dpi=120, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    # plot_some_training_levels()
    # plot_heatmaps()
    plot_grid()
