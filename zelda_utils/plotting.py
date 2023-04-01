"""
This script defines some functions that
plot levels.
"""
import os
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL

from mario_utils.levels import onehot_to_levels

filepath = Path(__file__).parent.resolve()
Tensor = torch.Tensor


def absolute(path_str):
    return str(Path(path_str).absolute())


encoding = {
    "#": 0,
    "-": 1,
    "B": 2,
    "D": 3,
    "F": 4,
    "I": 5,
    "L": 6,
    "M": 7,
    "O": 8,
    "P": 9,
    "S": 10,
    "U": 11,
    "V": 12,
    "W": 13,
}

sprites = {
    v: absolute(f"{filepath}/sprites/{k.lower()}.png") for k, v in encoding.items()
}


def save_level_from_array(path, level, title=None, dpi=150):
    # Assuming that the level is a bunch of classes.
    image = get_img_from_level(level)
    plt.imshow(255 * np.ones_like(image))  # White background
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def plot_level_from_array(ax, level, title=None):
    # Assuming that the level is a bunch of classes.
    image = get_img_from_level(level)
    ax.imshow(255 * np.ones_like(image))  # White background
    ax.imshow(image)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)


def get_img_from_level(level: np.ndarray):
    image = []
    for row in level:
        image_row = []
        for c in row:
            tile = np.asarray(PIL.Image.open(sprites[c]).convert("RGBA")).astype(int)
            tile[tile[:, :, 3] == 0] = [255, 255, 255, 255]
            image_row.append(tile)
        image.append(image_row)

    image = [np.hstack([tile for tile in row]) for row in image]
    image = np.vstack([np.asarray(row) for row in image])

    return image


def plot_level_from_decoded_tensor(dec: Tensor, ax):
    """
    Plots decoded tensor as level in ax.
    Expects {dec} to have a batch component.
    """
    level = onehot_to_levels(dec.detach().numpy())[0]
    plot_level_from_array(ax, level)
