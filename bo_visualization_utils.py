"""
This script contains some visualization
tools for the Bayesian Optimization experiment
that can be found in simple_bayesian_optimization.py.
"""

from typing import Tuple
from itertools import product

import matplotlib.pyplot as plt
import torch
import numpy as np

from gpytorch.models import ExactGP
from botorch.acquisition import AcquisitionFunction


def _image_from_values(
    values: torch.Tensor,
    limits: Tuple[float, float],
    n_points_in_grid: int,
):
    """
    Transforms a tensor of values into an
    {n_points_in_grid}x{n_points_in_grid} image.
    """
    z1s = torch.linspace(*limits, n_points_in_grid)
    z2s = torch.linspace(*limits, n_points_in_grid)

    fine_grid = torch.Tensor([[x, y] for x, y in product(z1s, z2s)])
    p_dict = {(x.item(), y.item()): v.item() for (x, y), v in zip(fine_grid, values)}

    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1s)
        for i, y in enumerate(reversed(z2s))
    }

    p_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        p_img[i, j] = p_dict[z]

    return p_img


def plot_prediction(model: ExactGP, ax: plt.Axes):
    """
    Plots mean of the GP in the axes in a fine grid
    in latent space. Assumes that the latent space
    is of size 2.
    """
    limits = [-5, 5]
    n_points_in_grid = 75

    fine_grid_in_latent_space = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(-5, 5, n_points_in_grid),
                torch.linspace(-5, 5, n_points_in_grid),
            )
        ]
    )

    predicted_distribution = model(fine_grid_in_latent_space)
    means = predicted_distribution.mean
    means_as_img = _image_from_values(means, limits, n_points_in_grid)

    plot = ax.imshow(means_as_img, extent=[*limits, *limits], cmap="Blues")
    plt.colorbar(plot, ax=ax)


def plot_acquisition(acq_function: AcquisitionFunction, ax: plt.Axes):
    limits = [-5, 5]
    n_points_in_grid = 75

    fine_grid_in_latent_space = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(-5, 5, n_points_in_grid),
                torch.linspace(-5, 5, n_points_in_grid),
            )
        ]
    ).unsqueeze(1)

    acq_values = acq_function(fine_grid_in_latent_space)
    acq_values_as_img = _image_from_values(acq_values, limits, n_points_in_grid)

    plot = ax.imshow(acq_values_as_img, extent=[*limits, *limits], cmap="Blues")
    plt.colorbar(plot, ax=ax)
