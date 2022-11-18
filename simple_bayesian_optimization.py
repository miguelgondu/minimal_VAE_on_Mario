"""
Implements vanilla Bayesian Optimization (without constraints)
in the latent space of our SMB VAEs. The objective function
is the number of jumps thrown by the simulator (divided by ten).

After running this, you can see the latent space queries at
./data/plots/bayesian_optimization/vanilla_bo.
"""
from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt

import torch as t

from torch.distributions import Uniform

import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood

from simulator import test_level_from_int_tensor
from vae import VAEMario
from bo_visualization_utils import plot_prediction, plot_acquisition


ROOT_DIR = Path(__file__).parent.resolve()

gpytorch.settings.cholesky_jitter(float=1e-3, double=1e-4)


def run_first_samples(
    vae: VAEMario,
    n_samples: int = 2,
    visualize: bool = False,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs the simulator on {n_samples} levels selected uniformly
    at random from the latent space (considered to be bounded in
    the [-5, 5]^2 square). Returns the latent codes, jumps and
    playabilities (a binary value stating whether the level was
    solved or not).
    """
    latent_codes = Uniform(t.Tensor([-5.0, -5.0]), t.Tensor([5.0, 5.0])).sample(
        (n_samples,)
    )
    levels = vae.decode(latent_codes).probs.argmax(dim=-1)

    playability = []
    jumps = []
    for i, level in enumerate(levels):
        results = test_level_from_int_tensor(level, visualize=visualize)
        playability.append(results["marioStatus"])
        jumps.append(results["jumpActionsPerformed"])
        print(
            "i:",
            i,
            "p:",
            results["marioStatus"],
            "jumps:",
            results["jumpActionsPerformed"],
        )

    # Returning.
    return latent_codes, t.Tensor(playability), t.Tensor(jumps)


def bayesian_optimization_iteration(
    latent_codes: t.Tensor,
    jumps: t.Tensor,
    iteration: int = 0,
    plot_latent_space: bool = False,
    img_save_folder: Path = None,
    visualize: bool = False,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    # Load the model
    vae = VAEMario()
    vae.load_state_dict(t.load("./models/example.pt"))
    vae.eval()

    kernel = gpytorch.kernels.MaternKernel()
    model = SingleTaskGP(latent_codes, (jumps / 10.0), covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    model.eval()
    acq_function = ExpectedImprovement(model, (jumps / 10.0).max())

    # Optimizing the acq. function by hand on a discrete grid.
    zs = t.Tensor(
        [
            [x, y]
            for x in t.linspace(-5, 5, 100)
            for y in reversed(t.linspace(-5, 5, 100))
        ]
    )
    acq_values = acq_function(zs.unsqueeze(1))
    candidate = zs[acq_values.argmax()]

    print(candidate, acq_values[acq_values.argmax()])
    level = vae.decode(candidate).probs.argmax(dim=-1)
    print(level)
    results = test_level_from_int_tensor(level[0], visualize=visualize)

    if plot_latent_space:
        fig, (ax, ax_acq) = plt.subplots(1, 2)
        plot_prediction(model, ax)
        plot_acquisition(acq_function, ax_acq)

        ax.scatter(
            latent_codes[:, 0].cpu().detach().numpy(),
            latent_codes[:, 1].cpu().detach().numpy(),
            c="black",
            marker="x",
        )
        ax.scatter(
            [candidate[0].cpu().detach().numpy()],
            [candidate[1].cpu().detach().numpy()],
            c="red",
            marker="d",
        )

        ax.scatter(
            latent_codes[:, 0].cpu().detach().numpy(),
            latent_codes[:, 1].cpu().detach().numpy(),
            c="black",
            marker="x",
        )
        ax.scatter(
            [candidate[0].cpu().detach().numpy()],
            [candidate[1].cpu().detach().numpy()],
            c="red",
            marker="d",
        )

        if img_save_folder is not None:
            img_save_folder.mkdir(exist_ok=True, parents=True)
            fig.tight_layout()
            fig.savefig(img_save_folder / f"{iteration:04d}.png")
        # plt.show()
        plt.close(fig)

    return (
        candidate,
        t.Tensor([[results["marioStatus"]]]),
        t.Tensor([[results["jumpActionsPerformed"]]]),
    )


def run_experiment(n_iterations: int = 50, visualize: bool = False):
    # Load the model
    vae = VAEMario()
    vae.load_state_dict(t.load("./models/example.pt"))
    vae.eval()

    # Get some first samples and save them.
    latent_codes, playabilities, jumps = run_first_samples(
        vae, n_samples=2, visualize=visualize
    )
    jumps = jumps.type(t.float32).unsqueeze(1)
    playabilities = playabilities.unsqueeze(1)

    # To disencourage exploiting unplayable levels,
    # I mask the non-playable ones with 0 jumps.
    # This is a hack, though.
    jumps[playabilities == 0.0] = 0.0

    # The path to save the images in.
    img_save_folder = (
        ROOT_DIR / "data" / "plots" / "bayesian_optimization" / "vanilla_bo"
    )
    img_save_folder.mkdir(exist_ok=True, parents=True)

    # The B.O. loops; they might hang because of numerical instabilities.
    for i in range(n_iterations):
        candidate, playability, jump = bayesian_optimization_iteration(
            latent_codes,
            jumps,
            plot_latent_space=True,
            iteration=i,
            img_save_folder=img_save_folder,
            visualize=visualize,
        )
        print(f"(Iteration {i+1}) tested {candidate} and got {jump} (p={playability})")

        if playability == 0.0:
            jump = t.zeros_like(jump)

        latent_codes = t.vstack((latent_codes, candidate))
        jumps = t.vstack((jumps, jump))
        playabilities = t.vstack((playabilities, playability))


if __name__ == "__main__":
    # visualize == seeing the agent playing on screen.
    run_experiment(visualize=True)
