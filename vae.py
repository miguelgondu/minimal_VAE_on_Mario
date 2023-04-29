"""
A categorical VAE that can train on Mario.

The notation very much follows the original VAE paper.
"""
from itertools import product
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
import torch.nn as nn

from mario_utils.plotting import get_img_from_level


def load_data(
    training_percentage=0.8,
    shuffle_seed=0,
    device="cpu",
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load("./data/all_levels_onehot.npz")["levels"]
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors.to(device), test_tensors.to(device)


class VAEMario(nn.Module):
    """
    A VAE that decodes to the Categorical distribution
    on "sentences" of shape (h, w).
    """
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super(VAEMario, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
        ).to(self.device)

        # The VAE prior on latent codes. Only used for the KL term in
        # the ELBO loss.
        self.p_z = Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

        self.train_data, self.test_data = load_data(device=self.device)

        # print(self)

    def encode(self, x: torch.Tensor) -> Normal:
        """
        An encoding function that returns the normal distribution
        q(z|x) for some data x.

        It flattens x after the first dimension, passes it through
        the encoder networks which parametrize the mean and log-variance
        of the Normal, and returns the distribution.
        """
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        """
        A decoding function that returns the Categorical distribution
        p(x|z) for some latent codes z.

        It passes it through the decoder network, which parametrizes
        the logits of the Categorical distribution of shape (h, w).
        """
        logits = self.decoder(z)
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: torch.Tensor) -> Tuple[Normal, Categorical]:
        """
        A forward pass for some data x, returning the tuple
        [q(z|x), p(x|z)] where the latent codes in the second
        distribution are sampled from the first one.
        """
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: torch.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> torch.Tensor:
        """
        The ELBO (Evidence Lower Bound) loss for the VAE,
        which is a linear combination of the reconconstruction
        loss (i.e. the negative log likelihood of the data), plus
        a Kullback-Leibler regularization term which shapes the
        approximate posterior q(z|x) to be close to the prior p(z), 
        which we take as the unit Gaussian in latent space.
        """
        x_ = x.to(self.device).argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def plot_grid(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        sample=False,
        ax=None,
    ) -> np.ndarray:
        """
        A helper function which plots, as images, the levels in a
        fine grid in latent space, specified by the provided limits,
        number of rows and number of columns.

        The figure can be plotted in a given axis; if none is passed,
        a new figure is created.

        This function also returns the final image (which is the result
        of concatenating all the individual decoded images) as a numpy
        array.
        """
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        images_dist = self.decode(torch.from_numpy(zs).type(torch.float))
        if sample:
            images = images_dist.sample()
        else:
            images = images_dist.probs.argmax(dim=-1)

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = 16 * 14
        final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
        for z, (i, j) in positions.items():
            final_img[
                i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels
            ] = img_dict[z]

        final_img = final_img.astype(int)

        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        return final_img


if __name__ == "__main__":
    vae = VAEMario()
    print(vae)
