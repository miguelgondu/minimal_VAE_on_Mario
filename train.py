"""
This script trains a VAEMario using early stopping.
"""
from time import time

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from vae import VAEMario, load_data


def fit(
    model: VAEMario,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
) -> torch.Tensor:
    """
    Runs a training epoch: evaluating the model in
    the data provided by the data_loader, computing
    the ELBO loss inside the model, and propagating
    the error backwards to the parameters.
    """
    model.train()
    running_loss = 0.0
    for (levels,) in data_loader:
        levels = levels.to(device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(levels)
        loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)


def test(
    model: VAEMario,
    test_loader: DataLoader,
    device: str,
    epoch: int = 0,
) -> torch.Tensor:
    """
    Evaluates the current model on the test set,
    returning the average loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (levels,) in test_loader:
            levels.to(device)
            q_z_given_x, p_x_given_z = model.forward(levels)
            loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_loader)}")
    return running_loss / len(test_loader)


def run(
    max_epochs: int = 500,
    batch_size: int = 64,
    lr: int = 1e-3,
    save_every: int = None,
    overfit: bool = False,
):
    """
    Trains a VAEMario on the dataset for the provided hyperparameters.

    This training uses early stopping with a patience of 25 epochs,
    by which we mean that we maintain the model with lowest test loss
    and, if we don't see any improvement on it for 25 epochs in a row,
    we stop the training. The model can be forced to overfit if you
    pass overfit=True.
    """
    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    comment = f"{timestamp}_mariovae"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    training_tensors, test_tensors = load_data()

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loading the model and optimizer
    print("Model:")
    vae = VAEMario()
    print(vae)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training and testing.
    print(f"Training experiment {comment}")
    best_loss = np.Inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        _ = fit(vae, optimizer, data_loader, device)
        test_loss = test(vae, test_loader, device, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            torch.save(vae.state_dict(), f"./models/{comment}_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        if save_every is not None and epoch % save_every == 0 and epoch != 0:
            # Saving the model
            print(f"Saving the model at checkpoint {epoch}.")
            torch.save(vae.state_dict(), f"./models/{comment}_epoch_{epoch}.pt")

        # Early stopping:
        if n_without_improvement == 25:
            print("Stopping early")
            break


if __name__ == "__main__":
    run()
