# A minimal example of a VAE for Mario

This folder contains three scripts: `vae.py` implements a categorical VAE with MLP encoders and decoders, `train.py` trains it, and `visualize.py` shows a snapshot of the latent space.

## Installing requirements

Create an environment using the tool of your choice. Python version `>=3.9`. Then do

```
pip install -r requirements.txt
```

## Visualizing a pretrained model

I added an already-trained model under `models/example.pt`. Run

```
python visualize.py
```

to get a look at this example's latent space.

## Using the simulator

I recently added a `simulator.jar` that lets you run levels directly from latent space. To do so, you'll need a version of Java that is above 8 (we used `OpenJDK 15.0.2`). Running

```
python simulator.py
```

should let you play content directly from latent space. Take a look at the functions implemented therein and get creative! The simulator outputs a JSON with telemetrics from the simulation, and if you set `human_player=False` it uses Robin Baumgarten's A star agent.

### Troubleshooting in remote servers/containers

If you are working on a remote server/dev container, you might face the following issue:

```
Exception in thread "main" java.awt.HeadlessException: 
No X11 DISPLAY variable was set,
but this program performed an operation which requires it.
```

This happens because the Java simulator is expecting a screen! A way to solve this is by installing a virtual frame buffer. In Ubuntu, this can be achieved with

```
# Install a virtual frame buffer
apt-get install xvfb
```

and then run

```
# Export a display enviroment variable
export DISPLAY=99

# Run the virtual frame buffer in the background.
Xvfb :99 -screen 0 1366x768x24 -ac +extension GLX +render -noreset &
```

## Using the simulator with Docker

We provide a `dockerfile` with the lightweight requirements for running `simulator.py`. This docker image builds on the Ubuntu 20.04 base, adding Python 3.9, Java 17, and installing `Xvfb` to be able to run `simulator.jar`. This image is already on `docker hub`, so you can either

```bash
docker pull miguelgondu/mario
docker run miguelgondu/mario
```

**or**

```bash
docker build -t mario .
docker run mario
```

## Running Bayesian Optimization in latent space

I also include an example of how to run Bayesian Optimization in the latent space of the VAE. In it, I try to maximize the number of jumps. It is built using `gpytorch` and `botorch`, and you can play around with your GP definition in `simple_bayesian_optimization.py`. To run it, just call

```
python simple_bayesian_optimization.py
```

## Cite this repository!

If you find this code useful, feel free to cite it!

```bibtex
@software{Gonzalez-Duque:Minimal_VAEMario:2023,
author = {González-Duque, Miguel},
title = {{Minimal implementation of a Variational Autoencoder on Super Mario Bros}},
url = {https://github.com/miguelgondu/minimal_VAE_on_Mario},
version = {0.1},
date = {2023-04-29}
}
```
