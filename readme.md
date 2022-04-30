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
