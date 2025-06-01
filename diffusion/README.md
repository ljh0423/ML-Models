# DDPM (Denoising Diffusion Probabilistic Models) on MNIST

This repository implements a UNet-based Denoising Diffusion Probabilistic Model (DDPM) for generating MNIST digits. It includes a custom UNet architecture with optional attention, a variance scheduler, training loop with Exponential Moving Average (EMA), and inference for sample generation.

## Features

- Custom UNet with:
  - Residual blocks
  - Multi-head self-attention (optional)
  - Sinusoidal time embeddings
- Linear variance scheduler for noise levels
- Exponential Moving Average (EMA) for model stabilization
- Checkpointing and resume support
- Inference pipeline with visualization of denoising steps

## Installation

Install the required Python packages:

```bash
pip install torch torchvision timm einops matplotlib tqdm
```

Training
To train the DDPM model:
```python
from main import train
train(checkpoint_path=None)  # Or provide a checkpoint path to resume
```

Inference
To generate samples using a trained model:
```python
from main import inference
inference(checkpoint_path='checkpoints/ddpm_checkpoint')
```

Notes
The model is trained on MNIST images padded to 32×32 resolution.

Default setup uses 1000 diffusion steps and EMA decay of 0.9999.

Intermediate denoising steps are visualized during inference.
