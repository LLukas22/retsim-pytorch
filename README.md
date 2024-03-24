# retsim-pytorch
[![PyPI Version](https://img.shields.io/pypi/v/retsim-pytorch.svg)](https://pypi.org/project/retsim-pytorch)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/retsim-pytorch.svg)](https://pypi.org/project/retsim-pytorch)

Welcome to `retsim-pytorch`, the PyTorch adaptation of Google's [RETSim](https://arxiv.org/abs/2311.17264) (Resilient and Efficient Text Similarity) model, which is part of the [UniSim (Universal Similarity)](https://github.com/google/unisim) framework.

This model is designed for efficient and accurate multilingual fuzzy string matching, near-duplicate detection, and assessing string similarity. For more information, please refer to the [UniSim documentation](https://github.com/google/unisim).

## Installation

You can easily install `retsim-pytorch` via pip:

```shell
pip install retsim-pytorch
```

## Usage

You can configure the model using the `RETSimConfig` class. By default, it utilizes the same configuration as the original UniSim model. If you wish to use the same weights as the original Google model, you can download a SafeTensors port of the weights [here](./weights/model.safetensors).

Here's how to use the model in your code:

```python
import torch
from retsim_pytorch import RETSim, RETSimConfig
from retsim_pytorch.preprocessing import binarize

# Configure the model
config = RETSimConfig()
model = RETSim(config)

# Prepare and run inference
binarized_inputs, chunk_ids = binarize(["hello world"])
embedded, unpooled = model(torch.tensor(binarized_inputs))
```