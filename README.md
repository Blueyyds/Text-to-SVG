# Text to SVG Generator

A system that generates SVG code from text descriptions using multimodal agents trained with reinforcement learning techniques.

## Project Structure

The project has been organized into modular components:

```
src/
├── __init__.py        # Package exports
├── config.py          # Configuration settings
├── tokenizer.py       # SVG tokenization utilities
├── models.py          # Neural network model implementations
├── reward.py          # Reward computation for RL
├── trainer.py         # PPO training implementation
├── dataset.py         # Dataset handling
├── train.py           # Training functions
├── test.py            # Testing/inference functions
└── main.py            # Main entry point
```

## Dependencies

The project requires the following dependencies:

- torch
- transformers
- cairosvg
- numpy
- PIL (Pillow)

These are listed in the pyproject.toml file.

## Usage

### Training

To train the model:

```python
from src.config import Config
from src.train import train

config = Config()
train(config)
```

### Inference

To generate SVGs from text prompts using a trained model:

```python
from src.config import Config
from src.test import test_generation
import os

config = Config()
prompts = [
    "A simple red circle",
    "A blue square with a yellow border",
    "A landscape with mountains and a sun"
]

results = test_generation(config, os.path.join(config.output_dir, "checkpoint_45.pt"), prompts)
```

## Model Architecture

The system utilizes:

- CLIP text/image encoders for text understanding and reward computation
- A transformer decoder for generating SVG tokens
- Proximal Policy Optimization (PPO) for reinforcement learning
- A reward model that evaluates generation quality based on visual similarity to the text description
