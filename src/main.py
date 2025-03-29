"""
Text to SVG Generation using Multimodal Agents and Reinforcement Learning

This project implements a system that generates SVG code from text descriptions
using multimodal agents trained with reinforcement learning techniques.
"""

import os
from .config import Config
from .train import train
from .test import test_generation


if __name__ == "__main__":
    # Initialize config
    config = Config()

    # Train model
    # train(config)

    # Test generation with sample prompts
    test_prompts = [
        "A simple red circle",
        "A blue square with a yellow border",
        "A landscape with mountains and a sun",
        "A cartoon character with a happy face",
        "A flowchart with three connected boxes",
    ]

    # For testing, uncomment this:
    # test_generation(config, os.path.join(config.output_dir, "checkpoint_45.pt"), test_prompts)
