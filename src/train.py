"""
Training functions for Text to SVG model
"""

import os
import torch
import numpy as np

from .config import Config
from .models import MultimodalRLAgent
from .reward import SVGReward
from .trainer import PPOTrainer
from .dataset import Text2SVGDataset


def train(config):
    """Main training function"""
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize agent
    agent = MultimodalRLAgent(config)

    # Initialize reward model
    reward_model = SVGReward(config, agent)

    # Initialize trainer
    trainer = PPOTrainer(config, agent, reward_model)

    # Initialize dataset
    dataset = Text2SVGDataset(config.data_dir)

    # Training loop
    for epoch in range(config.num_epochs):
        # Sample batch
        batch_indices = np.random.choice(len(dataset), config.batch_size)
        batch = [dataset[i] for i in batch_indices]
        batch = Text2SVGDataset.collate_fn(batch)

        # Train on batch
        metrics = trainer.train_batch(batch["text_prompts"], batch["svg_codes"])

        # Log metrics
        print(f"Epoch {epoch}, Reward: {metrics['reward']}, Loss: {metrics['total_loss']}")

        # Save model checkpoint
        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                },
                os.path.join(config.output_dir, f"checkpoint_{epoch}.pt"),
            )

            # Save vocabulary
            agent.svg_tokenizer.save_vocab(os.path.join(config.output_dir, "svg_vocab.txt"))

            # Generate and save example SVGs
            for i, prompt in enumerate(batch["text_prompts"][:5]):
                svg_code = agent.generate_svg(prompt)
                with open(os.path.join(config.output_dir, f"example_{epoch}_{i}.svg"), "w") as f:
                    f.write(svg_code)
