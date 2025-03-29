"""
Testing functions for Text to SVG model
"""

import os
import torch

from .config import Config
from .models import MultimodalRLAgent


def test_generation(config, model_path, prompts):
    """Test SVG generation with trained model"""
    # Load model
    agent = MultimodalRLAgent(config)
    checkpoint = torch.load(model_path)
    agent.load_state_dict(checkpoint["model_state_dict"])

    # Generate SVGs for each prompt
    results = []
    for prompt in prompts:
        svg_code = agent.generate_svg(prompt)
        svg_image = agent.render_svg(svg_code)

        results.append({"prompt": prompt, "svg_code": svg_code, "image": svg_image})

        # Save SVG
        with open(os.path.join(config.output_dir, f"test_{prompt[:20]}.svg"), "w") as f:
            f.write(svg_code)

    return results
