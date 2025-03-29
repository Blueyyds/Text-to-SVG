"""
Reward computation for reinforcement learning
"""

import torch
import torch.nn.functional as F


class SVGReward:
    """Computes rewards for generated SVGs based on:
    1. Visual similarity to text description (using CLIP)
    2. SVG validity
    3. Complexity and size optimization
    """

    def __init__(self, config, agent):
        self.config = config
        self.agent = agent

    def compute_reward(self, text_prompts, generated_svgs):
        rewards = []

        for prompt, svg in zip(text_prompts, generated_svgs):
            reward = 0.0

            # Check SVG validity
            try:
                image = self.agent.render_svg(svg)
                if image is None:
                    # Invalid SVG
                    reward -= 1.0
                else:
                    # Valid SVG - compute image-text similarity using CLIP
                    with torch.no_grad():
                        # Encode image
                        image_tensor = self.agent.image_encoder.preprocess(image).unsqueeze(0)
                        image_features = self.agent.image_encoder(image_tensor).image_features

                        # Encode text
                        text_features = self.agent.encode_text([prompt])

                        # Compute similarity
                        similarity = F.cosine_similarity(image_features, text_features.mean(dim=1))
                        reward += similarity.item() * 2.0

                    # Reward for compact SVG (avoid unnecessarily complex SVGs)
                    svg_length = len(svg)
                    if svg_length < 200:
                        reward += 0.5
                    elif svg_length > 1000:
                        reward -= 0.5
            except:
                # Invalid SVG
                reward -= 1.0

            rewards.append(reward)

        return torch.tensor(rewards)
