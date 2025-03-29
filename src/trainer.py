"""
Proximal Policy Optimization (PPO) trainer implementation
"""

import torch
import torch.nn.functional as F


class PPOTrainer:
    """Trains the agent using Proximal Policy Optimization (PPO)"""

    def __init__(self, config, agent, reward_model):
        self.config = config
        self.agent = agent
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    def compute_advantages(self, rewards, values, masks):
        """Compute GAE advantages"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * masks[t] - values[t]
            last_gae_lam = delta + self.config.gamma * self.config.lambda_gae * masks[t] * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        return advantages, returns

    def train_batch(self, text_prompts, reference_svgs=None):
        # Set agent to training mode
        self.agent.train()

        # Step 1: Generate SVGs and collect data
        with torch.no_grad():
            generated_token_ids = self.agent.forward(text_prompts)
            generated_svgs = [self.agent.svg_tokenizer.decode(ids.cpu().numpy()) for ids in generated_token_ids]

            # Compute rewards
            rewards = self.reward_model.compute_reward(text_prompts, generated_svgs)

            # Get old action probabilities and values
            old_logits, old_values = self.agent.forward(text_prompts, generated_token_ids[:, :-1])
            old_probs = F.softmax(old_logits, dim=-1)

            # Create masks (1 for non-padded tokens)
            masks = (generated_token_ids != self.agent.svg_tokenizer.token_to_id["<PAD>"]).float()
            masks = masks[:, 1:]  # align with targets

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, old_values.squeeze(-1), masks)

        # Step 2: PPO update
        for _ in range(5):  # Multiple epochs per batch
            # Get new predictions
            new_logits, new_values = self.agent.forward(text_prompts, generated_token_ids[:, :-1])
            new_probs = F.softmax(new_logits, dim=-1)

            # Get targets (shifted right)
            targets = generated_token_ids[:, 1:]

            # Compute ratio for PPO
            ratio = torch.gather(new_probs, 2, targets.unsqueeze(-1)).squeeze(-1) / torch.gather(
                old_probs, 2, targets.unsqueeze(-1)
            ).squeeze(-1)

            # Compute policy loss
            clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
            policy_loss = (policy_loss * masks).sum() / masks.sum()

            # Compute value loss
            value_loss = F.mse_loss(new_values.squeeze(-1), returns)

            # Compute entropy for exploration
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=-1)
            entropy = (entropy * masks).sum() / masks.sum()

            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
            "reward": rewards.mean().item(),
            "generated_svgs": generated_svgs,
        }
