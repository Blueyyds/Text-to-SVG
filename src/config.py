"""
Configuration for Text to SVG Generation
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Model settings
    text_encoder_name: str = "openai/clip-vit-large-patch14"  # Text encoder
    image_encoder_name: str = "openai/clip-vit-large-patch14"  # Image encoder
    decoder_layers: int = 6
    hidden_dim: int = 768
    num_heads: int = 12

    # SVG generation
    max_svg_length: int = 512
    svg_vocab_size: int = 1000  # Will be determined by tokenizer

    # RL settings
    gamma: float = 0.99
    lambda_gae: float = 0.95
    ppo_clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Training settings
    batch_size: int = 32
    learning_rate: float = 3e-5
    num_epochs: int = 50

    # Data
    data_dir: str = "data/"
    output_dir: str = "outputs/"
