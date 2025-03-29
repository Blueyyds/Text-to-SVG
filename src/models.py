"""
Neural network models for Text to SVG generation
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import cairosvg
import io

from .tokenizer import SVGTokenizer


class SVGTextDecoder(nn.Module):
    """Transformer decoder for generating SVG tokens from text embeddings"""

    def __init__(self, config, svg_vocab_size):
        super().__init__()
        self.config = config

        # Embedding layer for SVG tokens
        self.token_embedding = nn.Embedding(svg_vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_svg_length, config.hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim, nhead=config.num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, svg_vocab_size)

    def forward(self, text_embeddings, svg_tokens, attention_mask=None):
        # Get positions
        positions = torch.arange(svg_tokens.size(1), device=svg_tokens.device).unsqueeze(0)
        positions = positions.expand(svg_tokens.size(0), -1)

        # Embed tokens and positions
        token_embeddings = self.token_embedding(svg_tokens)
        pos_embeddings = self.position_embedding(positions)

        # Combine embeddings
        embeddings = token_embeddings + pos_embeddings

        # Create casual mask for autoregressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(svg_tokens.size(1), device=svg_tokens.device)

        # Decode
        decoder_output = self.transformer_decoder(
            embeddings,
            text_embeddings,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=attention_mask,
            memory_key_padding_mask=None,
        )

        # Project to vocabulary
        output = self.output_projection(decoder_output)
        return output


class MultimodalRLAgent(nn.Module):
    """Main agent class that combines text understanding and SVG generation"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize text encoder (CLIP)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)

        # Initialize SVG tokenizer
        self.svg_tokenizer = SVGTokenizer()
        config.svg_vocab_size = self.svg_tokenizer.vocab_size

        # Initialize image encoder for reward computation
        self.image_encoder = AutoModel.from_pretrained(config.image_encoder_name)

        # SVG decoder
        self.svg_decoder = SVGTextDecoder(config, self.svg_tokenizer.vocab_size)

        # Value head for RL
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def encode_text(self, text_prompts):
        # Tokenize and encode text
        inputs = self.text_tokenizer(text_prompts, padding=True, truncation=True, return_tensors="pt").to(
            next(self.text_encoder.parameters()).device
        )

        with torch.no_grad():
            text_features = self.text_encoder(**inputs).last_hidden_state

        return text_features

    def forward(self, text_prompts, svg_tokens=None, temperature=1.0):
        # Encode text
        text_embeddings = self.encode_text(text_prompts)

        # If SVG tokens are provided, get next token predictions
        if svg_tokens is not None:
            logits = self.svg_decoder(text_embeddings, svg_tokens)

            # Get value prediction for RL
            value = self.value_head(text_embeddings.mean(dim=1))

            return logits, value

        # Otherwise, generate SVG tokens auto-regressively
        else:
            batch_size = len(text_prompts)
            device = next(self.parameters()).device

            # Start with SOS token
            current_ids = torch.ones(batch_size, 1).long().to(device) * self.svg_tokenizer.token_to_id["<SOS>"]

            generated_ids = []
            for i in range(self.config.max_svg_length - 1):
                # Get predictions
                logits, _ = self.forward(text_prompts, current_ids)

                # Get predicted token (from last position only)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

                # Add predicted token to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated_ids.append(next_token)

                # Stop if all sequences have EOS
                if all([(self.svg_tokenizer.token_to_id["<EOS>"] in ids) for ids in current_ids]):
                    break

            return current_ids

    def generate_svg(self, text_prompt, temperature=1.0):
        # Set model to eval mode
        self.eval()

        with torch.no_grad():
            token_ids = self.forward([text_prompt], temperature=temperature)[0]

        # Convert to SVG code
        svg_code = self.svg_tokenizer.decode(token_ids.cpu().numpy())

        # Cleanup - remove everything after EOS token
        if "<EOS>" in svg_code:
            svg_code = svg_code[: svg_code.index("<EOS>")]

        # Remove SOS token
        svg_code = svg_code.replace("<SOS>", "")

        return svg_code

    def render_svg(self, svg_code):
        """Convert SVG code to a PIL Image for visualization"""
        try:
            png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
            image = Image.open(io.BytesIO(png_data))
            return image
        except Exception as e:
            print(f"Error rendering SVG: {e}")
            return None
