"""
SVG Tokenizer for handling the conversion between SVG code and token IDs
"""

import os


class SVGTokenizer:
    """Tokenizer for SVG code - handles conversion between SVG code and token IDs"""

    def __init__(self, vocab_file=None):
        # If vocab file exists, load it, otherwise create a new vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        # Special tokens
        self.add_special_tokens(["<PAD>", "<SOS>", "<EOS>", "<UNK>"])

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def add_special_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.vocab_size
            self.id_to_token[self.vocab_size] = token
            self.vocab_size += 1

    def tokenize(self, svg_code):
        # Simple character-level tokenization for now
        # In a more advanced implementation, we would tokenize SVG elements and attributes
        tokens = []
        for char in svg_code:
            if char not in self.token_to_id:
                self.add_token(char)
            tokens.append(self.token_to_id[char])
        return tokens

    def decode(self, token_ids):
        return "".join([self.id_to_token.get(id, "<UNK>") for id in token_ids])

    def save_vocab(self, vocab_file):
        with open(vocab_file, "w") as f:
            for token, id in self.token_to_id.items():
                f.write(f"{token}\t{id}\n")

    def load_vocab(self, vocab_file):
        self.token_to_id = {}
        self.id_to_token = {}
        with open(vocab_file, "r") as f:
            for line in f:
                token, id = line.strip().split("\t")
                id = int(id)
                self.token_to_id[token] = id
                self.id_to_token[id] = token
        self.vocab_size = len(self.token_to_id)
