"""
Dataset for text-to-SVG examples
"""


class Text2SVGDataset:
    """Dataset for text-to-SVG examples"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.examples = []
        self.load_data()

    def load_data(self):
        # Load data from files
        # In actual implementation, this would load paired text-SVG examples
        pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        # Collate function for DataLoader
        text_prompts = [item["text"] for item in batch]
        svg_codes = [item["svg"] for item in batch]
        return {"text_prompts": text_prompts, "svg_codes": svg_codes}
