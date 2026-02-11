from datasets import load_dataset, interleave_datasets
import torch
from torch.utils.data import IterableDataset, DataLoader

from archie.config import Config


def get_datasets():
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="CC-MAIN-2024-10",
        split="train",
        streaming=True,
    )

    wikipedia = load_dataset(
        "wikimedia/wikipedia",
        name="20231101.en",
        split="train",
        streaming=True,
    )

    dataset = interleave_datasets([fineweb, wikipedia], probabilities=[0.8, 0.2])
    dataset = dataset.shuffle(buffer_size=40000)
    return dataset


class TextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, config: Config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.eot = tokenizer._special_tokens.get("<|endoftext|>", config.vocab_size - 1)

    def __iter__(self):
        # Buffer to accumulate tokens across documents
        token_buffer = []

        # Iterate through the dataset
        for example in self.dataset:
            # Assuming your dataset has a 'text' field
            # Adjust the field name based on your dataset
            text = example["text"]

            # Tokenize
            tokens = self.tokenizer.encode(text)

            # Add tokens to buffer with EOT separator
            token_buffer.extend(tokens)
            token_buffer.append(self.eot)

            # Yield chunks of max_seq_len + 1 (for input and target)
            while len(token_buffer) >= self.config.max_seq_len + 1:
                # Extract chunk
                chunk = token_buffer[: self.config.max_seq_len + 1]
                token_buffer = token_buffer[self.config.max_seq_len + 1 :]

                # Split into input (x) and target (y)
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)

                yield x, y
