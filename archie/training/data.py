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
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # When using multiple workers, we need to shard the dataset
            ds = self.dataset.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )
        else:
            ds = self.dataset

        buffer = []
        chunk_size = self.config.block_size + 1

        for entry in ds:
            text = entry["text"]
            text = text.replace("’", "'").replace("“", '"').replace("”", '"')
            tokens = self.tokenizer.encode_ordinary(text)
            tokens.append(self.eot)
            buffer.extend(tokens)

            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[self.config.block_size :]  # Sliding by block_size

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
