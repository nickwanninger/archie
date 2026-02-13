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
        dataset = self.dataset
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        # Buffer to accumulate tokens across documents
        token_buffer = []
        offset = 0
        chunk_size = self.config.max_seq_len + 1
        trim_every = 100_000  # amortize the list copy cost

        for example in dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)
            token_buffer.append(self.eot)

            while len(token_buffer) - offset >= chunk_size:
                chunk = token_buffer[offset : offset + chunk_size]
                offset += chunk_size

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

                # Periodically trim consumed tokens to keep buffer from growing unbounded
                if offset >= trim_every:
                    token_buffer = token_buffer[offset:]
                    offset = 0
