from datasets import load_dataset, interleave_datasets
import torch
from torch.utils.data import IterableDataset, DataLoader
import sqlite3
import re

from archie.config import Config

from itertools import cycle
from datasets import load_dataset


class SeenDB:
    """Tracks seen document UUIDs in a SQLite database to avoid training on duplicates."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def _ensure_conn(self):
        if self.conn is not None:
            return
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (uuid TEXT PRIMARY KEY)")
        self.conn.commit()

    def check_and_add(self, uuid):
        """Returns True if the UUID was already seen, False if it's new (and inserts it)."""
        self._ensure_conn()
        try:
            self.conn.execute("INSERT INTO seen (uuid) VALUES (?)", (uuid,))
            self.conn.commit()
            return False
        except sqlite3.IntegrityError:
            return True

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


weights = {
    "High-Quality": 0.6,
    "Medium-High-Quality": 0.3,
    "High-Quality-Synthetic": 0.1,
}

def get_datasets():

    sets = []
    probs = []

    for subset, weight in weights.items():
        ds = load_dataset(
            "nvidia/Nemotron-CC-v2.1",
            name=subset,
            split="train",
            streaming=True
        )
        # Keep only the text column - subsets have different metadata schemas
        ds = ds.select_columns(["text", "uuid"])
        sets.append(ds)
        probs.append(weight)

    dataset = interleave_datasets(
        sets,
        probabilities=probs,
        stopping_strategy="first_exhausted",  # or "all_exhausted" for non-streaming
    )

    return dataset


class TextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, config: Config, db_path=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.db_path = db_path
        self.eot = tokenizer._special_tokens.get("<|endoftext|>", config.vocab_size - 1)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.dataset
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        seen_db = SeenDB(self.db_path) if self.db_path else None

        # Buffer to accumulate tokens across documents
        token_buffer = []
        offset = 0
        chunk_size = self.config.max_seq_len + 1
        trim_every = 100_000  # amortize the list copy cost

        for example in dataset:
            if seen_db and seen_db.check_and_add(example["uuid"]):
                continue

            text = example["text"]
            text = text.replace("\r\n", "\n").replace(
                "\r", "\n"
            )  # normalize line endings
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            token_buffer.extend(tokens)
            token_buffer.append(self.eot)

            while len(token_buffer) - offset >= chunk_size:
                chunk = token_buffer[offset : offset + chunk_size]
                offset += self.config.max_seq_len

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

                # Periodically trim consumed tokens to keep buffer from growing unbounded
                if offset >= trim_every:
                    token_buffer = token_buffer[offset:]
                    offset = 0
