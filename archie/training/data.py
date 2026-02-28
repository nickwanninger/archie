from datasets import load_dataset, interleave_datasets
import torch
from torch.utils.data import IterableDataset, DataLoader
import sqlite3
import re

from archie.config import Config

from itertools import cycle
from datasets import load_dataset


def init_seen_db(db_path):
    """Create the seen-documents database and table. Call once from the main process."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS seen (uuid TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()


class SeenDB:
    """Tracks seen document UUIDs in a SQLite database to avoid training on duplicates."""

    BATCH_SIZE = 256

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.pending = []

    def _ensure_conn(self):
        if self.conn is not None:
            return
        self.conn = sqlite3.connect(self.db_path, timeout=60)

    def _flush(self):
        if not self.pending:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO seen (uuid) VALUES (?)",
            [(u,) for u in self.pending],
        )
        self.conn.commit()
        self.pending.clear()

    def check_and_add(self, uuid):
        """Returns True if the UUID was already seen, False if it's new (and queues it for insert)."""
        self._ensure_conn()
        row = self.conn.execute(
            "SELECT 1 FROM seen WHERE uuid = ?", (uuid,)
        ).fetchone()
        if row:
            return True
        self.pending.append(uuid)
        if len(self.pending) >= self.BATCH_SIZE:
            self._flush()
        return False

    def close(self):
        if self.conn:
            self._flush()
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

        # Buffer to accumulate tokens across documents
        token_buffer = []
        offset = 0
        chunk_size = self.config.max_seq_len + 1
        trim_every = 100_000  # amortize the list copy cost

        for example in dataset:
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
