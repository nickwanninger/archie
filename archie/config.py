from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    name: str = "unknown"

    # GPT-2 rounded up to a multiple of 64 for efficiency + special tokens down the line.
    vocab_size: int = 50304
    device: str = "cuda"

    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4

    d_ff: int = 5504  # ??

    max_seq_len: int = 2048
    dropout: float = 0.0

    def to_name(self):
        return f"{self.dim}d_{self.n_layers}l_{self.n_heads}h_{self.n_kv_heads}kv"

    def get_checkpoint_dir(self):
        return Path("models") / Path(self.name)


# Smaller model.
flicker = Config(
    name="flicker",
    d_model=1024,
    n_layers=24,
    n_heads=24,
    n_kv_heads=4,
    d_ff=6144,
    # max_seq_len=1024,
    max_seq_len=4096,
    dropout=0.0,
)

# Medium 1.2b config.
glimmer = Config(
    name="glimmer",
    d_model=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    d_ff=5504,
    max_seq_len=2048,
    dropout=0.0,
)

# Larger 7b config.
blaze = Config(
    name="blaze",
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    d_ff=11008,
    # Should we grow this?
    max_seq_len=2048,
    dropout=0.0,
)
