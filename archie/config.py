from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 24
    n_kv_heads: int = 4

    # GPT-2 rounded up to a multiple of 64 for efficiency + special tokens down the line.
    vocab_size: int = 50304
    # For SwiGLU alignment
    multiple_of: int = 256
    norm_eps: float = 1e-5

    max_seq_len: int = 1024
    rope_theta: float = 10000.0

    block_size: int = 1024
    batch_size: int = 10

    device: str = "cuda"

    def to_name(self):
        return f"{self.dim}d_{self.n_layers}l_{self.n_heads}h_{self.n_kv_heads}kv"

    def get_checkpoint_dir(self):
        return Path("model") / Path(self.to_name())
