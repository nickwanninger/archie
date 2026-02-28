import tomllib
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_PATH = Path(__file__).parent / "configs.toml"


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

    d_ff: int = 5504

    max_seq_len: int = 2048
    dropout: float = 0.0

    num_experts: int | None = None
    top_k: int = 2
    aux_loss_weight: float = 0.01

    def to_name(self):
        return f"{self.dim}d_{self.n_layers}l_{self.n_heads}h_{self.n_kv_heads}kv"

    def get_checkpoint_dir(self):
        return Path("models") / Path(self.name)


def _load_all() -> dict[str, dict]:
    with open(_CONFIGS_PATH, "rb") as f:
        return tomllib.load(f).get("model", {})


def load(name: str) -> Config:
    models = _load_all()
    if name not in models:
        raise ValueError(f"Unknown config: {name!r}. Available: {list(models)}")
    return Config(name=name, **models[name])


def list_configs() -> list[str]:
    return list(_load_all().keys())
