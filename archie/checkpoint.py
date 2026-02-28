import torch


from archie.model import ArchieModel, ArchieMoEModel, create_model
from archie.config import Config
from archie.tokenizer import get_tokenizer


def load_model_from_checkpoint(config: Config, path=None):
    model = create_model(config).to(config.device).to(torch.bfloat16)
    tokenizer = get_tokenizer()

    if path is None:
        checkpoint_dir = config.get_checkpoint_dir()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint directory found at {checkpoint_dir}"
            )
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: int(p.stem))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
        path = checkpoints[-1]

    state = torch.load(path, map_location=config.device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer
