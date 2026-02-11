import torch
import torch.nn as nn
from dataclasses import dataclass
from mistral_model import MistralLite, ModelArgs, CHECKPOINT_PATH, console


def surgery():
    console.print("[bold red]Starting Model Surgery...[/]")

    # 1. Load the old checkpoint (Vocab 50257)
    if not CHECKPOINT_PATH.exists():
        console.print("[red]No checkpoint found![/]")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    old_state = checkpoint["model"]
    old_optimizer = checkpoint["optimizer"]

    # Get old embeddings
    old_emb = old_state["tok_embeddings.weight"]
    old_vocab_size, dim = old_emb.shape
    console.print(f"Old Vocab Size: {old_vocab_size}")

    if old_vocab_size >= 50304:
        console.print("[green]Model is already surgical! Aborting.[/]")
        return

    # 2. define New Size
    NEW_VOCAB_SIZE = 50304  # Multiple of 64
    console.print(
        f"New Vocab Size: {NEW_VOCAB_SIZE} (+{NEW_VOCAB_SIZE - old_vocab_size} tokens)"
    )

    # 3. Create New Embeddings
    # Initialize with the mean of the old embeddings to avoid shock
    emb_mean = old_emb.mean(dim=0, keepdim=True)

    new_emb = torch.zeros(NEW_VOCAB_SIZE, dim, dtype=old_emb.dtype)
    new_emb[:old_vocab_size] = old_emb
    new_emb[old_vocab_size:] = emb_mean  # Initialize new tokens as "average" tokens

    # 4. Patch State Dict
    # Update embeddings
    old_state["tok_embeddings.weight"] = new_emb
    # Update output layer (since weights are usually tied, but stored separately in state_dict)
    if "output.weight" in old_state:
        old_state["output.weight"] = new_emb

    console.print("[green]Patched Model Weights[/]")

    # 5. Patch Optimizer (AdamW momentum states)
    # The optimizer state is a dict of param_id -> state
    # We need to find which param_id corresponds to the embeddings.
    # Since we can't easily map ID back to name without the model, we can map by SHAPE.

    patched_params = 0
    for param_id, state in old_optimizer["state"].items():
        # Check if this parameter state matches the old vocab size
        if "exp_avg" in state:
            s_exp_avg = state["exp_avg"]
            if s_exp_avg.shape == (old_vocab_size, dim):
                console.print(f"Patching Optimizer State for Param ID {param_id}")

                # Create padded state
                new_exp_avg = torch.zeros(NEW_VOCAB_SIZE, dim, dtype=s_exp_avg.dtype)
                new_exp_avg[:old_vocab_size] = s_exp_avg
                state["exp_avg"] = new_exp_avg

                if "exp_avg_sq" in state:
                    s_exp_avg_sq = state["exp_avg_sq"]
                    new_exp_avg_sq = torch.zeros(
                        NEW_VOCAB_SIZE, dim, dtype=s_exp_avg_sq.dtype
                    )
                    new_exp_avg_sq[:old_vocab_size] = s_exp_avg_sq
                    state["exp_avg_sq"] = new_exp_avg_sq

                patched_params += 1

    if patched_params == 0:
        console.print(
            "[yellow]Warning: Could not find optimizer states to patch. Usually fine if starting fresh, but check if weight tying is confusing the optimizer.[/]"
        )
    else:
        console.print(f"[green]Patched {patched_params} Optimizer States[/]")

    # 6. Save back to disk
    torch.save(checkpoint, CHECKPOINT_PATH)
    console.print(f"[bold green]Surgery Complete! Saved to {CHECKPOINT_PATH}[/]")


if __name__ == "__main__":
    surgery()
