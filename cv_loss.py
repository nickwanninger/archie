import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from rich.console import Console
import numpy as np

# Import from the existing model file
from mistral_model import (
    MistralLite,
    MODEL_ARGS,
    TextDataset,
    get_tokenizer,
    get_device,
    load_model_from_checkpoint,
)


def compute_cv_loss(num_steps=500):
    console = Console()
    device = get_device()
    console.print(f"[bold]Using device:[/] {device}")

    # 1. Load Model
    # Try to load latest checkpoint, else init fresh
    try:
        model, tokenizer = load_model_from_checkpoint(device=device)
        console.print("[green]Loaded latest checkpoint.[/]")
    except Exception as e:
        console.print(
            f"[yellow]Could not load checkpoint ({e}), initializing fresh model for demonstration...[/]"
        )
        tokenizer = get_tokenizer()
        model = MistralLite(MODEL_ARGS).to(device).to(torch.bfloat16)

    model.eval()

    # 2. Setup Dataset
    # User requested running over the same dataset for now to see the mechanism
    console.print("[bold]Loading datasets...[/]")
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

    # Use the same interleaving
    dataset = interleave_datasets([fineweb, wikipedia], probabilities=[0.8, 0.2])

    # Create TextDataset
    # Note: TextDataset uses sliding window of block_size
    text_dataset = TextDataset(dataset, tokenizer, MODEL_ARGS)

    val_loader = DataLoader(
        text_dataset,
        batch_size=MODEL_ARGS.batch_size,
        num_workers=4,
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = 0

    console.print(f"[bold]Starting validation loop for {num_steps} steps...[/]")

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_steps:
                break

            x, y = x.to(device), y.to(device)

            logits = model(x)
            # Reshape for loss: (Batch * SeqLen, Vocab) vs (Batch * SeqLen)
            loss = criterion(logits.reshape(-1, MODEL_ARGS.vocab_size), y.reshape(-1))

            total_loss += loss.item()
            total_steps += 1

            if i % 5 == 0:
                console.print(f"Step {i}: Loss {loss.item():.4f}")

    if total_steps > 0:
        avg_loss = total_loss / total_steps
        ppl = np.exp(avg_loss)
        console.print(f"\n[bold green]Validation Complete[/]")
        console.print(f"Average Loss: {avg_loss:.4f}")
        console.print(f"Perplexity: {ppl:.2f}")
    else:
        console.print("[red]No steps computed.[/]")


if __name__ == "__main__":
    compute_cv_loss()
