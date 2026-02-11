import archie
import archie.training
import archie.config
import uuid

import torch
import torchinfo
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datetime import datetime

import sys
import math

import logging
from rich.logging import RichHandler

import jsonlines


config = archie.config.glow

checkpoint_dir = config.get_checkpoint_dir()
model_path = checkpoint_dir / "model.pt"
training_log_path = checkpoint_dir / "training.jsonl"
debug_log_path = checkpoint_dir / "archie.log"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("archie")


console_handler = RichHandler(level=logging.DEBUG)
log.addHandler(console_handler)


formatter = logging.Formatter("{asctime} [{levelname}] {name}: {message}", style="{")

file_handler = logging.FileHandler(debug_log_path)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

log.setLevel(logging.DEBUG)


# Helper stuff!
def enable_gradient_checkpointing(model):
    for layer in model.layers:
        layer._forward = layer.forward
        layer.forward = lambda x, m=layer: torch.utils.checkpoint.checkpoint(
            m._forward, x, use_reentrant=False
        )


# Learning rate scheduler (cosine with warmup)
def get_lr(step, warmup_steps=100, max_steps=100000):  # Changed from 2000 to 100
    if step < warmup_steps:
        return 3e-4 * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 3e-4 * 0.5 * (1 + math.cos(progress * math.pi))


@torch.no_grad()
def generate_text(model, tokenizer, prompt="The", max_tokens=50, temperature=0.8):
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt)]).to(model.config.device)

    for _ in range(max_tokens):
        # Forward pass
        logits, _ = model(tokens)

        # Get logits for last token
        logits = logits[:, -1, :] / temperature

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop at max sequence length
        if tokens.shape[1] >= model.config.max_seq_len:
            break

    model.train()
    return tokenizer.decode(tokens[0].cpu().tolist())


run_id = str(uuid.uuid4())
log.info(f"Starting run {run_id}")

log.info("Allocating model...")

# Create the model.

model = archie.ArchieModel(config).to(config.device).to(torch.bfloat16)
enable_gradient_checkpointing(model)

torchinfo.summary(model)

tokenizer = archie.get_tokenizer()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.1
)

global_step = 0
tokens_seen = 0

batch_size = 6
accumulation_steps = 8  # batch_size=4 → effective batch_size=32


# Load the model
if model_path.exists():
    log.info("Loading state...")
    state = torch.load(model_path, map_location=config.device)
    log.info("Loading model...")
    model.load_state_dict(state["model"])
    log.info("Loading optimizer...")
    optimizer.load_state_dict(state["optimizer"])
    global_step = state["step"]
    tokens_seen = state["tokens_seen"]
    log.info("Done.")
else:
    log.info("Model not found, starting from scratch.")


dataset = archie.training.get_datasets()
text_dataset = archie.training.TextDataset(dataset, tokenizer, config)
train_loader = DataLoader(
    text_dataset,
    batch_size=batch_size,
    num_workers=0,
    # prefetch_factor=64,  # Prefetch
    pin_memory=True,  # Optimization
)


total_params = sum(p.numel() for p in model.parameters())


def append_log_entry(data):
    data["run_id"] = run_id
    # log.info(data)
    with jsonlines.open(training_log_path, mode="a") as writer:
        writer.write(data)


log.info("Starting training...")


@torch.compile
def train_step(x, y):
    logits, loss = model(x, labels=y)
    loss = loss / accumulation_steps
    loss.backward()
    return loss


for i, (x, y) in enumerate(train_loader):
    x = x.to(config.device)
    y = y.to(config.device)

    # Track tokens (batch_size * seq_len)
    tokens_seen += x.numel()

    # Forward + backward
    # logits, loss = model_opt(x, labels=y)
    # loss = loss / accumulation_steps
    # loss.backward()
    loss = train_step(x, y)

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        global_step += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()

        lr = get_lr(global_step)
        # lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        effective_loss = loss.item() * accumulation_steps

        # Logging
        perplexity = math.exp(effective_loss)
        tokens_M = tokens_seen / 1_000_000
        tokens_per_param = tokens_seen / total_params
        log.info(
            f"Step {global_step} | Tokens: {tokens_M:.2f}M ({tokens_per_param:.3f}/param) | PPL: {perplexity:9.2f} | loss: {effective_loss:.2f} | LR: {lr:.2e}"
        )

        if global_step % 10 == 0:
            append_log_entry(
                {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "tokens_per_param": tokens_per_param,
                    "perplexity": perplexity,
                    "loss": effective_loss,
                    "lr": lr,
                    "datetime": datetime.utcnow().isoformat(),
                }
            )

        if global_step % 250 == 0:
            log.info("Saving model...")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                },
                model_path,
            )
            log.info("Done.")
