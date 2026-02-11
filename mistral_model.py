import csv
import random
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

import tiktoken
from datasets import load_dataset, interleave_datasets

from rich.console import Console
from rich.live import Live
from rich.table import Table

import math

import gzip
import requests

import archie
import archie.training

# For training

# For alignment (SFT)
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
# https://huggingface.co/datasets/DataMuncher-Labs/UltiMath

# For fine-tuning
# https://huggingface.co/datasets/Anthropic/hh-rlhf

console = Console()


MODEL_ARGS = archie.Config()

CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 4
MAX_CHECKPOINTS = 5

SAVE_DIR = Path("model")
CHECKPOINT_DIR = SAVE_DIR / Path(MODEL_ARGS.to_name())
LOG_CSV_PATH = SAVE_DIR / Path(f"{MODEL_ARGS.to_name()}.csv")


def save_checkpoint(model, optimizer, step, keep=MAX_CHECKPOINTS):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{step}.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )

    # Cleanup old checkpoints
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: int(p.stem))
    if len(checkpoints) > keep:
        for cp in checkpoints[:-keep]:
            cp.unlink()


def load_checkpoint(model, optimizer, device):
    if not CHECKPOINT_DIR.exists():
        return 0

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: int(p.stem))
    if not checkpoints:
        return 0

    latest_checkpoint_path = checkpoints[-1]
    print(f"loading checkpoint from {latest_checkpoint_path}")
    state = torch.load(latest_checkpoint_path, map_location=device)

    model_state = state["model"]
    # fix _orig_mod prefix
    new_model_state = {}
    for k, v in model_state.items():
        if k.startswith("_orig_mod."):
            new_model_state[k[len("_orig_mod.") :]] = v
        else:
            new_model_state[k] = v

    model.load_state_dict(new_model_state)
    optimizer.load_state_dict(state["optimizer"])
    console.print(f"[bold green]Resumed from checkpoint at step {state['step']}[/]")
    return state["step"]


def load_model_from_checkpoint(checkpoint_path=None, device=None):
    tokenizer = get_tokenizer()
    vocab_size = MODEL_ARGS.vocab_size

    dev = get_device() if device is None else device
    model = archie.ArchieModel(MODEL_ARGS).to(dev).to(torch.bfloat16)

    if checkpoint_path is None:
        if not CHECKPOINT_DIR.exists():
            raise FileNotFoundError(
                f"No checkpoint directory found at {CHECKPOINT_DIR}"
            )
        checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: int(p.stem))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
        checkpoint_path = checkpoints[-1]

    state = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state["model"])
    console.print(
        f"[bold green]Loaded model from checkpoint at step {state['step']}[/]"
    )
    model.eval()
    return model, tokenizer


accumulation_steps = 32
# Calculate total steps based on tokens seen
tokens_per_step = MODEL_ARGS.block_size * accumulation_steps * MODEL_ARGS.batch_size
total_steps = int(1.5e12 / tokens_per_step)  # ~9.16M steps
current_step = int(60e9 / tokens_per_step)  # ~366k steps (where you are now)
warmup_steps = 2000


def get_lr(step, warmup_steps, max_steps, base_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train_model(
    num_steps=None,
    vocab_size=8192,
    num_workers=8,
):
    tokenizer = archie.get_tokenizer()

    device = MODEL_ARGS.device
    console.print(f"[bold]Using device:[/] {device}")

    model = archie.ArchieModel(MODEL_ARGS).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_memory_gb = (param_count * 2) / (1024**3)

    console.print(f"[bold]Parameters:[/] {param_count:,}")
    console.print(f"[bold]Parameter memory (bfloat16):[/] {param_memory_gb:.2f} GB")

    model = model.to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.1
    )
    criterion = nn.CrossEntropyLoss()

    console.print(
        f"[bold]Batch size:[/] {MODEL_ARGS.batch_size} (effective {MODEL_ARGS.batch_size * accumulation_steps} with {accumulation_steps}x accumulation)"
    )

    step = load_checkpoint(model, optimizer, device)

    # 1. Rescale output projections
    # This is a common technique to stabilize training for deep models.
    # scale = 1.0 / math.sqrt(2 * MODEL_ARGS.n_layers)
    # for layer in model.layers:
    #     layer.attention.wo.weight.data *= scale
    #     layer.feed_forward.w2.weight.data *= scale

    print("Compiling model...")
    model = torch.compile(model)
    print("Model compiled.")
    if num_steps and step >= num_steps:
        console.print("[bold green]Training already complete.[/]")
        return model, tokenizer

    def fmt_duration(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    def make_table(step, avg_loss, steps_per_sec, best_loss, data_time_pct=None):
        table = Table(show_header=True, header_style="bold cyan", min_width=60)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", justify="right", width=40)

        if num_steps:
            progress_pct = step / num_steps * 100
            steps_remaining = num_steps - step

            if steps_per_sec:
                eta_total = fmt_duration(steps_remaining / steps_per_sec)
                progress_str = f"[magenta]{step:,} / {num_steps:,} ({progress_pct:.1f}%) — ETA {eta_total}[/]"
            else:
                progress_str = (
                    f"[magenta]{step:,} / {num_steps:,} ({progress_pct:.1f}%)[/]"
                )
        else:
            progress_str = f"[magenta]{step:,} steps[/]"

        steps_to_checkpoint = CHECKPOINT_INTERVAL - (step % CHECKPOINT_INTERVAL)
        tokens_seen = (
            step * MODEL_ARGS.block_size * MODEL_ARGS.batch_size * accumulation_steps
        ) / 2  # used to be 160 * 1024.

        if steps_per_sec:
            eta_checkpoint = fmt_duration(steps_to_checkpoint / steps_per_sec)
            checkpoint_str = f"{steps_to_checkpoint} steps (~{eta_checkpoint})"
        else:
            checkpoint_str = f"{steps_to_checkpoint} steps"

        table.add_row("Progress", progress_str)
        table.add_row(
            "Loss (avg)",
            (
                f"[yellow]{avg_loss:.4f}[/] (PPL={np.exp(avg_loss):.2f})"
                if avg_loss
                else "—"
            ),
        )
        table.add_row(
            "Best Loss",
            (
                f"[green]{best_loss:.4f}[/] (PPL={np.exp(best_loss):.2f})"
                if best_loss
                else "—"
            ),
        )
        table.add_row("Steps / sec", f"{steps_per_sec:.1f}" if steps_per_sec else "—")
        tokens_per_param = tokens_seen / param_count
        table.add_row(
            "Tokens seen", f"{tokens_seen:,} ({tokens_per_param:.2f} tok/param)"
        )
        if data_time_pct is not None:
            train_pct = 100 - data_time_pct
            bar_width = 20
            filled = int(bar_width * train_pct / 100)
            bar = f"[green]{'█' * filled}[/][dim]{'░' * (bar_width - filled)}[/] {train_pct:.0f}%"
            table.add_row("Efficiency", bar)
        table.add_row("Device", str(device))
        table.add_row("Next checkpoint", checkpoint_str)
        return table

    model.train()
    running_loss = torch.tensor(0.0, device=device)
    steps_since_log = 0
    best_loss = None
    avg_loss = None
    steps_per_sec = None
    t_start = time.time()
    t_log_start = time.time()
    total_data_time = 0.0
    total_train_time = 0.0

    write_header = not LOG_CSV_PATH.exists()
    log_csv = open(LOG_CSV_PATH, "a", newline="")
    log_writer = csv.writer(log_csv)
    if write_header:
        log_writer.writerow(["step", "avg_loss", "best_loss", "steps_per_sec"])

    dataset = archie.training.get_datasets()

    micro_step = 0
    with Live(
        make_table(step, avg_loss, steps_per_sec, best_loss),
        console=console,
        refresh_per_second=4,
    ) as live:
        text_dataset = archie.training.TextDataset(dataset, tokenizer, MODEL_ARGS)
        train_loader = DataLoader(
            text_dataset,
            batch_size=MODEL_ARGS.batch_size,
            num_workers=num_workers,
            prefetch_factor=64,  # Prefetch
            pin_memory=True,  # Optimization
        )
        for x, y in train_loader:
            data_load_time = time.time()
            total_data_time += data_load_time - t_start

            x, y = x.to(device), y.to(device)
            train_start = time.time()

            if micro_step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = (
                criterion(logits.reshape(-1, MODEL_ARGS.vocab_size), y.reshape(-1))
                / accumulation_steps
            )

            loss.backward()
            micro_step += 1

            if micro_step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                current_lr = get_lr(step, warmup_steps, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                optimizer.step()

            running_loss += loss.detach() * accumulation_steps
            total_train_time += time.time() - train_start
            step += 1
            steps_since_log += 1
            t_start = time.time()

            if steps_since_log == LOG_INTERVAL:
                avg_loss = running_loss.item() / LOG_INTERVAL
                elapsed = time.time() - t_log_start
                steps_per_sec = LOG_INTERVAL / elapsed
                data_time_pct = (
                    (total_data_time / (total_data_time + total_train_time) * 100)
                    if (total_data_time + total_train_time) > 0
                    else 0
                )
                if best_loss is None or avg_loss < best_loss:
                    best_loss = avg_loss
                live.update(
                    make_table(step, avg_loss, steps_per_sec, best_loss, data_time_pct)
                )
                if step % 50 == 0:
                    log_writer.writerow(
                        [
                            step,
                            f"{avg_loss:.6f}",
                            f"{best_loss:.6f}",
                            f"{steps_per_sec:.2f}",
                        ]
                    )
                    log_csv.flush()
                running_loss.zero_()
                steps_since_log = 0
                total_data_time = 0.0
                total_train_time = 0.0
                t_log_start = time.time()

            if step % CHECKPOINT_INTERVAL == 0:
                console.print(f"\n[bold yellow]Generating sample at step {step}:[/]")
                prompt = "The capital of France is"
                words = []
                for chunk in stream_text(
                    model, tokenizer, prompt, steps=50, temperature=0.8
                ):
                    words.append(chunk["text"])
                print(prompt + ("".join(words)))
                print("\n")

                model.train()  # Ensure model is back in training mode
                save_checkpoint(model._orig_mod, optimizer, step)

            if num_steps and step >= num_steps:
                break

    log_csv.close()
    save_checkpoint(model._orig_mod, optimizer, step)
    console.print("[bold green]Training complete.[/]")
    return model, tokenizer


def stream_text(model, tokenizer, prompt_text, steps=200, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt_text)
    current = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(steps):
            logits = model(current[:, -MODEL_ARGS.max_seq_len :])
            logits = logits[0, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            prob = probs[next_token].item()

            current = torch.cat(
                [current, torch.tensor([[next_token]], device=device)], dim=1
            )
            text = tokenizer.decode([next_token])
            yield {
                "text": text,
                "prob": prob,
                "id": next_token,
            }


def generate_text(model, tokenizer, prompt_text, steps=200, temperature=0.8):
    return prompt_text + "".join(
        chunk["text"]
        for chunk in stream_text(model, tokenizer, prompt_text, steps, temperature)
    )


if __name__ == "__main__":
    model, tokenizer = train_model()
    # prompt = "Once upon a time"
    # console.print("[bold]Generated Text:[/]")
    # console.print(prompt, end="", highlight=False)
    # for chunk in stream_text(model, tokenizer, prompt, steps=200, temperature=0.8):
    #     sys.stdout.write(chunk)
    #     sys.stdout.flush()
    print()
