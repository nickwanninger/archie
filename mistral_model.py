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


import gzip
import requests

# For training

# For alignment (SFT)
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
# https://huggingface.co/datasets/DataMuncher-Labs/UltiMath

# For fine-tuning
# https://huggingface.co/datasets/Anthropic/hh-rlhf

console = Console()


console = Console()


def get_tokenizer():
    """
    Returns a custom tiktoken encoder based on GPT-2 but with added special tokens
    for ChatML-style interactions (<|im_start|>, <|im_end|>).
    """
    base = tiktoken.get_encoding("gpt2")

    # We extend the special tokens.
    # Base GPT-2 has 50,257 tokens (0 to 50,256).
    # We add ours starting at 50,257.
    special_tokens = {
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
    }

    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **special_tokens},
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 24
    n_kv_heads: int = 4

    n_kv_heads: int = 4

    vocab_size: int = (
        50304  # 50257 (GPT-2) rounded up to multiple of 64 for efficiency + special tokens
    )
    multiple_of: int = 256  # For SwiGLU alignment
    # Norm Epsilon (norm_eps):
    # This acts as a stabilizer to prevent division by zero in RMSNorm.
    # - 1e-5: Default for Mistral-7B-v0.1 and Llama-2.
    # - 1e-6: Often recommended for bfloat16 stability in deeper models or when training from scratch.
    # CRITICAL: Changing this value requires retraining, as the model's weights adapt to the specific
    # squashing effect of this epsilon during training.
    norm_eps: float = 1e-5

    max_seq_len: int = 1024

    # RoPE Theta (rope_theta):
    # The base frequency for Rotary Positional Embeddings.
    # - 10,000.0: Standard for context lengths up to ~4096 (Llama 2, original Mistral).
    # - 100,000+ : Required for long context (8k, 32k, etc.) to prevent "attention collapse".
    # Changing this value fundamentally alters how the model perceives position and distance,
    # and strictly requires retraining or complex frequency scaling techniques.
    rope_theta: float = 10000.0

    block_size: int = 1024
    batch_size: int = 10

    def to_name(self):
        return f"{self.dim}d_{self.n_layers}l_{self.n_heads}h_{self.n_kv_heads}kv"


MODEL_ARGS = ModelArgs()

CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 4
MAX_CHECKPOINTS = 5

SAVE_DIR = Path("model")
CHECKPOINT_DIR = SAVE_DIR / Path(MODEL_ARGS.to_name())
LOG_CSV_PATH = SAVE_DIR / Path(f"{MODEL_ARGS.to_name()}.csv")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        x_float = x.float()
        return (
            x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        ).to(x.dtype)

    def forward(self, x):
        return (self.weight * self._norm(x)).to(torch.bfloat16)


class MistralMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # SwiGLU requires three linear layers
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

        # Initialize with proper scaling
        nn.init.normal_(self.w1.weight, mean=0, std=args.dim**-0.5)
        nn.init.normal_(self.w2.weight, mean=0, std=hidden_dim**-0.5)
        nn.init.normal_(self.w3.weight, mean=0, std=args.dim**-0.5)

    def forward(self, x):
        # SwiGLU: (Swish(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given theta.

    Args:
        dim: Dimension of the frequency tensor (head_dim).
        end: Maximum sequence length to precompute for.
        theta: The base frequency for RoPE.
               10000.0 is the standard "handshake" value for 1024-4096 context.
               For longer contexts, this must be increased (e.g. 500k for Llama 3)
               to avoid the "Gibberish Factor" where distant tokens become indistinguishable.
    """
    # RoPE works on pairs of dimensions
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # freqs (T, D/2). reshape to (1, T, 1, D/2)
    cos = freqs_cos.view(1, xq_r.size(1), 1, xq_r.size(3))
    sin = freqs_sin.view(1, xq_r.size(1), 1, xq_r.size(3))

    # x0 is x[..., 0], x1 is x[..., 1]
    xq_out_r = xq_r[..., 0] * cos - xq_r[..., 1] * sin
    xq_out_i = xq_r[..., 0] * sin + xq_r[..., 1] * cos
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)

    xk_out_r = xk_r[..., 0] * cos - xk_r[..., 1] * sin
    xk_out_i = xk_r[..., 0] * sin + xk_r[..., 1] * cos
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Initialize with proper scaling
        nn.init.normal_(self.wq.weight, mean=0, std=args.dim**-0.5)
        nn.init.normal_(self.wk.weight, mean=0, std=args.dim**-0.5)
        nn.init.normal_(self.wv.weight, mean=0, std=args.dim**-0.5)
        nn.init.normal_(self.wo.weight, mean=0, std=args.dim**-0.5)

    def forward(self, x, freqs_cis):
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, L, self.n_heads, self.head_dim)
        xk = xk.view(B, L, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(
            xq, xk, freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1]
        )

        # Grouped-Query Attention: Repeat KV heads to match Q heads
        # Logic: If 16 Q heads and 4 KV heads, each KV head is shared by 4 Q heads
        xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
        xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        # Scaled Dot-Product Attention
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), is_causal=True
        )

        output = output.transpose(1, 2).reshape(B, L, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MistralMLP(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        # Pre-norm residuals (Mistral/Llama style)
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MistralLite(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Token embedding (weights will be tied to output)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0, std=args.dim**-0.5)

        # Build the layers using the args
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Initialize output layer
        nn.init.normal_(self.output.weight, mean=0, std=args.dim**-0.5)

        # Weight Tying: use embedding weights for output projection
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies for the maximum possible length.
        # We explicitly pass rope_theta to ensure the model aligns with the intended context window.
        # If this is not calibrated (e.g. using 10k theta for 32k context), the model will 'hallucinate'
        # or lose coherence past token 4096.
        # Precompute RoPE frequencies
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2, theta=args.rope_theta
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _batch, seqlen = tokens.shape
        h = self.tok_embeddings(tokens).to(torch.bfloat16)

        # Get the rotation frequencies for this specific sequence length
        self.freqs_cos = self.freqs_cos.to(h.device)
        self.freqs_sin = self.freqs_sin.to(h.device)

        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]

        for layer in self.layers:
            h = layer(h, (freqs_cos, freqs_sin))

        h = self.norm(h)
        return self.output(h).to(torch.bfloat16)


class TextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.eot = tokenizer._special_tokens.get("<|endoftext|>", config.vocab_size - 1)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # When using multiple workers, we need to shard the dataset
            ds = self.dataset.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )
        else:
            ds = self.dataset

        buffer = []
        chunk_size = self.config.block_size + 1

        for entry in ds:
            text = entry["text"]
            text = text.replace("’", "'").replace("“", '"').replace("”", '"')
            tokens = self.tokenizer.encode_ordinary(text)
            tokens.append(self.eot)
            buffer.extend(tokens)

            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[self.config.block_size :]  # Sliding by block_size

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


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
    model = MistralLite(MODEL_ARGS).to(dev).to(torch.bfloat16)

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


def train_model(
    num_steps=None,
    vocab_size=8192,
    accumulation_steps=4,
    num_workers=8,
):
    tokenizer = get_tokenizer()

    device = get_device()
    console.print(f"[bold]Using device:[/] {device}")

    model = MistralLite(MODEL_ARGS).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_memory_gb = (param_count * 2) / (1024**3)

    console.print(f"[bold]Parameters:[/] {param_count:,}")
    console.print(f"[bold]Parameter memory (bfloat16):[/] {param_memory_gb:.2f} GB")

    model = model.to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    console.print(
        f"[bold]Batch size:[/] {MODEL_ARGS.batch_size} (effective {MODEL_ARGS.batch_size * accumulation_steps} with {accumulation_steps}x accumulation)"
    )

    step = load_checkpoint(model, optimizer, device)
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
        )

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

    dataset = interleave_datasets([fineweb, wikipedia], probabilities=[0.8, 0.2])
    dataset = dataset.shuffle(buffer_size=40000)

    micro_step = 0
    with Live(
        make_table(step, avg_loss, steps_per_sec, best_loss),
        console=console,
        refresh_per_second=4,
    ) as live:
        text_dataset = TextDataset(dataset, tokenizer, MODEL_ARGS)
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
    model, tokenizer = train_model(
        accumulation_steps=16,
    )
    # prompt = "Once upon a time"
    # console.print("[bold]Generated Text:[/]")
    # console.print(prompt, end="", highlight=False)
    # for chunk in stream_text(model, tokenizer, prompt, steps=200, temperature=0.8):
    #     sys.stdout.write(chunk)
    #     sys.stdout.flush()
    print()
