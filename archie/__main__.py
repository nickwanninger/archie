import argparse
import archie
import archie.training
import archie.config
import uuid

import torch
import torchinfo
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datetime import datetime
import time

import sys
import math

import logging
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, SpinnerColumn

import jsonlines


def setup_logging(debug_log_path):
    log = logging.getLogger("archie")
    console_handler = RichHandler(level=logging.DEBUG, show_path=False)
    log.addHandler(console_handler)
    formatter = logging.Formatter(
        "{asctime} [{levelname}] {name}: {message}", style="{"
    )
    file_handler = logging.FileHandler(debug_log_path)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.setLevel(logging.DEBUG)
    return log


def enable_gradient_checkpointing(model):
    for layer in model.layers:
        layer._forward = layer.forward
        layer.forward = lambda x, m=layer: torch.utils.checkpoint.checkpoint(
            m._forward, x, use_reentrant=False
        )


def get_lr(step, warmup_steps=100, max_steps=30000, max_lr=3e-4, min_lr=1e-4):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def _prob_to_bg_ansi(prob: float) -> str:
    """Map a token probability to an ANSI 24-bit background color escape.

    Low probability  → muted red
    High probability → muted green
    """
    prob = max(0.0, min(1.0, prob))
    if prob < 0.5:
        r, g = 140, int(prob * 2 * 140)
    else:
        r, g = int((1.0 - prob) * 2 * 140), 140
    return f"\033[48;2;{r};{g};30m"


_ANSI_RESET = "\033[0m"


@torch.no_grad()
def stream_tokens(
    model, tokenizer, prompt, max_tokens=200, temperature=0.8, stop_token=None
):
    """Yield (fragment, probability) tuples one token at a time."""
    tokens = torch.tensor([tokenizer.encode(prompt)]).to(model.config.device)

    for _ in range(max_tokens):
        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        if stop_token is not None and token_id == stop_token:
            break
        prob = probs[0, token_id].item()
        tokens = torch.cat([tokens, next_token], dim=1)
        try:
            yield tokenizer.decode([token_id]), prob
        except:
            pass
        if tokens.shape[1] >= model.config.max_seq_len:
            break


@torch.no_grad()
def generate_text(model, tokenizer, prompt="The", max_tokens=50, temperature=0.8):
    model.eval()
    text = prompt + "".join(
        f
        for f, _ in stream_tokens(
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            stop_token=tokenizer.eot_token,
        )
    )
    model.train()
    return text


def load_model_for_training(config, log):
    """Load or create a model + optimizer, resuming from checkpoint if present.

    Returns (model, optimizer, global_step, tokens_seen).
    """
    checkpoint_dir = config.get_checkpoint_dir()
    model_path = checkpoint_dir / "model.pt"

    model = archie.create_model(config).to(config.device).to(torch.bfloat16)
    torchinfo.summary(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=config.device == "cuda",
    )

    global_step = 0
    tokens_seen = 0

    if model_path.exists():
        log.info("Loading state...")
        state = torch.load(model_path, map_location=config.device)
        log.info("Loading model...")
        model.load_state_dict(state["model"], strict=False)
        if "optimizer" in state:
            log.info("Loading optimizer...")
            optimizer.load_state_dict(state["optimizer"])
        global_step = state["step"]
        tokens_seen = state["tokens_seen"]
        log.info("Done.")
    else:
        log.info("Model not found, starting from scratch.")

    return model, optimizer, global_step, tokens_seen


def load_model_for_inference(config, log):
    """Load a model from checkpoint for inference or validation.

    Returns (model, tokenizer).
    """
    checkpoint_dir = config.get_checkpoint_dir()
    model_path = checkpoint_dir / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {model_path}")

    model = archie.create_model(config).to(config.device).to(torch.bfloat16)
    tokenizer = archie.get_tokenizer()

    log.info("Loading model...")
    state = torch.load(model_path, map_location=config.device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    log.info("Done.")

    return model, tokenizer


def cmd_train(args, config, log):
    run_id = str(uuid.uuid4())
    log.info(f"Starting run {run_id}")

    checkpoint_dir = config.get_checkpoint_dir()
    model_path = checkpoint_dir / "model.pt"
    training_log_path = checkpoint_dir / "training.jsonl"
    expert_usage_path = checkpoint_dir / "expert_usage.jsonl"
    eval_log_path = checkpoint_dir / "evaluation.jsonl"

    log.info("Allocating model...")
    model, optimizer, global_step, tokens_seen = load_model_for_training(config, log)
    tokenizer = archie.get_tokenizer()

    batch_size = config.training_batch_size
    desired_batch_size = config.training_tokens_per_batch // config.max_seq_len
    accumulation_steps = int(desired_batch_size // batch_size)

    log.info(f"Batch size: {batch_size}")
    log.info(f"Accumulation steps: {accumulation_steps}")
    log.info(f"Effective batch size: {batch_size * accumulation_steps}")

    all_on_gpu = all(p.device.type == "cuda" for p in model.parameters())
    log.info(f"Fully on GPU: {all_on_gpu}")
    for name, buf in model.named_buffers():
        if buf.device.type != "cuda":
            print(f"Buffer on CPU: {name} on {buf.device}")

    # return

    dataset = archie.training.get_datasets()
    text_dataset = archie.training.TextDataset(dataset, tokenizer, config)
    train_loader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    total_params = sum(p.numel() for p in model.parameters())

    def append_log_entry(data):
        data["run_id"] = run_id
        with jsonlines.open(training_log_path, mode="a") as writer:
            writer.write(data)

    log.info("Starting training...")

    opt_model = torch.compile(model, mode="default")
    enable_gradient_checkpointing(opt_model)

    running_loss = 0.0
    accumulated_expert_counts = None
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Step {task.fields[step]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]loss: {task.fields[loss]:.2f}"),
        console=log.handlers[0].console,
        transient=True,
    )

    with progress:
        acc_task = progress.add_task(
            "accumulation", total=accumulation_steps, step=global_step, loss=0.0
        )

        for i, (x, y) in enumerate(train_loader):
            x = x.to(config.device)
            y = y.to(config.device)

            tokens_seen += x.numel()

            start = time.perf_counter()
            logits, loss = opt_model(x, labels=y)
            end = time.perf_counter()

            if hasattr(model, "_expert_counts"):
                counts = model._expert_counts.detach()
                if accumulated_expert_counts is None:
                    accumulated_expert_counts = counts
                else:
                    accumulated_expert_counts = accumulated_expert_counts + counts

            loss = loss / accumulation_steps
            running_loss += loss.item()
            loss.backward()

            progress.update(acc_task, advance=1, loss=running_loss)

            if (i + 1) % accumulation_steps != 0:
                continue

            global_step += 1

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

            optimizer.step()

            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            if accumulated_expert_counts is not None:
                expert_counts_list = accumulated_expert_counts.tolist()
                with jsonlines.open(expert_usage_path, mode="a") as writer:
                    writer.write({
                        "step": global_step,
                        "expert_counts": expert_counts_list,
                        "datetime": datetime.utcnow().isoformat(),
                    })
                accumulated_expert_counts = None

            effective_loss = running_loss
            running_loss = 0.0

            perplexity = math.exp(effective_loss)
            tokens_M = tokens_seen / 1_000_000
            tokens_B = tokens_M / 1_000
            tokens_per_param = tokens_seen / total_params
            log.info(
                f"Step {global_step} | Tokens: {tokens_B:.4f}B ({tokens_per_param:.3f}/param)"
                f" | PPL: {perplexity:9.2f} | loss: {effective_loss:.2f}"
                f" | norm: {norm:.4f} | LR: {lr:.2e}"
            )

            append_log_entry(
                {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "tokens_per_param": tokens_per_param,
                    "perplexity": perplexity,
                    "loss": effective_loss,
                    "norm": norm,
                    "lr": lr,
                    "datetime": datetime.utcnow().isoformat(),
                    "batch_size": batch_size,
                    "acc_steps": accumulation_steps,
                }
            )

            progress.reset(acc_task, step=global_step, loss=0.0)

            if global_step % 500 == 0:
                log.info("Running lm_eval benchmarks...")
                run_lm_eval(
                    model,
                    tokenizer,
                    config,
                    tasks=["arc_easy", "hellaswag", "lambada_openai", "winogrande"],
                    eval_log_path=eval_log_path,
                    log=log,
                    step=global_step,
                )

            if global_step % 25 != 0:
                continue

            log.info("Generating samples...")
            prompts = [
                "The",
                "In the",
                "Scientists have",
                "Once upon a time",
                "The Capital of France is",
                "The 2008 financial crisis was caused by",
                "The python code to reverse a list is:\n```python\ndef reverse_list(l):",
            ]
            for prompt in prompts:
                generated = generate_text(model, tokenizer, prompt=prompt, max_tokens=100)
                log.info(f"  '{prompt}' → {generated}")

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


def cmd_validate(args, config, log):
    log.info("Loading model for validation...")
    model, tokenizer = load_model_for_inference(config, log)

    log.info("Loading dataset...")
    dataset = archie.training.get_datasets()
    text_dataset = archie.training.TextDataset(dataset, tokenizer, config)
    val_loader = DataLoader(
        text_dataset,
        batch_size=8,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    num_batches = args.batches
    log.info(f"Evaluating cross-validation loss over {num_batches} batches...")

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_batches:
                break
            x = x.to(config.device)
            y = y.to(config.device)
            _, loss = model(x, labels=y)
            total_loss += loss.item()
            count += 1

            if (i + 1) % 10 == 0:
                avg = total_loss / count
                log.info(
                    f"Batch {i + 1}/{num_batches} | Loss: {avg:.4f} | PPL: {math.exp(avg):.2f}"
                )

    avg_loss = total_loss / count
    log.info(
        f"Validation complete | Avg Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f}"
    )


def cmd_test(args, config, log):
    log.info("Loading model for inference...")
    model, tokenizer = load_model_for_inference(config, log)

    print(
        "\nReady. Enter a prompt for next-token prediction. Empty line or Ctrl-C to exit.\n"
    )

    while True:
        try:
            prompt = input("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            break

        print(f"\n{prompt}", end="", flush=True)
        for fragment, prob in stream_tokens(
            model, tokenizer, prompt, max_tokens=200, temperature=0.8
        ):
            print(
                f"{_prob_to_bg_ansi(prob)}{fragment}{_ANSI_RESET}", end="", flush=True
            )
        print("\n")


def run_lm_eval(model, tokenizer, config, tasks, eval_log_path, log, step=None):
    """Run lm_eval on the given tasks and append results to eval_log_path.

    The model is temporarily set to eval mode and restored afterwards.
    """
    import json
    import lm_eval
    from lm_eval.api.model import LM

    class ArchieEvalWrapper(LM):
        """Thin wrapper bridging Archie's tiktoken tokenizer and ArchieModel to
        the lm_eval.api.model.LM interface.

        Arc_easy and hellaswag are multiple-choice tasks that only call
        loglikelihood(), so generate_until() is left unimplemented.
        """

        def __init__(self, arch_model, tokenizer, config):
            super().__init__()
            self._model = arch_model
            self._tokenizer = tokenizer
            self._config = config

        @property
        def eot_token_id(self) -> int:
            return 50256  # GPT-2 <|endoftext|>

        @property
        def tokenizer_name(self) -> str:
            return "archie_gpt2"

        @property
        def max_length(self) -> int:
            return self._config.max_seq_len

        @property
        def max_gen_toks(self) -> int:
            return 256

        def tok_encode(self, text: str) -> list[int]:
            return self._tokenizer.encode(text)

        def tok_decode(self, tokens) -> str:
            return self._tokenizer.decode(list(tokens))

        @torch.no_grad()
        def loglikelihood(self, requests):
            results = []
            for req in requests:
                context, continuation = req.args
                ctx_tokens = self._tokenizer.encode(context)
                cont_tokens = self._tokenizer.encode(continuation)

                # Truncate context so combined length fits in max_seq_len
                max_ctx = self._config.max_seq_len - len(cont_tokens)
                if max_ctx <= 0:
                    cont_tokens = cont_tokens[: self._config.max_seq_len - 1]
                    max_ctx = 1
                ctx_tokens = ctx_tokens[-max_ctx:]

                input_ids = ctx_tokens + cont_tokens
                input_tensor = torch.tensor(
                    [input_ids], dtype=torch.long, device=self._config.device
                )

                logits, _ = self._model(input_tensor)
                # logits[i] predicts token[i+1], so continuation starts at len(ctx)-1
                cont_start = len(ctx_tokens) - 1
                cont_logits = logits[0, cont_start : cont_start + len(cont_tokens), :]

                log_probs = F.log_softmax(cont_logits, dim=-1)
                cont_tensor = torch.tensor(
                    cont_tokens, dtype=torch.long, device=self._config.device
                )
                token_log_probs = log_probs[range(len(cont_tokens)), cont_tensor]
                sum_log_prob = token_log_probs.sum().item()

                greedy = cont_logits.argmax(dim=-1)
                is_greedy = (greedy == cont_tensor).all().item()

                results.append((sum_log_prob, bool(is_greedy)))
            return results

        @torch.no_grad()
        def loglikelihood_rolling(self, requests):
            results = []
            for req in requests:
                (text,) = req.args
                tokens = [self.eot_token_id] + self._tokenizer.encode(text)
                tokens = tokens[: self._config.max_seq_len]

                input_tensor = torch.tensor(
                    [tokens], dtype=torch.long, device=self._config.device
                )
                logits, _ = self._model(input_tensor)

                log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
                targets = torch.tensor(
                    tokens[1:], dtype=torch.long, device=self._config.device
                )
                sum_log_prob = log_probs[range(len(targets)), targets].sum().item()
                results.append(sum_log_prob)
            return results

        def generate_until(self, requests):
            raise NotImplementedError(
                "generate_until is not supported; use loglikelihood-based tasks"
            )

    was_training = model.training
    model.eval()
    try:
        wrapper = ArchieEvalWrapper(model, tokenizer, config)
        results = lm_eval.simple_evaluate(model=wrapper, tasks=tasks, batch_size=1)
    finally:
        if was_training:
            model.train()

    for task, metrics in results["results"].items():
        log.info(f"lm_eval {task}: {metrics}")

    entry = {
        "datetime": datetime.utcnow().isoformat(),
        "tasks": tasks,
        "results": json.loads(json.dumps(results["results"], default=str)),
    }
    if step is not None:
        entry["step"] = step

    with jsonlines.open(eval_log_path, mode="a") as writer:
        writer.write(entry)
    log.info(f"Appended eval results to {eval_log_path}")

    return results


def cmd_upcycle(args, config, log):
    target_config = archie.config.load(args.target)
    if target_config.num_experts is None:
        raise ValueError(f"Target config {args.target!r} is not an MoE config")

    source_path = config.get_checkpoint_dir() / "model.pt"
    if not source_path.exists():
        raise FileNotFoundError(f"No dense checkpoint at {source_path}")

    log.info(f"Loading dense checkpoint from {source_path}...")
    state = torch.load(source_path, map_location="cpu")
    dense_sd = state["model"]

    log.info(f"Creating MoE model with config {args.target!r}...")
    moe_model = archie.create_model(target_config)
    moe_sd = moe_model.state_dict()

    new_sd = {}
    for key in moe_sd:
        if ".moe.experts." in key:
            # Map layers.{i}.moe.experts.{j}.{param} -> layers.{i}.ffn.{param}
            parts = key.split(".")
            layer_idx = parts[1]
            param_parts = parts[5:]  # after "experts.{j}"
            dense_key = f"layers.{layer_idx}.ffn.{'.'.join(param_parts)}"
            if dense_key in dense_sd:
                new_sd[key] = dense_sd[dense_key].clone()
                continue
            log.warning(f"No dense key for {key} (expected {dense_key})")
            new_sd[key] = moe_sd[key]
        elif ".moe.router." in key:
            # Keep random init
            new_sd[key] = moe_sd[key]
        elif key in dense_sd:
            new_sd[key] = dense_sd[key]
        else:
            log.warning(f"Key {key} not found in dense checkpoint, using random init")
            new_sd[key] = moe_sd[key]

    moe_model.load_state_dict(new_sd)

    target_dir = target_config.get_checkpoint_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "model.pt"

    log.info(f"Saving MoE checkpoint to {target_path}...")
    torch.save(
        {
            "model": moe_model.state_dict(),
            "step": state.get("step", 0),
            "tokens_seen": state.get("tokens_seen", 0),
        },
        target_path,
    )

    import shutil
    source_dir = config.get_checkpoint_dir()
    for log_file in ["training.jsonl", "evaluation.jsonl"]:
        src = source_dir / log_file
        if src.exists():
            shutil.copy2(src, target_dir / log_file)
            log.info(f"Copied {log_file} to {target_dir}")

    dense_params = sum(p.numel() for p in dense_sd.values())
    moe_params = sum(p.numel() for p in moe_model.parameters())
    log.info(
        f"Upcycled {dense_params/1e6:.1f}M dense -> {moe_params/1e6:.1f}M MoE "
        f"({target_config.num_experts} experts, top-{target_config.top_k})"
    )


def cmd_evaluate(args, config, log):
    import json

    checkpoint_dir = config.get_checkpoint_dir()
    eval_log_path = checkpoint_dir / "evaluation.jsonl"

    model, tokenizer = load_model_for_inference(config, log)
    tasks = [t.strip() for t in args.tasks.split(",")]
    log.info(f"Evaluating on tasks: {tasks}")

    results = run_lm_eval(model, tokenizer, config, tasks, eval_log_path, log)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Saved full results to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Archie language model")
    parser.add_argument(
        "--model",
        choices=archie.config.list_configs(),
        default="flicker",
        help="Model configuration to use (default: flicker)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run the pretraining loop")

    val_parser = subparsers.add_parser(
        "validate", help="Evaluate cross-validation loss"
    )
    val_parser.add_argument(
        "--batches",
        type=int,
        default=100,
        help="Number of batches to evaluate (default: 100)",
    )

    subparsers.add_parser("test", help="Interactive next-token prediction (input loop)")

    upcycle_parser = subparsers.add_parser(
        "upcycle", help="Convert dense checkpoint to MoE"
    )
    upcycle_parser.add_argument(
        "--target",
        required=True,
        choices=archie.config.list_configs(),
        help="Target MoE config name",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Run lm_eval benchmarks")
    eval_parser.add_argument(
        "--tasks",
        default="arc_easy,hellaswag,lambada_openai,winogrande",
        help="Comma-separated task names (default: arc_easy,hellaswag,lambada_openai,winogrande)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Forward-pass batch size (default: 1)",
    )
    eval_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results",
    )

    args = parser.parse_args()

    config = archie.config.load(args.model)
    checkpoint_dir = config.get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = checkpoint_dir / "archie.log"

    log = setup_logging(debug_log_path)

    dispatch = {
        "train": cmd_train,
        "validate": cmd_validate,
        "test": cmd_test,
        "evaluate": cmd_evaluate,
        "upcycle": cmd_upcycle,
    }
    dispatch[args.command](args, config, log)


if __name__ == "__main__":
    main()
