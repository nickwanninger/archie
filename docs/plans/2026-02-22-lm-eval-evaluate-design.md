# lm_eval Evaluate Subcommand Design

**Date:** 2026-02-22

## Goal

Add an `evaluate` subcommand to `archie/__main__.py` that runs the pretrained
model against the `arc_easy` and `hellaswag` benchmarks using
`lm_eval.simple_evaluate`.

## Context

- Archie uses a custom tiktoken GPT-2 tokenizer (~50K vocab).
- The model is a raw PyTorch `nn.Module`, not a HuggingFace model.
- `arc_easy` and `hellaswag` are multiple-choice tasks — they only require
  `loglikelihood`, not open-ended generation.
- `lm_eval` is not currently installed.

## Changes

### 1. Dependency

Add `lm_eval>=0.4.0` to `pyproject.toml`.

### 2. `ArchieEvalWrapper` (in `__main__.py`)

A thin subclass of `lm_eval.api.model.LM` that bridges the Archie model and
tiktoken tokenizer to the lm_eval interface.

Key properties:
- `eot_token_id` → 50256 (GPT-2 `<|endoftext|>`)
- `max_length` → `config.max_seq_len`

Key methods:
- `loglikelihood(requests)` — for each `(context, continuation)` pair:
  tokenize both, run a forward pass, sum log-probs over the continuation
  tokens, check greedy match. Truncates context if combined length exceeds
  `max_seq_len`.
- `loglikelihood_rolling(requests)` — sliding-window perplexity.
- `generate_until(requests)` — `NotImplementedError` (not needed for these
  tasks).

### 3. `cmd_evaluate` function

Loads the model via the existing `load_model_for_inference`, wraps it in
`ArchieEvalWrapper`, calls `lm_eval.simple_evaluate`, logs results via the
existing logger, and optionally saves the full results dict to JSON.

### 4. Subparser arguments

| Argument       | Default                  | Description                          |
|----------------|--------------------------|--------------------------------------|
| `--tasks`      | `arc_easy,hellaswag`     | Comma-separated lm_eval task names   |
| `--output`     | _(none)_                 | Optional path to save JSON results   |
| `--batch-size` | `1`                      | Requests per forward pass            |

## Usage

```
python -m archie --model flicker evaluate
python -m archie --model glimmer evaluate --tasks arc_easy,hellaswag --output results.json
```
