import tiktoken


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
