import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from archie.config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len):
        # x: (batch, seq_len, n_heads, head_dim)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x, cos, sin):
    # x: (batch, seq_len, n_heads, head_dim)
    # cos, sin: (seq_len, head_dim)

    # Reshape cos/sin: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Split into even/odd
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices

    # Split cos/sin the same way
    cos = cos[..., 0::2]
    sin = sin[..., 0::2]

    # Rotate
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    # Interleave back together properly
    y = torch.stack([y1, y2], dim=-1)
    y = y.flatten(-2)  # Merge last two dims to restore head_dim

    return y


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # 4 in your case

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Project and reshape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        # K/V stay at n_kv_heads — SDPA handles GQA natively (PyTorch >= 2.2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=0.0 if not self.training else 0.1,
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GroupedQueryAttention(
            config.d_model, config.n_heads, config.n_kv_heads
        )
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ArchieModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss
