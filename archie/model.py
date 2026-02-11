import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from archie.config import Config


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
        return self.weight * self._norm(x)


class MistralMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

        # Standard initialization
        std = config.dim**-0.5
        nn.init.normal_(self.w1.weight, mean=0, std=std)
        nn.init.normal_(self.w3.weight, mean=0, std=std)

        # w2 is the output projection - scale by depth
        std = hidden_dim**-0.5 / math.sqrt(2 * config.n_layers)
        nn.init.normal_(self.w2.weight, mean=0, std=std)

    def forward(self, x):
        # SwiGLU: (Swish(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # Initialize with proper scaling
        std = config.dim**-0.5
        nn.init.normal_(self.wq.weight, mean=0, std=std)
        nn.init.normal_(self.wk.weight, mean=0, std=std)
        nn.init.normal_(self.wv.weight, mean=0, std=std)
        # wo is output - scale by depth
        std = (config.n_heads * self.head_dim) ** -0.5 / math.sqrt(2 * config.n_layers)
        nn.init.normal_(self.wo.weight, mean=0, std=std)

    def forward(self, x, freqs_cis):
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, L, self.n_heads, self.head_dim)
        xk = xk.view(B, L, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = self.apply_rotary_emb(
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

    def apply_rotary_emb(self, xq, xk, freqs_cos, freqs_sin):
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


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MistralMLP(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, freqs_cis):
        # Pre-norm residuals (Mistral/Llama style)
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ArchieModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # Token embedding (weights will be tied to output)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0, std=config.dim**-0.5)

        # Build the layers using the config
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Initialize output layer
        nn.init.normal_(self.output.weight, mean=0, std=config.dim**-0.5)

        # Weight Tying: use embedding weights for output projection
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies for the maximum possible length.
        # We explicitly pass rope_theta to ensure the model aligns with the intended context window.
        # If this is not calibrated (e.g. using 10k theta for 32k context), the model will 'hallucinate'
        # or lose coherence past token 4096.
        # Precompute RoPE frequencies
        self.freqs_cos, self.freqs_sin = self.precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            theta=config.rope_theta,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _batch, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Get the rotation frequencies for this specific sequence length
        self.freqs_cos = self.freqs_cos.to(h.device)
        self.freqs_sin = self.freqs_sin.to(h.device)

        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]

        for layer in self.layers:
            h = layer(h, (freqs_cos, freqs_sin))

        h = self.norm(h)
        return self.output(h)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        # RoPE works on pairs of dimensions
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cos = torch.cos(freqs)  # real part
        freqs_sin = torch.sin(freqs)  # imaginary part
        return freqs_cos, freqs_sin
