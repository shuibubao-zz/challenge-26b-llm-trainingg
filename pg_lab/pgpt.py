"""
pgpt.py — 官方 baseline GPT 的**忠实缩小型**，用于在没有 H100 的机器上测量"评测协议"这一变量。

拿来的部分（全部来自 OpenAI parameter-golf 官方 train_gpt.py  records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py）：
    - RMSNorm + RoPE(half rotation) + GQA + query-RMSNorm + q_gain
    - relu^2 MLP（modded-nanogpt 原配方）
    - resid_mix[n] 混合 x 与 x0（首层残差）
    - attn_scale / mlp_scale 可学习缩放
    - U-Net 式 skip_weights（前半层存 skip，后半层反序复用）
    - tied embedding + logit_softcap * tanh(logits / softcap)
    - zero-init 的 proj 层

改了什么（以及为什么必须改）：
    1. 词汇表 1024 → **256 裸字节**。官方指标 BPB 本就是字节口径，字节级下
       bits/byte = nats/byte / ln2 精确成立，去掉 tokenizer 换算这一混淆变量。
    2. 尺寸缩小到 CPU 可训练量级（见 Config 默认值），模型**结构完全不变**。
    3. forward 除了返回 mean loss，还额外返回 **逐位置 loss 矩阵** (B, L)，
       这是本实验的核心测量手段：一次前向即可得到"ρ(p) = 给定恰好 p 个前驱字节时的期望损失"。
    4. 去掉 DDP / bf16 autocast / flash-attn（CPU 上不可用），改用无副作用等价实现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Config:
    vocab_size: int = 256          # 裸字节
    num_layers: int = 4
    model_dim: int = 128
    num_heads: int = 4
    num_kv_heads: int = 2
    mlp_mult: int = 2
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5

    def param_groups(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos: Optional[Tensor] = None
        self._sin: Optional[Tensor] = None

    def forward(self, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        if self._cos is None or self._seq_len_cached != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        if dim % num_heads or num_heads % num_kv_heads:
            raise ValueError("dim/num_heads 或 num_heads/num_kv_heads 不整除")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2:
            raise ValueError("head_dim 必须为偶数（RoPE）")
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (q.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tie_embeddings = cfg.tie_embeddings
        self.logit_softcap = cfg.logit_softcap
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.num_encoder_layers = cfg.num_layers // 2
        self.num_decoder_layers = cfg.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, cfg.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(cfg.model_dim, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_mult,
                  cfg.rope_base, cfg.qk_gain_init)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm()
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=cfg.tied_embed_init_std)
        else:
            raise ValueError("本实验台固定使用 tied embedding（与官方 baseline 一致）")

    def hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def logits(self, input_ids: Tensor) -> Tensor:
        h = self.hidden(input_ids)
        flat = h.reshape(-1, h.size(-1))
        lp = F.linear(flat, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None):
        """返回 (mean_loss, per_position_loss)。per_position_loss 形状为 input_ids.shape。"""
        lg = self.logits(input_ids).float().view(*input_ids.shape, self.cfg.vocab_size)
        per_pos = F.cross_entropy(
            lg.reshape(-1, self.cfg.vocab_size), target_ids.reshape(-1), reduction="none"
        ).view(input_ids.shape)
        return per_pos.mean(), per_pos.detach()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- training

class AdamW_(torch.optim.AdamW):
    pass


def build_optimizer(model: GPT, lr: float, weight_decay: float):
    """简化的单一 AdamW（本实验台比较的是**评测协议**，优化器必须对所有 arm 完全一致）。"""
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=weight_decay)


def lr_schedule(step: int, total: int, warmup: int, base: float, min_frac: float = 0.05):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def count_flops_per_token(model: GPT) -> float:
    """粗略的每 token 前向 FLOPs（2N），用于把评测开销换算成可比算力单位。"""
    return 2.0 * model.n_params()
