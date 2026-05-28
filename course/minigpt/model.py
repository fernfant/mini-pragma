"""
A from-scratch GPT — the SAME architecture as GPT-3, just small enough to train
on one Mac. This is your nursery-rhyme CharGPT (lesson 4g) with the real grown-up
recipe bolted on:

  - decoder-only Transformer blocks, pre-LayerNorm (LN before attention & MLP)
  - multi-head causal self-attention (look left only)
  - GELU feed-forward, 4x wider in the middle
  - learned token + position embeddings
  - weight tying (the input embedding table IS the output head)
  - dropout

GPT-3 is literally this, with the dials turned up (Table 2.1 of the GPT-3 paper
defines 8 sizes from 125M to 175B — all this exact code). Heavily inspired by
Karpathy's nanoGPT.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256      # context length (how many tokens it sees at once)
    vocab_size: int = 50257    # tiktoken gpt2 BPE vocabulary
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384          # d_model — the "width" of every vector
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)  # q,k,v in one go
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # flash-style attention with the causal mask baked in
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """pre-LN: x = x + attn(ln(x)); x = x + mlp(ln(x))"""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # position embeddings
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer)),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying

        self.apply(self._init_weights)
        # scaled init on residual projections (GPT-2/3 trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()  # report non-position params like the paper

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        # inference: only need the last position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, lr, betas):
        # decay 2D weights (matmuls, embeddings); don't decay biases/LayerNorms
        decay, nodecay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


# named presets sized for a fanless M4 Air (16 GB)
PRESETS = {
    # quick end-to-end sanity check (~10M params, trains in minutes)
    "shakespeare": GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
    # the real "watch it write English" run (~30M params, a few hours on M4 Air).
    # block 256 keeps the fanless Air happy; bump to 512 + n_embd 512 if you're patient.
    "tinystories": GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1),
    # a true GPT-3 "Small" config (125M) — provided for reference; needs a real GPU/cloud
    "gpt3-small": GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0),
}
