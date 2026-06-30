"""Self-contained nano-case model — no dependencies beyond torch + safetensors.

A ~1M-parameter byte-level decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU)
that converts a messy identifier to a target case style — and, crucially, segments
boundary-DESTROYED inputs (no separators, single global case: `sdkmodel`,
`SENDREQUEST`) that a regex cannot split. This single file vendors the exact
architecture the model was trained with, so you can load and run the published
weights without the training lab.

    python modeling_nano_case.py            # runs a few examples
    # or, from your own code:
    from modeling_nano_case import load, to_case
    m = load("model.safetensors", "config.json")
    print(to_case(m, "const", "sdkmodel"))   # -> SDK_MODEL
    print(to_case(m, "camel", "usertablehandler"))  # -> userTableHandler

Prompt format the model was trained on (byte-for-byte):

    <case> | <messy identifier> => <answer><newline>

`<case>` is one of: snake, kebab, camel, pascal, const. The answer ends at the
first newline (byte 10), the supervised EOS — `to_case()` decodes a fixed budget
and cuts there.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def apply(self, x, offset: int = 0):
        seq = x.size(-2)
        cos = self.cos[offset:offset + seq]
        sin = self.sin[offset:offset + seq]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack((rot1, rot2), dim=-1).flatten(-2).type_as(x)


class GQA(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, positional):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.n_rep = n_heads // n_kv_heads
        self.positional = positional
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, mask):
        b, seq, _ = x.shape
        q = self.q_proj(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.positional.apply(q)
        k = self.positional.apply(k)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, positional):
        super().__init__()
        hidden = int(cfg["dim"] * cfg["ffn_mult"])
        self.attn_norm = RMSNorm(cfg["dim"], cfg["norm_eps"])
        self.attn = GQA(cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], positional)
        self.ffn_norm = RMSNorm(cfg["dim"], cfg["norm_eps"])
        self.ffn = SwiGLU(cfg["dim"], hidden)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NanoCase(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.positional = RoPE(cfg["head_dim"], cfg["max_seq_len"], cfg["rope_theta"])
        self.blocks = nn.ModuleList([Block(cfg, self.positional) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["dim"], cfg["norm_eps"])
        self.lm_head = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

    def forward(self, tokens):
        seq = tokens.size(1)
        x = self.tok_emb(tokens)
        mask = torch.triu(torch.full((seq, seq), float("-inf"), device=tokens.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.final_norm(x))


def load(weights="model.safetensors", config="config.json", device="cpu"):
    from safetensors.torch import load_file
    with open(config) as f:
        cfg = json.load(f)
    model = NanoCase(cfg).to(device)
    sd = load_file(weights)
    sd["lm_head.weight"] = sd["tok_emb.weight"]  # restore tied weight
    model.load_state_dict(sd)
    model.eval()
    return model


_EOS = 10  # newline terminates the answer


@torch.no_grad()
def to_case(model, case: str, ident: str, device="cpu", max_new: int = 40) -> str:
    """`case` in {snake,kebab,camel,pascal,const}, `ident` a messy identifier.

    Returns the identifier rendered in `case`. Decodes greedily and stops at the
    newline EOS the model was trained to emit."""
    prompt = f"{case} | {ident} => "
    toks = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    max_seq = model.cfg["max_seq_len"]
    out = []
    for _ in range(max_new):
        nxt = int(model(toks[:, -max_seq:])[:, -1, :].argmax(-1))
        if nxt == _EOS:
            break
        out.append(nxt)
        toks = torch.cat([toks, torch.tensor([[nxt]], device=device)], dim=1)
    return bytes(b & 0xFF for b in out).decode("utf-8", "replace")


if __name__ == "__main__":
    m = load()
    examples = [
        ("const", "sdkmodel"), ("camel", "usertablehandler"), ("kebab", "sqlqueryname"),
        ("snake", "md5cache"), ("pascal", "rendertokenerror"), ("const", "NUMBERRESULTCODE"),
        ("camel", "user_table_handler"), ("snake", "HTTPRequestHandler"),
    ]
    for case, ident in examples:
        print(f"{case:7} {ident:<22} -> {to_case(m, case, ident)}")
