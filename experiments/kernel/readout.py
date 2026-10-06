"""Exact CPU readout for Qwen2.5-3B layer 35 (the last block).

After layer 35 only the final RMSNorm and the tied embedding rows for '0'/'1' are applied, and the
LoRA adapters only touch attention, so the answer margin is a fixed function of the layer-35 residual
at the answer position. Mirrors experiments.residual.projection + Qwen2 RMSNorm in bfloat16.
"""
import json
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ZERO, ONE = 15, 16  # token ids of '0' and '1'


@lru_cache
def weights():
    idx = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json")).read_text())["weight_map"]
    eps = json.loads(Path(hf_hub_download(MODEL, "config.json")).read_text())["rms_norm_eps"]
    get = lambda name: safe_open(hf_hub_download(MODEL, idx[name]), "pt")
    with get("model.embed_tokens.weight") as f:
        rows = f.get_slice("model.embed_tokens.weight")[[ZERO, ONE]]
    with get("model.norm.weight") as f:
        norm = f.get_tensor("model.norm.weight")
    return rows, norm, eps


def margin(h):
    """h: (n, 2048) layer-35 output at the answer position (bf16 values). Returns logit('1') - logit('0')."""
    rows, norm, eps = weights()
    x = h.to(torch.bfloat16).float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16) * norm
    logits = (x @ rows.T).float()
    return logits[:, 1] - logits[:, 0]


def ablate(h, q, c):
    """Same edit as experiments.residual.projection: remove the centred component along orthonormal q."""
    t = h.float()
    return (t - ((t - c.float()) @ q.float()) @ q.float().T).to(torch.bfloat16)
