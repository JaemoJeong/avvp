"""
Temporal Context Injection (TCI) — Full Self-Attention with Identity Projections

Each segment attends to ALL segments (including itself):
    Q = f_t,  K = f_all,  V = f_all   (no learned projections)
    attn = softmax(Q · K^T / √d)       (T, T)
    f̂_t = f_t + attn · V              (residual injection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalContextInjector(nn.Module):
    """
    Injects global temporal context into per-frame features via sigmoid gating.
    Uses frozen Wq, Wk, Wv from CLIP/CLAP's last transformer layer.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    @torch.no_grad()
    def forward(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Full self-attention across segments (Identity projections).
        
        Q = f_t, K = f_all, V = f_all
        attn = softmax(Q · K^T / √d)   # (T, T)
        f̂_t = f_t + attn · V           # residual
        """
        seg = segments.float()
        d = seg.shape[-1]
        scale = math.sqrt(d)

        attn_scores = (seg @ seg.T) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (T, T)
        context = attn_weights @ seg                     # (T, d)
        output = seg + context

        return output.to(segments.dtype)


def extract_clip_qkv(clip_model):
    last_block = clip_model.visual.transformer.resblocks[-1]
    in_proj_weight = last_block.attn.in_proj_weight
    d = in_proj_weight.shape[1]
    Wq = in_proj_weight[0:d, :].T.cpu()
    Wk = in_proj_weight[d:2*d, :].T.cpu()
    Wv = in_proj_weight[2*d:3*d, :].T.cpu()
    return Wq.detach(), Wk.detach(), Wv.detach()


def extract_clap_qkv(clap_model):
    last_layer = clap_model.model.audio_branch.layers[-1]
    last_block = last_layer.blocks[-1]
    qkv_weight = last_block.attn.qkv.weight
    d = qkv_weight.shape[1]
    Wq = qkv_weight[0:d, :].T.cpu()
    Wk = qkv_weight[d:2*d, :].T.cpu()
    Wv = qkv_weight[2*d:3*d, :].T.cpu()
    return Wq.detach(), Wk.detach(), Wv.detach()
