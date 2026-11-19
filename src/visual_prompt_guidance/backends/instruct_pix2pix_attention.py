"""Gradient-preserving U-Net attention extraction for InstructPix2Pix.

Uses short-lived forward pre-hooks on Diffusers' `attn2` cross-attention
modules. Native attention processors are left untouched. Maps are resized to
one common guidance grid; optional `attn1` self-attention provides affinity.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def object_tokens(tokenizer, prompt: str, phrase: str) -> tuple[int, ...]:
    """Find the CLIP token positions of a phrase in the *actual* prompt.

    Prefer fast-tokenizer character offsets, falling back to token-subsequence
    matching for slow CLIP tokenizers (which do not implement offsets).
    """
    start = prompt.casefold().find(phrase.casefold())
    if not phrase.strip() or start < 0:
        raise ValueError(f"Object phrase {phrase!r} must occur in prompt {prompt!r}")
    end = start + len(phrase)
    max_len = min(int(tokenizer.model_max_length), 77)
    try:
        result = tokenizer(
            prompt, padding="max_length", max_length=max_len, truncation=True,
            return_offsets_mapping=True,
        )
        offsets = result["offset_mapping"]
        # Fast tokenizers sometimes return a batch dimension.
        if offsets and isinstance(offsets[0][0], (tuple, list)):
            offsets = offsets[0]
        matches = tuple(i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a)
        if matches:
            return matches
    except (NotImplementedError, TypeError, ValueError, KeyError):
        pass

    encoded = tokenizer(prompt, padding="max_length", max_length=max_len, truncation=True)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    for text in (phrase, " " + phrase):
        cand = tokenizer(text, add_special_tokens=False)["input_ids"]
        if cand and isinstance(cand[0], list):
            cand = cand[0]
        for i in range(len(ids) - len(cand) + 1):
            if list(ids[i:i + len(cand)]) == list(cand) and cand:
                return tuple(range(i, i + len(cand)))
    raise ValueError(
        f"Could not locate the CLIP tokens for {phrase!r}. Use a short prompt "
        "containing the exact object phrase."
    )


def cross_attention_layers(unet) -> dict[str, torch.nn.Module]:
    """Return named Diffusers U-Net cross-attention modules."""
    return {
        name: module for name, module in unet.named_modules()
        if name.endswith(".attn2") and hasattr(module, "to_q") and hasattr(module, "to_k")
    }


def default_layers(unet) -> tuple[str, ...]:
    """A small default: prefer the 2nd down- and 2nd up-scale cross-attn layers.

    Prefer head maps near half the latent grid's resolution rather than all
    cross-attention blocks (which increases memory substantially).
    """
    names = tuple(cross_attention_layers(unet))
    preferred = tuple(n for n in names if (
        n.startswith("down_blocks.1.attentions.0.")
        or n.startswith("up_blocks.2.attentions.0.")
    ))
    return preferred[:2] or names[len(names) // 2: len(names) // 2 + 1]


def _project(attn, hidden_states, encoder_hidden_states=None):
    """Recompute Q,K in the form used by Diffusers' default SD processors."""
    b = hidden_states.shape[0]
    if hidden_states.ndim == 4:
        _, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
    if getattr(attn, "group_norm", None) is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if encoder_hidden_states is not None and getattr(attn, "norm_cross", False):
        context = attn.norm_encoder_hidden_states(context)
    q = attn.to_q(hidden_states)
    k = attn.to_k(context)
    heads = int(attn.heads)
    if q.shape[-1] % heads != 0 or k.shape[-1] % heads != 0:
        raise RuntimeError("Invalid U-Net attention head size")
    q = q.reshape(b, -1, heads, q.shape[-1] // heads).permute(0, 2, 1, 3)
    k = k.reshape(b, -1, heads, k.shape[-1] // heads).permute(0, 2, 1, 3)
    if getattr(attn, "norm_q", None) is not None:
        q = attn.norm_q(q)
    if getattr(attn, "norm_k", None) is not None:
        k = attn.norm_k(k)
    return q.float(), k.float()


def _spatial_shape(tokens: int, latent_size: tuple[int, int]) -> tuple[int, int]:
    """Infer U-Net spatial token size from known input latent size."""
    lh, lw = latent_size
    for downsample in (1, 2, 4, 8, 16):
        # Diffusers stride-2 downsample convolutions round *up* for odd inputs.
        h, w = math.ceil(lh / downsample), math.ceil(lw / downsample)
        if h > 0 and w > 0 and h * w == tokens:
            return h, w
    raise ValueError(f"Unrecognized U-Net attention grid ({tokens} tokens) for latent {latent_size}")


class UNetAttentionCapture:
    """Collect selected individual cross-attention heads on an output target grid.

    Attention probabilities are computed for the *positive prompt branch only*
    during native 3-way InstructPix2Pix CFG. This collector does not replace
    UNet's attention processor.
    """
    def __init__(self, unet, tokens: tuple[int, ...], latent_size: tuple[int, int],
                 target_size: tuple[int, int], layers: tuple[str, ...] | None,
                 *, affinity_layers: tuple[str, ...] | None = None,
                 need_affinity: bool = False, query_chunk: int = 64):
        self.tokens = tokens
        self.latent_size = latent_size
        self.target_size = target_size
        self.query_chunk = query_chunk
        all_cross = cross_attention_layers(unet)
        if not all_cross:
            raise ValueError("U-Net has no cross-attention modules named *.attn2")
        self.layers = tuple(layers) if layers is not None else default_layers(unet)
        if not self.layers:
            raise ValueError("No U-Net attention layers selected")
        missing = set(self.layers) - set(all_cross)
        if missing:
            raise ValueError(f"Unknown InstructPix2Pix attention layer(s): {sorted(missing)}")
        self.cross = {name: all_cross[name] for name in self.layers}
        self.self_layers = tuple(affinity_layers or self.layers) if need_affinity else ()
        self.self_modules = {}
        named = dict(unet.named_modules())
        for name in self.self_layers:
            candidate = name.removesuffix(".attn2") + ".attn1"
            if candidate not in named or not hasattr(named[candidate], "to_q"):
                raise ValueError(f"Self-attention module missing for {name}")
            self.self_modules[name] = named[candidate]
        self._maps = {}
        self._affinity = {}
        self._labels = []

    def _capture_cross(self, label, attn, hidden, context):
        q, k = _project(attn, hidden[:1], context[:1])
        if max(self.tokens) >= k.shape[-2]:
            raise ValueError("Object token index exceeds available text context")
        spatial = _spatial_shape(q.shape[-2], self.latent_size)
        # Query chunks bound the temporary QK matrix, even at 64x64 attention.
        idx = torch.as_tensor(self.tokens, dtype=torch.long, device=q.device)
        key_t = k.transpose(-1, -2)
        parts = []
        for start in range(0, q.shape[-2], self.query_chunk):
            logits = (q[:, :, start:start + self.query_chunk] @ key_t) * float(getattr(attn, "scale", q.shape[-1] ** -0.5))
            mass = torch.exp(logits.index_select(-1, idx).logsumexp(-1) - logits.logsumexp(-1))
            parts.append(mass)
        maps = torch.cat(parts, -1).reshape(1, q.shape[1], *spatial)
        if spatial != self.target_size:
            maps = F.interpolate(maps, size=self.target_size, mode="bilinear", align_corners=False)
        self._maps[label] = maps

    @torch.no_grad()
    def _capture_self(self, label, attn, hidden):
        q, k = _project(attn, hidden[:1])
        spatial = _spatial_shape(q.shape[-2], self.latent_size)
        if spatial != self.target_size:
            raise ValueError(
                f"Affinity layer {label!r} is at grid {spatial}; expected {self.target_size}. "
                "Choose an attention layer matching the target grid or disable propagation."
            )
        key_t = k.transpose(-1, -2)
        pieces = []
        for start in range(0, q.shape[-2], self.query_chunk):
            logits = (q[:, :, start:start + self.query_chunk] @ key_t) * float(getattr(attn, "scale", q.shape[-1] ** -0.5))
            pieces.append(logits.softmax(-1).mean(1))
        self._affinity[label] = torch.cat(pieces, 1)

    @contextmanager
    def capture(self):
        handles = []
        try:
            for name, attn in self.cross.items():
                def hook(module, args, kwargs, label=name):
                    if not args and "hidden_states" not in kwargs:
                        raise RuntimeError("UNet cross-attention hook did not receive hidden states")
                    hidden = args[0] if args else kwargs["hidden_states"]
                    context = kwargs.get("encoder_hidden_states")
                    if context is None and len(args) > 1:
                        context = args[1]
                    if context is None:
                        raise RuntimeError(f"U-Net cross-attention {label} missing text conditioning")
                    self._capture_cross(label, module, hidden, context)
                handles.append(attn.register_forward_pre_hook(hook, with_kwargs=True))
            for name, attn in self.self_modules.items():
                def hook(module, args, kwargs, label=name):
                    hidden = args[0] if args else kwargs["hidden_states"]
                    self._capture_self(label, module, hidden)
                handles.append(attn.register_forward_pre_hook(hook, with_kwargs=True))
            yield self
        finally:
            for h in handles:
                h.remove()

    def result(self):
        from ..contracts import AttentionBatch
        if set(self._maps) != set(self.layers):
            raise RuntimeError(f"Some U-Net attention layers were not called: {sorted(set(self.layers) - set(self._maps))}")
        tensors, labels = [], []
        for name in self.layers:
            head = self._maps[name]
            tensors.append(head)
            labels.extend(f"{name}:H{i:02d}" for i in range(head.shape[1]))
        affinity = None
        if self.self_modules:
            if set(self._affinity) != set(self.self_layers):
                raise RuntimeError("Some U-Net self-attention layers were not called")
            affinity = torch.stack([self._affinity[name] for name in self.self_layers]).mean(0)
        return AttentionBatch(torch.cat(tensors, dim=1), tuple(labels), affinity)
