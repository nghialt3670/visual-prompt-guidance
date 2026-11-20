"""InstructPix2Pix adapter for the generic spatial-attention Engine.

Native image-conditioned U-Net forward and 3-way text/image CFG follow
Diffusers StableDiffusionInstructPix2PixPipeline. Only attention extraction and
latent-gradient updates are added by the framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageOps

from ..contracts import AttentionRequest, Prediction, Request, Space, Target
from .instruct_pix2pix_attention import UNetAttentionCapture, object_tokens, cross_attention_layers, default_layers


@dataclass
class _State:
    latents: torch.Tensor
    image_latents: torch.Tensor
    embeddings: torch.Tensor
    timesteps: torch.Tensor
    token_indices: tuple[int, ...]
    original_size: tuple[int, int]
    native_size: tuple[int, int]
    text_guidance: float
    image_guidance: float
    cfg: bool
    step_kwargs: dict[str, Any]
    query_chunk: int


class InstructPix2Pix:
    """Image editing using timbrooks/instruct-pix2pix (Stable Diffusion U-Net).

    Request.data: {"image": PIL.Image, "prompt": str, "object_phrase": str}
    Request.options: steps, seed, max_side, cfg (text guidance, default 7.5),
      image_cfg (image guidance, default 1.5), negative_prompt, eta, query_chunk.

    Native model uses the input image as *conditioning*, not as an inpaint mask.
    All visual prompts guide only the latent through their spatial objectives.
    """
    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix", *,
                 device: str = "auto", dtype: str = "auto", cpu_offload: bool = False,
                 pipeline: Any | None = None):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self._pipe = pipeline

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        from diffusers import StableDiffusionInstructPix2PixPipeline
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        dtype = ((torch.float16 if device != "cpu" else torch.float32)
                 if self.dtype == "auto" else
                 {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[self.dtype])
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        if self.cpu_offload:
            if device != "cuda":
                raise ValueError("CPU offload currently requires CUDA")
            pipe.enable_model_cpu_offload(device=device)
        else:
            pipe.to(device)
        pipe.unet.eval().requires_grad_(False)
        self._pipe = pipe
        return pipe

    @staticmethod
    def _inputs(request: Request):
        if not isinstance(request.data, dict):
            raise TypeError("InstructPix2Pix expects Request.data as a dict")
        image = request.data.get("image")
        prompt = request.data.get("prompt")
        phrase = request.data.get("object_phrase")
        if not isinstance(image, Image.Image):
            raise TypeError("InstructPix2Pix requires an image (PIL.Image.Image)")
        if not isinstance(prompt, str) or not isinstance(phrase, str) or not prompt or not phrase:
            raise ValueError("Both prompt and object_phrase are required")
        return image, prompt, phrase

    def space(self, request: Request) -> Space:
        image, _, _ = self._inputs(request)
        w, h = image.size
        max_side = int(request.options.get("max_side", 512))
        if max_side < 64:
            raise ValueError("max_side must be >= 64")
        ratio = min(1., max_side / max(w, h))
        # 32px alignment accommodates the three U-Net downsampling stages
        # while keeping the original aspect ratio much closer than 64px steps.
        native_w = max(32, round(w * ratio / 32) * 32)
        native_h = max(32, round(h * ratio / 32) * 32)
        cap = max(32, (max_side // 32) * 32)
        native_w, native_h = min(native_w, cap), min(native_h, cap)
        return Space((w, h), (native_h // 16, native_w // 16))

    def prepare(self, request: Request, target: Target) -> _State:
        pipe = self._load()
        image, prompt, phrase = self._inputs(request)
        opts = request.options
        if opts.get("mode", "base") != "base":
            raise ValueError("InstructPix2Pix is an image-conditioned editor, not a native inpaint pipeline; choose mode='base'")
        steps = int(opts.get("steps", 30))
        cfg = float(opts.get("cfg", 7.5))
        image_cfg = float(opts.get("image_cfg", 1.5))
        eta = float(opts.get("eta", 0.))
        if steps < 1 or cfg < 1. or image_cfg < 1.:
            raise ValueError("Require steps >= 1 and text/image guidance scales >= 1")
        do_cfg = cfg > 1. and image_cfg >= 1.
        size = self.space(request)
        gh, gw = size.grid_size
        native_h, native_w = gh * 16, gw * 16
        image = ImageOps.exif_transpose(image).convert("RGB").resize(
            (native_w, native_h), Image.Resampling.LANCZOS
        )
        pipe.unet.eval().requires_grad_(False)
        device = pipe._execution_device
        generator = torch.Generator(device="cpu").manual_seed(int(opts.get("seed", 42)))
        tokens = object_tokens(pipe.tokenizer, prompt, phrase)
        with torch.no_grad():
            # Official InstructPix2Pix `_encode_prompt` emits embeddings in
            # [positive, negative+image, negative-no-image] batch order.
            embeddings = pipe._encode_prompt(
                prompt, device, 1, do_cfg,
                opts.get("negative_prompt", None),
            )
            processed = pipe.image_processor.preprocess(image)
            image_latents = pipe.prepare_image_latents(
                processed, 1, 1, embeddings.dtype, device, do_cfg,
            )
            pipe.scheduler.set_timesteps(steps, device=device)
            timesteps = pipe.scheduler.timesteps
            height = image_latents.shape[-2] * pipe.vae_scale_factor
            width = image_latents.shape[-1] * pipe.vae_scale_factor
            latents = pipe.prepare_latents(
                1, pipe.vae.config.latent_channels, height, width,
                embeddings.dtype, device, generator, None,
            )
            extra_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
        if latents.shape[-2:] != (native_h // pipe.vae_scale_factor, native_w // pipe.vae_scale_factor):
            raise RuntimeError("InstructPix2Pix latent grid mismatch")
        if target.density.shape[-2:] != (gh, gw):
            raise ValueError("Visual prompt target resolution does not match native image")
        if latents.shape[1] + image_latents.shape[1] != pipe.unet.config.in_channels:
            raise ValueError("InstructPix2Pix checkpoint UNet has incompatible input channels")
        return _State(
            latents=latents, image_latents=image_latents, embeddings=embeddings,
            timesteps=timesteps, token_indices=tokens,
            original_size=size.source_size, native_size=(native_w, native_h),
            text_guidance=cfg, image_guidance=image_cfg, cfg=do_cfg,
            step_kwargs=extra_kwargs, query_chunk=int(opts.get("query_chunk", 64)),
        )

    def steps(self, state: _State):
        return list(state.timesteps)

    def latent(self, state: _State) -> torch.Tensor:
        return state.latents

    def set_latent(self, state: _State, latent: torch.Tensor) -> None:
        state.latents = latent

    def attention_layers(self) -> tuple[str, ...]:
        """Discover names for `Guidance(layers=...)` and manual head policies."""
        return tuple(cross_attention_layers(self._load().unet))

    def _forward_unet(self, state: _State, step, *, conditional_only: bool) -> torch.Tensor:
        pipe = self._pipe
        latent_input = state.latents
        if state.cfg and not conditional_only:
            latent_input = torch.cat([latent_input] * 3)
            ref_latents = state.image_latents
            embeddings = state.embeddings
        else:
            ref_latents = state.image_latents[:1]
            embeddings = state.embeddings[:1]
        scaled = pipe.scheduler.scale_model_input(latent_input, step)
        model_input = torch.cat([scaled, ref_latents], dim=1)
        return pipe.unet(
            model_input, step, encoder_hidden_states=embeddings,
            return_dict=False,
        )[0]

    def forward(self, state: _State, step, *, attention: AttentionRequest | None) -> Prediction:
        # Only compute the positive branch when gradients are enabled: the
        # engine discards this prediction and differentiates attention maps.
        guiding = torch.is_grad_enabled() and attention is not None
        if attention is None:
            pred = self._forward_unet(state, step, conditional_only=False)
            batch = None
        else:
            collector = UNetAttentionCapture(
                self._pipe.unet, state.token_indices,
                state.latents.shape[-2:],
                (state.native_size[1] // 16, state.native_size[0] // 16),
                attention.layers,
                affinity_layers=attention.affinity_layers,
                need_affinity=attention.need_affinity,
                query_chunk=state.query_chunk,
            )
            with collector.capture():
                pred = self._forward_unet(state, step, conditional_only=guiding)
            batch = collector.result()
        if state.cfg and not guiding:
            t, i, u = pred.chunk(3)
            pred = u + state.text_guidance * (t - i) + state.image_guidance * (i - u)
        return Prediction(pred, batch)

    def advance(self, state: _State, step, prediction: Prediction) -> None:
        state.latents = self._pipe.scheduler.step(
            prediction.output, step, state.latents,
            **state.step_kwargs, return_dict=False,
        )[0]

    def decode(self, state: _State) -> Image.Image:
        pipe = self._pipe
        with torch.no_grad():
            latents = state.latents / pipe.vae.config.scaling_factor
            pixels = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype), return_dict=False)[0]
            # Respect the checkpoint's configured safety checker, if present.
            if hasattr(pipe, "run_safety_checker"):
                pixels, flagged = pipe.run_safety_checker(
                    pixels, pipe._execution_device, state.embeddings.dtype
                )
            else:
                flagged = None
            normalization = ([not bool(x) for x in flagged] if flagged is not None else [True])
            output = pipe.image_processor.postprocess(pixels, output_type="pil", do_denormalize=normalization)[0]
            if output.size != state.original_size:
                output = output.resize(state.original_size, Image.Resampling.LANCZOS)
            if hasattr(pipe, "maybe_free_model_hooks"):
                pipe.maybe_free_model_hooks()
            return output
