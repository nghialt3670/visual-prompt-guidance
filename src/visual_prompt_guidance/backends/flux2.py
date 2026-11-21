"""FLUX.2 Klein backend. All Diffusers-specific code is confined here.

This is an implementation of the public ModelAdapter protocol, not a dependency
of the framework core. Both base editing and native inpainting are supported.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from ..contracts import Request, Space, Target, AttentionBatch, AttentionRequest, Prediction
from .flux2_attention import AttentionCollector, find_object_token_indices


@dataclass
class _KleinState:
    latent_data: torch.Tensor
    latent_ids: torch.Tensor
    reference: torch.Tensor
    reference_ids: torch.Tensor
    prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    negative_embeds: torch.Tensor | None
    negative_ids: torch.Tensor | None
    schedule: torch.Tensor
    token_indices: list[int]
    grid: tuple[int, int]
    native_size: tuple[int, int]
    original_size: tuple[int, int]
    cfg: float
    mode: str
    edit_mask: torch.Tensor
    source_latents: torch.Tensor | None
    noise: torch.Tensor | None
    position: int = 0


class Flux2Klein:
    """Model-specific options are Request.options, not fields on Guidance/Engine.

    Required Request.data: dict with image (PIL), prompt, object_phrase.
    Optional Request.options: steps, seed, mode, inpaint_strength, cfg,
    max_side, max_text_tokens, layers, query_chunk.
    """
    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-klein-base-4B", *,
                 device: str = "auto", dtype: str = "auto", cpu_offload: bool = False):
        self.model_id, self.device, self.dtype, self.cpu_offload = model_id, device, dtype, cpu_offload
        self._pipe = None
        self._loaded_mode = None

    def attention_catalog(self) -> dict:
        """Describe head IDs using only the tiny transformer/config.json file."""
        from .attention_catalog import component_config, flux_catalog
        return flux_catalog(component_config(self.model_id, 'transformer'))

    def _load(self, mode: str):
        from diffusers import Flux2KleinPipeline, Flux2KleinInpaintPipeline
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = (torch.float32 if device == "cpu" else torch.bfloat16) if self.dtype == "auto" else {
            "fp32": torch.float32, "bf16": torch.bfloat16,
        }[self.dtype]
        cls = Flux2KleinInpaintPipeline if mode == "inpaint" else Flux2KleinPipeline
        pipe=cls.from_pretrained(self.model_id, torch_dtype=dtype)
        if self.cpu_offload:
            if device != "cuda":
                raise ValueError("CPU offload is only supported for CUDA")
            pipe.enable_model_cpu_offload(device=device)
        else:
            pipe.to(device)
        pipe.transformer.eval().requires_grad_(False)
        pipe.vae.eval().requires_grad_(False)
        self._pipe, self._loaded_mode = pipe,mode
        return pipe

    @staticmethod
    def _get(request: Request):
        if not isinstance(request.data, dict):
            raise TypeError("Klein backend expects Request.data as a mapping")
        image = request.data.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError("Klein requires a PIL source image")
        return image, request.data.get("prompt"), request.data.get("object_phrase")

    def space(self, request: Request) -> Space:
        image,_,_=self._get(request)
        w,h=image.size
        max_side=int(request.options.get("max_side",512))
        if max_side < 16:
            raise ValueError("max_side must be >= 16")
        scale=min(1., max_side/max(w,h))
        native_w=max(16,round(w*scale/16)*16)
        native_h=max(16,round(h*scale/16)*16)
        return Space((w,h),(native_h//16,native_w//16))

    def prepare(self, request: Request, target: Target) -> _KleinState:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        image,prompt,phrase=self._get(request)
        if not prompt or not phrase:
            raise ValueError("Klein requires prompt and object_phrase")
        opts=request.options
        mode=opts.get("mode","base")
        if mode not in {"base","inpaint"}:
            raise ValueError("Klein mode must be base or inpaint")
        if self._pipe is None or self._loaded_mode != mode:
            self._load(mode)
        pipe=self._pipe
        size=self.space(request)
        gh,gw=size.grid_size
        h,w=gh*16,gw*16
        image=ImageOps.exif_transpose(image).convert("RGB").resize((w,h),Image.Resampling.LANCZOS)
        num_steps=int(opts.get("steps",30)); seed=int(opts.get("seed",42))
        cfg=float(opts.get("cfg",4.)); strength=float(opts.get("inpaint_strength",.8))
        token_limit=int(opts.get("max_text_tokens",256))
        if num_steps<1 or not 0<strength<=1 or cfg<1:
            raise ValueError("Invalid FLUX sampling configuration")
        device=pipe._execution_device
        generator=torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            prompt_embeds,text_ids=pipe.encode_prompt(
                prompt=prompt,device=device,num_images_per_prompt=1,
                max_sequence_length=token_limit,text_encoder_out_layers=(9,18,27))
            neg_embeds=neg_ids=None
            if cfg>1:
                neg_embeds,neg_ids=pipe.encode_prompt(
                    prompt="",device=device,num_images_per_prompt=1,
                    max_sequence_length=token_limit,text_encoder_out_layers=(9,18,27))
            tokens=find_object_token_indices(pipe.tokenizer,prompt,phrase,max_length=token_limit)
            processed=pipe.image_processor.preprocess(image,height=h,width=w,resize_mode="crop")
            seq_len=gh*gw
            sigmas=np.linspace(1.,1./num_steps,num_steps)
            if getattr(pipe.scheduler.config,"use_flow_sigmas",False):
                sigmas=None
            mu=compute_empirical_mu(seq_len,num_steps)
            schedule,_=retrieve_timesteps(pipe.scheduler,num_steps,device,sigmas=sigmas,mu=mu)
            if mode == "inpaint":
                schedule,_=pipe.get_timesteps(num_steps,strength,device)
                if len(schedule)<1:
                    raise ValueError("Inpaint strength yields zero steps")
                latents,noise,source_latents,encoded,latent_ids=pipe.prepare_latents(
                    processed,schedule[:1].repeat(1),1,pipe.transformer.config.in_channels//4,
                    h,w,prompt_embeds.dtype,device,generator,None)
                ref,ref_ids=pipe.prepare_image_latents(
                    images=[encoded],batch_size=1,generator=generator,device=device,dtype=prompt_embeds.dtype)
            else:
                latents,latent_ids=pipe.prepare_latents(
                    batch_size=1,num_latents_channels=pipe.transformer.config.in_channels//4,
                    height=h,width=w,dtype=prompt_embeds.dtype,device=device,
                    generator=generator,latents=None)
                ref,ref_ids=pipe.prepare_image_latents(
                    images=[processed],batch_size=1,generator=generator,device=device,dtype=pipe.vae.dtype)
                source_latents=noise=None
                if hasattr(pipe.scheduler,"set_begin_index"):
                    pipe.scheduler.set_begin_index(0)
        if latents.shape[1] != seq_len:
            raise RuntimeError("Klein latent grid mismatch")
        if target.edit_mask.shape != (1,1,gh,gw):
            raise ValueError("Klein edit mask must match native grid")
        return _KleinState(latents,latent_ids,ref.to(dtype=pipe.transformer.dtype),ref_ids,
                           prompt_embeds,text_ids,neg_embeds,neg_ids,schedule,tokens,
                           (gh,gw),(w,h),size.source_size,cfg,mode,
                           target.edit_mask.to(device,latents.dtype).flatten(1),
                           source_latents,noise)

    def steps(self,state: _KleinState):
        return list(state.schedule)

    def latent(self,state: _KleinState) -> torch.Tensor:
        return state.latent_data

    def set_latent(self,state: _KleinState,latent: torch.Tensor) -> None:
        state.latent_data=latent

    def _predict(self,state: _KleinState, step: torch.Tensor, conditional: bool) -> torch.Tensor:
        pipe=self._pipe
        latents=state.latent_data.to(dtype=pipe.transformer.dtype)
        hidden=torch.cat([latents,state.reference.to(dtype=latents.dtype)],dim=1)
        ids=torch.cat([state.latent_ids,state.reference_ids],dim=1)
        embeds=state.prompt_embeds if conditional else state.negative_embeds
        text_ids=state.text_ids if conditional else state.negative_ids
        with pipe.transformer.cache_context("cond" if conditional else "uncond"):
            pred=pipe.transformer(
                hidden_states=hidden,timestep=step.expand(latents.shape[0]).to(latents.dtype)/1000,
                guidance=None,encoder_hidden_states=embeds,txt_ids=text_ids,
                img_ids=ids,joint_attention_kwargs=None,return_dict=False)[0]
        return pred[:,:state.latent_data.shape[1]]

    def forward(self,state: _KleinState,step: torch.Tensor,*,attention: AttentionRequest | None) -> Prediction:
        pipe=self._pipe
        batch=None
        if attention is None:
            conditional_pred=self._predict(state,step,True)
        else:
            blocks=list(attention.layers or tuple(f"D{i}" for i in range(len(pipe.transformer.transformer_blocks))))
            self_blocks=list(attention.affinity_layers or tuple(blocks)) if attention.need_affinity else []
            collector=AttentionCollector(
                pipe.transformer,state.token_indices,state.latent_data.shape[1],blocks,
                query_chunk=int(getattr(self,"query_chunk",64)), self_blocks=self_blocks,capture_heads=True)
            with collector.capture(text_len=state.prompt_embeds.shape[1],capture_self=attention.need_affinity):
                conditional_pred=self._predict(state,step,True)
            heads=collector.heads_tensor()
            gh,gw=state.grid
            head_maps=heads.reshape(heads.shape[0],heads.shape[1],gh,gw)
            labels=tuple(collector.head_labels())
            affinity=collector.mean_self_map() if attention.need_affinity else None
            batch=AttentionBatch(head_maps,labels,affinity)
        if state.cfg>1:
            # Avoid constructing an unnecessary unconditional gradient graph.
            # During differentiable sampling the negative branch contributes
            # to the derivative of the denoising prediction. During ordinary
            # guidance, computing it without a graph saves substantial memory.
            if torch.is_grad_enabled() and attention is None:
                negative_pred=self._predict(state,step,False)
            else:
                with torch.no_grad():
                    negative_pred=self._predict(state,step,False)
            conditional_pred=negative_pred+state.cfg*(conditional_pred-negative_pred)
        return Prediction(conditional_pred,batch)

    def advance(self,state: _KleinState,step: torch.Tensor,prediction: Prediction) -> None:
        pipe=self._pipe
        latents=pipe.scheduler.step(prediction.output,step,state.latent_data,return_dict=False)[0]
        if state.mode == "inpaint":
            original=state.source_latents.to(latents.dtype)
            if state.position < len(state.schedule)-1:
                t_next=state.schedule[state.position+1]
                original=pipe.scheduler.scale_noise(original,t_next.reshape(1),state.noise.to(original.dtype))
            mask=state.edit_mask.to(latents.dtype).unsqueeze(-1)
            latents=(1-mask)*original+mask*latents
        state.latent_data=latents
        state.position+=1

    def decode_tensor(self, state: _KleinState) -> torch.Tensor:
        """Differentiable FLUX.2 VAE decode, RGB float [B,3,H,W] in [0,1]."""
        import torch.nn.functional as F
        from diffusers.utils.torch_utils import get_module_execution_device
        pipe = self._pipe
        gh, gw = state.grid
        try:
            unpacked = pipe._unpack_latents_with_ids(state.latent_data, state.latent_ids)
        except TypeError:
            unpacked = pipe._unpack_latents_with_ids(state.latent_data, state.latent_ids, gh, gw)
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
            + pipe.vae.config.batch_norm_eps
        )
        latents = pipe._unpatchify_latents(unpacked * std + mean)
        latents = latents.to(get_module_execution_device(pipe.vae), dtype=pipe.vae.dtype)
        pixels = pipe.vae.decode(latents, return_dict=False)[0]
        pixels = (pixels.float() / 2 + 0.5).clamp(0, 1)
        original_w, original_h = state.original_size
        if pixels.shape[-2:] != (original_h, original_w):
            pixels = F.interpolate(pixels, size=(original_h, original_w),
                                   mode="bilinear", align_corners=False)
        return pixels

    def decode(self,state: _KleinState) -> Image.Image:
        from diffusers.utils.torch_utils import get_module_execution_device
        pipe=self._pipe
        gh,gw=state.grid
        with torch.no_grad():
            try:
                unpacked=pipe._unpack_latents_with_ids(state.latent_data,state.latent_ids)
            except TypeError:
                unpacked=pipe._unpack_latents_with_ids(state.latent_data,state.latent_ids,gh,gw)
            mean=pipe.vae.bn.running_mean.view(1,-1,1,1).to(unpacked.device,unpacked.dtype)
            std=torch.sqrt(pipe.vae.bn.running_var.view(1,-1,1,1).to(unpacked.device,unpacked.dtype)+
                           pipe.vae.config.batch_norm_eps)
            unpacked=pipe._unpatchify_latents(unpacked*std+mean)
            unpacked=unpacked.to(get_module_execution_device(pipe.vae),dtype=pipe.vae.dtype)
            pixels=pipe.vae.decode(unpacked,return_dict=False)[0]
            output=pipe.image_processor.postprocess(pixels,output_type="pil")[0]
            if output.size!=state.original_size:
                output=output.resize(state.original_size,Image.Resampling.LANCZOS)
            return output
