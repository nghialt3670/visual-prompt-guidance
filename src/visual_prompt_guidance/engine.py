"""Generic gradient-guidance engine. Contains no model, prompt or loss names."""
from __future__ import annotations
from dataclasses import dataclass, replace
from contextlib import nullcontext
from typing import Callable, Sequence
import torch
from .contracts import (
    Request, Result, StepEvent, ModelAdapter, VisualPrompt, TargetTransform,
    HeadPolicy, Observer, AttentionRequest,
)
from .heads import Uniform, aggregate
from .objectives import Objective


@dataclass(frozen=True)
class Guidance:
    """Only universal guidance controls. Model-specific options belong to Request."""
    strength: float = .01
    max_steps: int | None = 10
    layers: tuple[str,...] | None = None
    affinity_layers: tuple[str,...] | None = None
    normalize_gradient: bool = True


def rms_update(latent: torch.Tensor, gradient: torch.Tensor, cfg: Guidance) -> tuple[torch.Tensor, float]:
    """Latent guidance with stable higher-order derivatives at zero gradient.

    Clamping *after* sqrt() produces NaN second-order gradients at exactly
    zero RMS (sqrt'(0) is infinite, yielding 0 * inf during backward).
    Regularize the squared RMS before sqrt so both the normalized update and
    its derivative remain finite; do not detach the normalization term.
    """
    g = gradient.float()
    if not bool(torch.isfinite(g).all().item()):
        raise FloatingPointError("Non-finite inner guidance gradient (before latent update)")
    mean_square = g.square().mean()
    scale = mean_square.clamp_min(1e-16).sqrt() if cfg.normalize_gradient else 1.0
    updated = latent.float() - cfg.strength * g / scale
    # Monitoring should report the actual RMS (zero is a valid measurement).
    observed_rms = float(mean_square.detach().sqrt().item())
    return updated.to(dtype=latent.dtype), observed_rms


class Engine:
    """Inject backend, loss object, head policy, optional transform & observers.

    No global registries or implicit prompt/loss defaults. Every component may be
    replaced by an application. Compatible backends explicitly support
    differentiable latents and head-resolved spatial attention.
    """
    def __init__(
        self, backend: ModelAdapter, objective: Objective | None, *,
        guidance: Guidance = Guidance(),
        policy: HeadPolicy | None = None,
        scoring: Objective | None = None,
        transform: TargetTransform | None = None,
        observers: Sequence[Observer] = (),
        updater: Callable = rms_update,
    ):
        if guidance.strength < 0 or (guidance.max_steps is not None and guidance.max_steps < 0):
            raise ValueError("Invalid guidance settings")
        self.backend, self.objective, self.guidance = backend, objective, guidance
        self.policy = policy if policy is not None else Uniform()
        self.scoring = scoring if scoring is not None else objective
        self.transform, self.observers, self.updater = transform, tuple(observers), updater

    def run(self, request: Request, prompt: VisualPrompt) -> Result:
        """Standard low-memory inference; detaches latent updates and uses no-grad sampling."""
        return self._run(request, prompt, differentiable=False)

    def run_differentiable(self, request: Request, prompt: VisualPrompt) -> Result:
        """Higher-order differentiable sampling for head-weight training.

        output is a [B,3,H,W] float image tensor in [0,1] with an autograd
        path through every denoising step AND each inner latent guidance step.
        This intentionally retains the full computation graph; model adapters
        must provide a differentiable decode_tensor(state).
        """
        if not hasattr(self.backend, "decode_tensor"):
            raise TypeError("Differentiable sampling requires backend.decode_tensor(state)")
        if self.policy.needs_feedback:
            raise ValueError("Differentiable training requires a non-adaptive head policy")
        if not self.guidance.strength or self.guidance.max_steps == 0:
            raise ValueError("Differentiable training needs at least one guided sampling step")
        # Only the inner guidance gradient requires second-order attention.
        # Native post-guidance denoising uses PyTorch's memory-efficient SDPA.
        return self._run(request, prompt, differentiable=True)

    @staticmethod
    def _guidance_attention_context(differentiable: bool, current: torch.Tensor):
        if differentiable and current.is_cuda:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            return sdpa_kernel(backends=[SDPBackend.MATH])
        return nullcontext()

    def _run(self, request: Request, prompt: VisualPrompt, *, differentiable: bool) -> Result:
        space = self.backend.space(request)
        target = prompt.project(space)
        state = self.backend.prepare(request, target)
        steps = self.backend.steps(state)
        if not steps:
            raise ValueError("Backend prepared an empty sampling schedule")
        target = target.to(self.backend.latent(state).device)
        self.policy.reset()
        if self.transform is not None:
            self.transform.reset(target)
        records=[]
        cfg=self.guidance
        for index, step in enumerate(steps):
            capture = AttentionRequest(
                cfg.layers, cfg.affinity_layers,
                need_affinity=(self.transform is not None and self.transform.requires_affinity(index)),
            )
            guided = cfg.strength > 0 and (cfg.max_steps is None or index < cfg.max_steps)
            losses={}; rms=None; labels=(); weights=(); before=(); after=(); maps_cpu=None; extras={}
            if guided:
                if self.objective is None:
                    raise ValueError("Guided sampling requires an attention objective")
                with torch.enable_grad():
                    current = self.backend.latent(state)
                    original = current if differentiable else current.detach()
                    if not original.requires_grad:
                        original = original.requires_grad_(True)
                    self.backend.set_latent(state,original)
                    with self._guidance_attention_context(differentiable, original):
                        pre=self.backend.forward(state,step,attention=capture)
                        batch=pre.attention
                        if batch is None:
                            raise RuntimeError("Backend returned no attention for a guided step")
                        labels=batch.labels
                        if tuple(batch.maps.shape[2:]) != tuple(target.density.shape[2:]):
                            raise ValueError("Backend attention grid differs from prompt target grid")
                        if self.transform is not None:
                            target=self.transform.apply(target,batch,index)
                        w=self.policy.weights(labels,device=batch.maps.device)
                        w=w.float() if differentiable else w.detach().float()
                        if w.shape!=(len(labels),) or (w<0).any() or not torch.isfinite(w).all() or w.sum()<=0:
                            raise ValueError("Head policy produced invalid weights")
                        w=w/w.sum()
                        weighted=aggregate(batch.maps,w)
                        component_total, parts=self.objective.evaluate(weighted,target)
                        scalar=component_total.mean()
                        grad=torch.autograd.grad(
                            scalar, original, create_graph=differentiable, allow_unused=True
                        )[0]
                        if grad is None:
                            raise RuntimeError("Guidance loss has no gradient to model latent; check attention extraction")
                    new_latent,rms=self.updater(original,grad,cfg)
                    if differentiable:
                        if not new_latent.requires_grad or new_latent.grad_fn is None:
                            raise RuntimeError("Updater detached the latent; cannot train head weights")
                    else:
                        new_latent=new_latent.detach()
                    self.backend.set_latent(state,new_latent)
                    losses={k:float(v.mean().item()) for k,v in parts.items()}
                    losses["total"]=float(scalar.item())
                    collect_scores = self.policy.needs_feedback or any(o.wants_scores for o in self.observers)
                    before_scores = (self.scoring(batch.maps.detach(),target).detach()
                                     if collect_scores else None)
                    if before_scores is not None:
                        before = tuple(float(x) for x in before_scores.mean(0).cpu().tolist())
                    if any(o.wants_maps for o in self.observers):
                        maps_cpu=batch.maps.detach().float().cpu()
                    weights=tuple(float(x) for x in w.detach().cpu().tolist())
                    del pre,batch,weighted,component_total,parts,scalar,grad,original
            # The native backend owns CFG/negative predictions and scheduler behavior.
            with torch.set_grad_enabled(differentiable):
                post=self.backend.forward(state,step,attention=capture if guided and self.policy.needs_feedback else None)
                if guided and self.policy.needs_feedback:
                    if post.attention is None or post.attention.labels!=labels:
                        raise RuntimeError("Cannot match post-guidance head identities")
                    after_scores=self.scoring(post.attention.maps,target).detach()
                    self.policy.update(labels,before_scores,after_scores)
                    before=tuple(float(x) for x in before_scores.mean(0).cpu().tolist())
                    after=tuple(float(x) for x in after_scores.mean(0).cpu().tolist())
                    extras["weights_next"]=tuple(float(x) for x in self.policy.weights(labels, device=after_scores.device).cpu().tolist())
                self.backend.advance(state,step,post)
            event=StepEvent(index=index,timestep=step,losses=losses,grad_rms=rms,
                            labels=labels,weights=weights,scores_before=before,
                            scores_after=after,head_maps=maps_cpu,extras=extras)
            for obs in self.observers:
                obs.step(event)
            # Streaming observers own map export; don't retain huge maps in every Result.
            records.append(replace(event, head_maps=None))
        output = self.backend.decode_tensor(state) if differentiable else self.backend.decode(state)
        if differentiable and (not isinstance(output, torch.Tensor) or not output.requires_grad):
            raise RuntimeError("Backend returned a detached/non-Tensor image in differentiable mode")
        result=Result(output=output,events=tuple(records),target=target)
        for obs in self.observers:
            obs.end(result)
        return result
