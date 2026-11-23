"""Optional target transformations, separate from the engine and prompt types."""
from __future__ import annotations
from dataclasses import replace
import torch
from .contracts import Target, AttentionBatch


class AffinityGrow:
    """Grow any spatial target using the adapter's image-token affinity matrix.

    No scribble-specific checks. The policy chooses when and how to grow.
    """
    def __init__(self, *, start: int=0, stop: int=10, limit: int=12,
                 threshold: float=.2, max_fraction: float=.3, grow_edit_mask: bool=False):
        if start < 0 or stop < start or limit < 1 or threshold <= 0 or not 0 < max_fraction <= 1:
            raise ValueError("Invalid propagation settings")
        self.start, self.stop = start, stop
        self.limit, self.threshold = limit, threshold
        self.max_fraction = max_fraction
        self.grow_edit_mask = grow_edit_mask
        self._original = None
        self._active = None

    def requires_affinity(self, step_index: int) -> bool:
        return self.start <= step_index < self.stop

    def reset(self, target: Target) -> None:
        self._original = (target.support[:,0]>0).detach().clone()
        self._active = self._original.clone()

    @torch.no_grad()
    def apply(self, target: Target, attention: AttentionBatch, step_index: int) -> Target:
        if not self.start <= step_index < self.stop:
            return target
        if attention.affinity is None:
            raise ValueError("Target transform requires image-to-image affinity from backend")
        if self._active is None:
            self.reset(target)
        orig = self._original.to(attention.maps.device)
        active = self._active.to(attention.maps.device)
        b,h,w=active.shape
        n=h*w
        a=attention.affinity.detach().float().clamp_min(1e-12)
        a=a/a.sum(-1,keepdim=True)
        for k in range(b):
            occupancy=int(active[k].sum())
            room=int(n*self.max_fraction)-occupancy
            if room <= 0:
                continue
            neighbors=torch.zeros_like(active[k])
            neighbors[1:,:] |= active[k][:-1,:]
            neighbors[:-1,:] |= active[k][1:,:]
            neighbors[:,1:] |= active[k][:,:-1]
            neighbors[:,:-1] |= active[k][:,1:]
            candidates=(neighbors & ~active[k]).flatten().nonzero().flatten()
            if not len(candidates):
                continue
            ref=a[k,orig[k].flatten()].mean(dim=0).clamp_min(1e-12)
            ref=ref/ref.sum()
            q=a[k].index_select(0,candidates).clamp_min(1e-12)
            kl=.5 * ((q*(q.log()-ref.log())).sum(-1)+
                       (ref*(ref.log()-q.log())).sum(-1))
            top=min(self.limit,room,len(candidates))
            values,indices=torch.topk(kl,top,largest=False)
            active[k].flatten()[candidates[indices[values < self.threshold]]]=True
        self._active=active
        support=active[:,None].float()
        density=support/support.sum((-1,-2),keepdim=True).clamp_min(1.)
        edit=torch.maximum(target.edit_mask, support) if self.grow_edit_mask else target.edit_mask
        return replace(target, density=density, support=support, edit_mask=edit)
