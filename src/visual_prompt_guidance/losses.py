"""Built-in objectives; custom losses are ordinary functions with the same shape contract."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .contracts import Target


def _spatial_probs(maps: torch.Tensor) -> torch.Tensor:
    p = maps.float().clamp_min(1e-12)
    return p / p.sum((-2, -1), keepdim=True).clamp_min(1e-12)


def coverage(maps: torch.Tensor, target: Target, eps: float = 1e-12) -> torch.Tensor:
    """Original cross-entropy between target density and normalized attention."""
    p = _spatial_probs(maps)
    t = target.density / target.density.sum((-2, -1), keepdim=True)
    return -(t * p.clamp_min(eps).log()).sum((-2, -1))


def focal(maps: torch.Tensor, target: Target, alpha: float = .25, beta: float = 2.) -> torch.Tensor:
    """ScribbleDiff's BCE-with-logits focal formula (on the supplied spatial maps)."""
    y = target.support.expand_as(maps)
    ce = F.binary_cross_entropy_with_logits(maps, y, reduction="none")
    pt = (-ce).exp()
    at = alpha * y + (1 - alpha) * (1 - y)
    return (((1 - pt) ** beta) * at * ce).mean((-2, -1))


def _coords(x: torch.Tensor):
    h, w = x.shape[-2:]
    y, xx = torch.meshgrid((torch.arange(h, device=x.device, dtype=x.dtype) + .5)/h,
                           (torch.arange(w, device=x.device, dtype=x.dtype) + .5)/w,
                           indexing="ij")
    return xx[None,None], y[None,None]


def centroid(maps: torch.Tensor, target: Target) -> torch.Tensor:
    p = _spatial_probs(maps)
    x, y = _coords(p)
    if "centroid" in target.features:
        cx, cy = target.features["centroid"]
        cx = torch.as_tensor(cx, device=p.device, dtype=p.dtype)
        cy = torch.as_tensor(cy, device=p.device, dtype=p.dtype)
    else:
        q = target.density / target.density.sum((-2,-1), keepdim=True)
        cx = (q*x).sum((-2,-1)); cy = (q*y).sum((-2,-1))
    return ((p*x).sum((-2,-1)) - cx).square().add(
            ((p*y).sum((-2,-1)) - cy).square()).add(1e-12).sqrt()


def _moments(p: torch.Tensor):
    x, y = _coords(p)
    mass = p.sum((-2, -1)).clamp_min(1e-12)
    cx = (p*x).sum((-2,-1)) / mass
    cy = (p*y).sum((-2,-1)) / mass
    dx = x - cx[...,None,None]; dy = y - cy[...,None,None]
    mu20 = (dx.square()*p).sum((-2,-1))
    mu02 = (dy.square()*p).sum((-2,-1))
    mu11 = (dx*dy*p).sum((-2,-1))
    angle = 0.5*torch.atan2(2*mu11, mu20-mu02)
    return cx, cy, angle


def moment(maps: torch.Tensor, target: Target, angle_weight: float = 1.) -> torch.Tensor:
    """Centroid + axial-angle matching, adapting ScribbleDiff moment objective."""
    pmin = maps.amin((-2,-1), keepdim=True)
    pmax = maps.amax((-2,-1), keepdim=True)
    norm = (maps - pmin)/(pmax - pmin + 1e-5)
    ax, ay, aa = _moments(norm)
    bx, by, ba = _moments(target.support)
    distance = ((ax-bx).square() + (ay-by).square() + 1e-12).sqrt()
    # Principal axis is undirected: angle and angle + pi represent the same axis.
    diff = torch.atan2(torch.sin(2*(aa-ba)), torch.cos(2*(aa-ba))).abs() / (2*math.pi)
    return distance + angle_weight * diff.square()


def outside(maps: torch.Tensor, target: Target) -> torch.Tensor:
    return (_spatial_probs(maps) * (1 - target.support)).sum((-2,-1))


def centerline(maps: torch.Tensor, target: Target) -> torch.Tensor:
    """Expected distance of attention from the target line/support cells."""
    b, _, h, w = maps.shape
    p = _spatial_probs(maps)
    yy, xx = torch.meshgrid(torch.arange(h, device=p.device, dtype=torch.float32),
                            torch.arange(w, device=p.device, dtype=torch.float32), indexing="ij")
    grid = torch.stack((yy,xx), dim=-1).reshape(1, -1, 2)
    dist_maps=[]
    for k in range(b):
        points = (target.support[k,0] > 0).nonzero().to(p.device, torch.float32)
        if len(points) == 0:
            raise ValueError("Centerline loss requires nonempty target support")
        distance = torch.cdist(grid, points.unsqueeze(0))[0].amin(dim=-1).reshape(h,w)
        dist_maps.append(distance / max(math.hypot(h-1,w-1), 1))
    distances = torch.stack(dist_maps)[:,None]
    return (p*distances).sum((-2,-1))


BUILTIN_LOSSES = {
    "coverage": coverage, "focal": focal, "moment": moment,
    "centerline": centerline, "centroid": centroid, "outside": outside,
}
