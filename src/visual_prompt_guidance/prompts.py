"""Standard prompt implementations; users can supply any VisualPrompt Protocol."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from .contracts import Space, Target


def _grid(space: Space):
    width, height = space.source_size
    gh, gw = space.grid_size
    if min(width, height, gh, gw) < 1:
        raise ValueError("Space dimensions must be positive")
    y, x = torch.meshgrid((torch.arange(gh).float()+.5)*height/gh,
                          (torch.arange(gw).float()+.5)*width/gw, indexing="ij")
    return x, y


def _target(heat: torch.Tensor, support: torch.Tensor, *, margin: int=0, features=None) -> Target:
    if heat.ndim != 2 or heat.sum() <= 0:
        raise ValueError("Spatial prompt has no pixels")
    density = (heat.float()/heat.sum())[None,None]
    support = (support > 0).float()[None,None]
    if margin < 0:
        raise ValueError("Edit margin must be >= 0")
    region = F.max_pool2d(support, 2*margin+1, 1, margin) if margin else support.clone()
    return Target(density, support, region, features or {})


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    sigma: float = 12.0          # image pixels
    edit_radius: float = 48.0   # image pixels

    def project(self, space: Space) -> Target:
        w,h = space.source_size
        if not (0 <= self.x < w and 0 <= self.y < h) or self.sigma <= 0 or self.edit_radius <= 0:
            raise ValueError("Invalid point coordinates or radius")
        x,y = _grid(space)
        d2=(x-self.x).square()+(y-self.y).square()
        heat = torch.exp(-d2/(2*self.sigma**2))
        if heat.max() <= 1e-10:
            heat.reshape(-1)[d2.argmin()] = 1
        support = (d2 <= self.edit_radius**2).float()
        if not support.any():
            support.reshape(-1)[d2.argmin()] = 1
        return _target(heat, support, features={"centroid":(self.x/w,self.y/h)})


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    margin: float = 0.0  # edit-mask padding in image pixels, not target padding

    def project(self, space: Space) -> Target:
        w,h=space.source_size
        x0,x1=sorted((self.x0,self.x1)); y0,y1=sorted((self.y0,self.y1))
        if x1<=x0 or y1<=y0 or x0<0 or y0<0 or x1>w or y1>h:
            raise ValueError("Invalid box")
        x,y = _grid(space)
        inside = ((x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)).float()
        if not inside.any():
            dist=(x-(x0+x1)/2).square()+(y-(y0+y1)/2).square()
            inside.reshape(-1)[dist.argmin()]=1
        ex = ((x>=max(0,x0-self.margin))&(x<=min(w,x1+self.margin))&
              (y>=max(0,y0-self.margin))&(y<=min(h,y1+self.margin))).float()
        target = _target(inside, inside, features={"centroid":((x0+x1)/(2*w),(y0+y1)/(2*h)),"box":(x0,y0,x1,y1)})
        return Target(target.density, target.support, torch.maximum(target.edit_mask,ex[None,None]),target.features)


@dataclass(frozen=True)
class Scribble:
    """Accept original ordered points (preferred) or an existing raster mask."""
    path: Sequence[tuple[float,float]] | None = None
    mask: Image.Image | None = None
    width: float = 6.0
    edit_margin: float = 24.0

    def project(self, space: Space) -> Target:
        w,h=space.source_size
        if self.path is None and self.mask is None:
            raise ValueError("Provide either a path or mask")
        if self.path is not None:
            if not self.path or self.width <= 0:
                raise ValueError("Scribble path must contain points and positive width")
            image=Image.new("L",(w,h),0)
            drawing=ImageDraw.Draw(image)
            pts=[(float(x),float(y)) for x,y in self.path]
            if any(x<0 or x>=w or y<0 or y>=h for x,y in pts):
                raise ValueError("Scribble points must lie inside the image")
            pixel_width=max(1,round(self.width))
            if len(pts)>1:
                drawing.line(pts,fill=255,width=pixel_width,joint="curve")
            for x,y in pts:
                r=max(.5,pixel_width/2)
                drawing.ellipse((x-r,y-r,x+r,y+r),fill=255)
        else:
            image=self.mask.convert("L")
            if image.size!=(w,h):
                raise ValueError("Scribble mask must match source image dimensions")
        mask=torch.from_numpy(np.array(image, copy=True)).float()[None,None]/255.
        gh,gw=space.grid_size
        pooled=F.adaptive_max_pool2d(mask,(gh,gw))[0,0]
        if pooled.sum() <= 0:
            raise ValueError("Scribble disappears at the target grid resolution")
        token_margin=round(self.edit_margin*max(gw/w,gh/h))
        return _target(pooled, pooled, margin=max(0,token_margin),
                       features={"path": tuple(self.path) if self.path is not None else None})
