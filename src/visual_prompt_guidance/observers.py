"""Optional file diagnostics. Pure side-effect layer; core never writes files."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .contracts import StepEvent, Result


class FileObserver:
    """Write per-step and per-head statistics, with optional maps/PNG heatmaps."""
    def __init__(self, directory: str | Path, *, maps: bool=False, images: bool=False, interval: int=1):
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.directory=Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.wants_maps=bool(maps or images)
        self.wants_scores=True
        self.export_maps=bool(maps)
        self.export_images=bool(images)
        self.interval=interval
        self._steps=[]; self._head_rows=[]

    def step(self, event: StepEvent) -> None:
        self._steps.append({"step":event.index,"timestep":str(event.timestep),
                            "grad_rms":event.grad_rms,"losses":dict(event.losses),
                            "extra":dict(event.extras)})
        for i, label in enumerate(event.labels):
            map_i = event.head_maps[:,i] if event.head_maps is not None else None
            self._head_rows.append({
                "step":event.index,"head":label,"weight":event.weights[i],
                "before":event.scores_before[i] if event.scores_before else "",
                "after":event.scores_after[i] if event.scores_after else "",
                "mean_attention":float(map_i.mean()) if map_i is not None else "",
                "max_attention":float(map_i.max()) if map_i is not None else "",
            })
            if map_i is None or event.index % self.interval != 0:
                continue
            folder=self.directory/"head_maps"/f"step_{event.index:03d}"
            folder.mkdir(parents=True,exist_ok=True)
            safe=f"head_{i:03d}"
            data=map_i.numpy()
            if self.export_maps:
                np.save(folder/f"{safe}.npy",data)
            if self.export_images:
                from matplotlib import colormaps
                view=data[0]
                lo,hi=float(view.min()),float(view.max())
                normalized=(view-lo)/max(hi-lo,1e-9)
                color=(colormaps["inferno"](normalized)[...,:3]*255).astype(np.uint8)
                Image.fromarray(color).save(folder/f"{safe}.png")
        
    def end(self, result: Result) -> None:
        (self.directory/"steps.json").write_text(json.dumps(self._steps,indent=2,default=str))
        if self._head_rows:
            with (self.directory/"heads.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=self._head_rows[0].keys())
                writer.writeheader(); writer.writerows(self._head_rows)
        (self.directory/"head_labels.json").write_text(json.dumps(list(dict.fromkeys(r["head"] for r in self._head_rows)),indent=2))
