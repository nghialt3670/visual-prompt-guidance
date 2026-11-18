"""Public Python API for composable spatial-attention guidance."""
from .contracts import (Request, Space, Target, AttentionBatch, AttentionRequest,
                        Prediction, StepEvent, Result, ModelAdapter, VisualPrompt)
from .engine import Engine, Guidance, rms_update
from .heads import Uniform, Manual, Adaptive
from .objectives import Loss, Objective
from .prompts import Point, Box, Scribble
from .transforms import AffinityGrow
from .observers import FileObserver
from . import losses

__all__ = [
    "Request","Space","Target","AttentionBatch","AttentionRequest","Prediction",
    "StepEvent","Result","ModelAdapter","VisualPrompt","Engine","Guidance",
    "rms_update","Uniform","Manual","Adaptive","Loss","Objective","Point",
    "Box","Scribble","AffinityGrow","FileObserver","losses",
]
