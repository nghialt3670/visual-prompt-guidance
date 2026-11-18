# Visual Prompt Guidance 0.2.0

A **small composable framework** for training-free, latent-gradient guidance using spatial attention. A complete rewrite of the public API: the framework is not a FLUX.2 application disguised as a library.

## Design

Core has **five independent contracts**:

| Interface | Supplies | Framework depends on |
|---|---|---|
| `ModelAdapter` | Steps, latent access, attention, prediction, advance, decode | The published protocol, not Diffusers |
| `VisualPrompt` | `Target` projected onto a model's spatial grid | One `project(space)` method |
| `Objective` | Arbitrary weighted differentiable loss functions | Each loss returns `[batch, heads]` |
| `HeadPolicy` | Uniform, manual, adaptive, or custom weights | Opaque head IDs; no assumptions about names |
| `TargetTransform` | Optional attention-based mask expansion or other transforms | The target + attention |

Observers are optional. The **engine has no prompt names, no loss names, no network block names, no CFG, and no inpaint branches**. Those details are supplied by the host application or model adapter. There is deliberately **no global registration system**: just pass objects. A host application can add its own registry or JSON/Pydantic configuration if it needs serializable settings.

**Scope:** The current engine targets models that expose a *differentiable latent state* and *spatial, head-resolved attention during stepwise sampling*. Other generator families can reuse the prompt/loss layers but may require a different engine contract. An abstraction should not pretend GANs and autoregressive models implement denoising steps.

## Install (uv)

```bash
uv sync --extra all
uv run --extra all python examples/gradio_app.py --host 0.0.0.0 --port 7861
```

Without FLUX/UI dependencies:

```bash
uv sync --extra dev
uv run --extra dev pytest -q
```

Build the distributions:

```bash
uv build --no-sources
# If offline, use: uv build --offline --no-build-isolation
```

The package uses a conventional `src/` layout, PEP 621 `pyproject.toml`, optional dependencies, and generates both `.whl` and `.tar.gz`. It has **not** been uploaded to PyPI.

## Python API

```python
from PIL import Image
from visual_prompt_guidance import (
    Engine, Guidance, Request, Point, Loss, Objective, Adaptive, FileObserver, losses,
)
from visual_prompt_guidance.backends import Flux2Klein

engine = Engine(
    backend=Flux2Klein(),
    objective=Objective(
        Loss("coverage", losses.coverage, weight=1.0),
        Loss("centroid", losses.centroid, weight=0.5),
    ),
    guidance=Guidance(strength=0.01, max_steps=10, layers=("D2", "D3")),
    policy=Adaptive(temperature=0.01),
    scoring=Objective(Loss("coverage", losses.coverage)),
    observers=[FileObserver("outputs/example", maps=True, images=True, interval=3)],
)
result = engine.run(
    Request(
        data={"image": Image.open("input.png"), "prompt": "Insert a bird", "object_phrase": "bird"},
        options={"mode": "base", "steps": 30, "max_side": 512, "seed": 42},
    ),
    Point(200, 130),
)
result.output.save("output.png")
```

Swap `Point(...)` for `Box(x0,y0,x1,y1)` or `Scribble(path=[(x,y), ...])`. To use a raster scribble, pass `Scribble(mask=pil_grayscale_mask)`. A path is the primary representation; masks are accepted for compatibility. The prompt's `Target` retains geometry and has **separate spatial density, support, and edit mask**.

The app chooses its own loss combination. Built-in functions: `coverage`, `focal`, `centroid`, `moment`, `centerline`, `outside` (all in `visual_prompt_guidance.losses`). These functions can be combined freely; no engine changes required. `moment` and `focal` follow ScribbleDiff's general formulas, adapted to this framework's spatial attention maps; they are **not** a guaranteed paper-exact reproduction of SD U-Net attention preprocessing.

### Extensibility with zero modifications to the core

**New loss:**

```python
import torch
from visual_prompt_guidance import Loss

def my_loss(attention: torch.Tensor, target, scale=1.0):
    # attention: [B, heads, H, W]; target.density: [B,1,H,W]
    return scale * (attention - target.density).square().mean(dim=(-2, -1))

term = Loss("custom", my_loss, weight=2.0, params={"scale": 0.5})
```

**New visual prompt:** implement one method `project(space: Space) -> Target` in any application class.

**New model:** implement `ModelAdapter` (`space`, `prepare`, `steps`, `latent`, `set_latent`, `forward`, `advance`, `decode`). The backend returns an `AttentionBatch` containing maps `[B,heads,H,W]` and stable opaque head labels. `forward` returns a `Prediction` with the backend's own scheduler prediction. The backend owns all text conditioning, CFG, inpaint, etc.; the engine never handles those concepts.

**New head policy:** implement `reset()`, `weights(labels, device)`, `update(labels, before, after)` and `needs_feedback`. The `before` and `after` matrices are calculated by the **same `Objective` evaluator** used for guidance; no duplicated loss/scoring implementation.

**New propagation:** supply any `TargetTransform`. `AffinityGrow` is optional and works for point, box, scribble, or a custom support mask; it is **not** tied to a scribble type.

## Files

```text
src/visual_prompt_guidance/
  contracts.py           # stable API data/protocol types
  engine.py              # only generic orchestration
  objectives.py          # composable loss terms
  losses.py              # optional built-in differentiable losses
  prompts.py             # optional point/box/path-or-mask scribble
  heads.py               # uniform/manual/adaptive policies
  transforms.py          # optional affinity region growth
  observers.py           # optional CSV, JSON, PNG, NPY exports
  backends/
    flux2.py             # only model-specific sampling/inpaint code
    flux2_attention.py   # only FLUX2-specific Q/K hooks
examples/
  python_api.py          # host app using the public API
  gradio_app.py          # optional UI (not imported by framework)
tests/
  test_framework.py      # mock-backend end-to-end tests
```

No model loader, UI, or implicit preset runs when importing the package.

## Validation and limitations

- Framework tested using a mock backend; tests cover custom losses, gradients, shape contracts, adaptive/manual weights, configurable spatial prompts, transforms, observability, and running without guidance.
- Gradio application construction has been checked.
- **A full 4B FLUX.2 inference run was not available in this environment.** Native attention hooks and inpainting integrations must still be validated against the installed Diffusers release and weights on a GPU. This backend is supplied as an integration example, not certified working for every Diffusers version.
- Head weighting changes the **guidance objective**, not the generator's native attention operation.
- The target edit mask is prepared once. Subsequent target transforms modify guidance targets; they do not change the backend's native inpaint mask mid-run.
- Head map exports can consume considerable space and memory; set `maps=False` when not needed.
