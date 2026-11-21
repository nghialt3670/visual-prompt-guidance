# Visual Prompt Guidance 0.3.0

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

## InstructPix2Pix backend (0.3.0)

`InstructPix2Pix` is now a second implementation of `ModelAdapter`, using the
lightweight Stable Diffusion U-Net rather than FLUX.2. It uses the public
`Engine`, loss functions, head policies, prompts, observers and propagation
without changes to any core interfaces.

Install only the InstructPix2Pix + UI extras:

```bash
uv sync --extra pix2pix --extra ui
uv run --extra pix2pix --extra ui python examples/gradio_app.py --host 0.0.0.0 --port 7861
```

Select **InstructPix2Pix** in the `Model backend` dropdown. Upload your image,
mark a point/box/scribble, enter an editing *instruction* (e.g. "Make the bird
blue") and set **Target object phrase** to a substring of that instruction
(`bird`). The source image is supplied as the model's image conditioning; the
visual prompt **only guides attention** and is not composited onto the input.

A runnable Python example:

```bash
uv run --extra pix2pix python examples/instruct_pix2pix.py input.png \
  --prompt 'Make the bird blue' --object bird --point 128,100 \
  --max-side 256 --steps 20 --guide-steps 8
```

Public API:

```python
from visual_prompt_guidance.backends import InstructPix2Pix
from visual_prompt_guidance import Engine, Guidance, Request, Point, Loss, Objective, losses

backend = InstructPix2Pix(device="auto", dtype="auto")
engine = Engine(backend, Objective(Loss("coverage", losses.coverage)),
                guidance=Guidance(strength=0.01, max_steps=8))
result = engine.run(
    Request({"image": input_image, "prompt": "Make the bird blue", "object_phrase": "bird"},
            options={"steps": 20, "max_side": 256, "seed": 42,
                     "cfg": 7.5, "image_cfg": 1.5}),
    Point(128, 100, sigma=16),
)
result.output.save("result.png")
```

- `cfg`: InstructPix2Pix text guidance scale; default 7.5.
- `image_cfg`: InstructPix2Pix source-image guidance scale; default 1.5.
- `max_side`: Longest input side at inference. The U-Net implementation rounds
  dimensions to multiples of 32, then resizes the output to the input size.
- `Guidance(layers=...)`: Leave unset for the adapter's small automatic U-Net
  selection, or choose full module names discovered via
  `backend.attention_layers()`. Manual head IDs have the form
  `down_blocks.1.attentions.0.transformer_blocks.0.attn2:H00`.
- `AffineGrow` works using U-Net self-attention from sibling `attn1` modules.
  If a selected affinity layer's spatial grid does not match the target grid,
  choose another U-Net layer or disable the transform.
- InstructPix2Pix does **not** perform native masked inpainting. It is
  instruction-based image-to-image editing, with separate text and image CFG.
  The FLUX.2 `base` and `inpaint` options are not applicable.
- Head selection/weighting changes the **guidance loss**, not native U-Net
  inference attention.
- Inference with gradients uses more memory than ordinary InstructPix2Pix.
  Try 256px and 1-2 heads initially on a modest GPU or M-series Mac.

Validation: 7 new offline adapter/hook integration tests and 9 existing
framework tests pass. Full `timbrooks/instruct-pix2pix` checkpoint inference
has not been executed in the build environment; exact runtime compatibility
and image quality still require GPU/MPS validation.

## Train attention-head weights from edited-image examples (v0.4)

**Learned** is a fourth head policy alongside Uniform, Manual and Adaptive. It
optimizes *only the per-head softmax logits* against reference edited images.
The generator's weights remain frozen. The training path uses higher-order PyTorch
gradients through the attention loss, latent guidance update, denoising scheduler,
and VAE decoder. This is **image-level supervision**, not attention-mask fitting.

### Dataset

Provide a folder containing `manifest.json` and the referenced image files, or a
ZIP whose root contains them. Input and reference edited images **must be the
same dimensions**. Coordinates in the visual prompt refer to the input image.

```text
my_training_data/
  manifest.json
  input_001.png
  edited_001.png
  edit_mask_001.png   # optional, white=changed region
```

`manifest.json` is a JSON array (or `{"samples": [...]}`):

```json
[
  {
    "input_image": "input_001.png",
    "target_image": "edited_001.png",
    "edit_mask": "edit_mask_001.png",
    "prompt": "Add a red dog following the curved line",
    "object_phrase": "dog",
    "visual": {
      "type": "scribble",
      "path": [[20, 30], [48, 35], [76, 50]],
      "width": 6
    }
  }
]
```

Valid visual annotations:
- `{"type":"point", "x":40, "y":50, "sigma":12, "edit_radius":48}`
- `{"type":"box", "x0":20, "y0":30, "x1":100, "y1":150}`
- `{"type":"scribble", "path":[[x,y], ...], "width":6}`
- `{"type":"scribble", "mask":"scribble_mask.png"}`

The `edit_mask` is optional; when omitted, the visual prompt's projected edit
region is used as a rough training mask. For high-quality supervision, provide a
true changed-region mask. Global LPIPS (if enabled) is calculated **on the entire
output image**, not on a cropped or masked image; preservation is controlled by
the outside-region pixel loss.

### UI

Launch `examples/gradio_app.py` and choose **Learned** under **Attention heads**.
Optionally select layers and individual heads using checkboxes. Under the
**Trainable head weights** panel, upload the ZIP or supply a dataset directory,
select an outer image loss (`l1`, `mse`, `charbonnier`, or `smooth_l1`), set the
inside/outside weights, epochs, optimizer, learning rate, gradient clipping, and
checkpoint path. Click **Train head weights**. The generated JSON checkpoint is
used by the **Generate** button while Learned mode is selected. Adaptive and
Manual remain available and unchanged.

### Command line

```bash
uv sync --extra pix2pix --extra ui --extra dev

uv run --extra pix2pix python examples/train_heads.py \
  --backend pix2pix \
  --dataset ./my_training_data \
  --steps 4 --guided-steps 1 --max-side 128 \
  --epochs 2 --lr 0.001 --loss l1 \
  --inside-weight 1 --outside-weight 1 \
  --output outputs/learned_heads.json
```

For optional LPIPS: `uv sync --extra pix2pix --extra perceptual`; then set
`--lpips-weight 0.1` (may download LPIPS feature-network weights).

### Python API

```python
from visual_prompt_guidance import (
    Engine, Guidance, Objective, Loss, Learned, HeadTrainer,
    TrainConfig, ImageLossConfig, losses,
)
from visual_prompt_guidance.backends import InstructPix2Pix

policy = Learned(selected=None, temperature=1.0)
engine = Engine(
    backend=InstructPix2Pix(),
    objective=Objective(Loss("coverage", losses.coverage)),
    guidance=Guidance(strength=0.01, max_steps=1),
    policy=policy,
)
trainer = HeadTrainer(
    engine,
    config=TrainConfig(epochs=2, learning_rate=1e-3,
                       checkpoint="outputs/learned_heads.json"),
    image_loss=ImageLossConfig(pixel="l1", inside_weight=1.0,
                               outside_weight=1.0),
)
trainer.fit("my_training_data", options={"steps":4, "max_side":128, "seed":42})

# Later, for normal low-memory inference:
trained_policy = Learned.load("outputs/learned_heads.json")
```

**Memory:** This is full unrolled, second-order training, not a cheap
attention-only proxy. Every denoising step remains differentiable. PyTorch's
math SDPA is forced on CUDA because fused Flash/memory-efficient attention may
not support double backward. It can use **significantly more VRAM** than normal
sampling. Start with 4 steps, 1 guided step and 128px resolution, and train
one image per optimizer step. Large FLUX models may still be impractical.
No real-checkpoint/GPU training run is claimed: the included CPU tests use a
differentiable mock generator and validate higher-order gradients numerically.

## Persistent Gradio settings and experiment runs (v0.5.0)

The optional Gradio app saves **two independent UI configuration snapshots** in
`configs/ui_settings.json`: one for **Generate** and one for **Training**.
Change controls and click **Save Generate settings** or **Save Training settings**;
or simply click **Generate** / **Train head weights**, which saves that workflow's
settings first. On restart, the last-saved workflow's controls are restored.
Use **Load Generate settings** and **Load Training settings** to switch presets.
Model weights are *not* loaded just to restore settings. The active model's
previously selected layer/head IDs are retained; use **Load available layers / heads**
when you want to browse the full catalog (this still loads the model checkpoint).
Image uploads, canvas drawings, and temporary uploaded dataset ZIP files are not
stored in the preferences file. Use a persistent `dataset_path` to re-run training
later. Settings are stored on the server and shared by its UI users, so don't
use this single-user preference file as multi-user account storage.

Run outputs are kept separate and never intentionally overwrite another run:

```text
outputs/
  generate/YYYYMMDD_HHMMSS_microseconds/
    config.json          # effective generation parameters
    input.png
    visual_prompt.png    # only if using a scribble
    result.png
    steps.json           # observer output
    heads.csv            # observer output (when heads were recorded)
    head_maps/           # only when map export is enabled
  train/YYYYMMDD_HHMMSS_microseconds/
    config.json          # model, dataset, objectives and training settings
    learned_heads.json  # trained weights
    history.json        # per-sample losses, gradient norm, components
    head_weights.csv    # weights sorted descending
    loss_curve.png      # training image loss by iteration
```

All paths are relative to the working directory from which the app starts.
Specify `VPG_UI_CONFIG=/path/to/settings.json` to move the saved UI settings.

CLI training also creates a timestamped run by default:

```bash
uv run --extra pix2pix python examples/train_heads.py \
  --backend pix2pix --dataset ./my_training_data \
  --max-side 128 --steps 4 --guided-steps 1 --epochs 1
```

To use a particular location, specify `--output-dir outputs/train/my_experiment`.
The legacy `--output checkpoint.json` option is retained for existing scripts;
it writes the other training artifacts next to the specified checkpoint.
