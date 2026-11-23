# Visual Prompt Guidance 0.9.0

A **small composable framework** for training-free, latent-gradient guidance using spatial attention. A complete rewrite of the public API: the framework is not a FLUX.2 application disguised as a library.

## Region-aware image losses and evaluation (v0.9.0)

The training objective keeps **independent, pluggable edit/background losses**.
When Edit loss is `lpips`, LPIPS runs with `spatial=True` on the **full
unmasked** generated and reference images. Its per-pixel perceptual difference
map is then averaged **inside** the ground-truth `edit_mask` (or projected
visual edit region if missing). This prevents the large identical masked-out
background from diluting edit LPIPS. LPIPS still sees context through its
receptive field; its values are not directly comparable with older run losses.
Background L1 remains the average RGB pixel error outside the edit mask.

The Training UI overlays completed **epoch-average loss** and one curve per
training image. The new `epoch_losses.csv` and `sample_losses.csv` are saved
with `loss_curve.png` and existing reports. Epoch averages are not plotted
until all samples in that epoch have completed.

After training, compare **learned vs uniform weights** with paired seeds,
identical prompts, guidance configuration and selected heads:

```bash
uv run --extra flux2 --extra perceptual python examples/compare_heads.py \
  --checkpoint outputs/train/YOUR_RUN/learned_heads.json \
  --dataset data/compbench-vpg/add/manifest_scribble.json \
  --max-samples 5
```

The script uses the training run's saved `config.json` if present and stores
paired images, `paired_losses.csv`, and `summary.json` under `outputs/compare/`.
Negative `mean_delta_learned_minus_uniform` means learned weights obtained
lower image loss **on those samples**. Use a held-out set for generalization.
This comparison is optional; it does not run inside the training loop.

## Inference modes and canvas cleanup (v0.8.7)

The Inference UI has two saved modes in `generate.inference_mode`:

- **Standard (no guidance):** plain image upload + instruction. Runs the
  backend's ordinary denoising/CFG/scheduler without spatial objectives, latent
  guidance, propagation, or learned-head checkpoint. Turn on **Save attention
  maps** to capture token attention at the configured export interval; otherwise
  attention capture remains disabled. No scribble, point, or box is required.
  FLUX native inpaint mode uses a full-image edit mask when no spatial prompt is
  supplied.
- **Visual guidance:** draw a scribble, point, or box and configure attention
  guidance normally. A single **Clear annotations** button resets *all*
  visual prompt types without deleting the input source image.

Switching modes copies the source image to the appropriate input component;
any existing annotation is not passed to Standard mode. Both input and output
panels use matching 500px-height image components in a dedicated image row,
with visual prompt controls placed underneath so they cannot stretch output.
The mode is saved in `configs/project.json`, and training configuration is
unaffected.

## Training UI responsiveness (v0.8.3)

The Training UI acknowledges the click immediately (`queue=False`), then streams
stage and per-denoising-step statuses while it loads the model, processes each
sample and runs higher-order backward. Live metric plots still update only
**after each optimizer step**, when the image loss and learned head gradients
become available. A heartbeat keeps the UI alive during long steps; the same
messages print to the terminal and are saved in
`outputs/train/<timestamp>/training.log`.

If you get no `[VPG training]` message after clicking **Train**, the request
has not reached the Gradio backend (check browser/server connectivity). If
it prints `preparing model and denoising` for a long time, the backend is
inside its first heavyweight forward pass rather than waiting on the UI.

## Dedicated inference and training applications (v0.8.0)

The monolithic Gradio app was split into two independent entry points:

```bash
# Inference/editor, with learned-checkpoint loading
uv run --extra all python examples/ui_inference.py --host 0.0.0.0 --port 7861

# Training, with per-sample live loss, gradient norm and head-weight plots
uv run --extra all python examples/ui_training.py --host 0.0.0.0 --port 7862

# Browse saved inference and training images, maps, and reports
uv run --extra all python examples/ui_visualize.py --host 0.0.0.0 --port 7863
```

`examples/gradio_app.py` is a backwards-compatible alias for **Inference only**.
Both applications share `configs/project.json` (`--config` or `VPG_CONFIG`) and the
same application-level widgets in `examples/ui_common.py`, but maintain **separate
Generate and Train profiles**. Normal CLI commands use the exact same profiles.
The read-only Run Explorer browses `outputs/generate` and `outputs/train`,
including a zoomable attention-map grid (steps × flattened layer/head),
per-step metrics, training loss reports, and learned head weights. Filter the
grid to selected layers or show all layers.

Both pages **automatically restore their own saved profile on startup** and
populate a compact, scrollable **head table with every layer and head**. Model
switching reloads the table using lightweight architecture metadata. The
**Target layers** shortcut selects every head in chosen layers; the table's
`Use` column still supports individual head selection. The default is all heads
for Uniform, Adaptive, or Learned (unless a checkpoint supplies a specific
learned subset). The checked rows produce the engine's `selected_heads` and
`selected_layers` settings.

The controls are organized into **Backend** (including model-specific CFG/
inpainting), **Sampling**, **Guidance** (steps, losses, policy/checkpoints),
**Propagation**, and Inference-only **Diagnostics**. The project config remains
flat for CLI compatibility and is still shared by both applications. Each
workflow keeps its own `generate`/`train` profile. There is no Restore button;
use Save to persist changes. A full run also saves its effective settings.

In **Inference**, select **Learned** under Head policy, choose a checkpoint from
`outputs/train/*/learned_heads.json`, and click **Use checkpoint**. The table
continues showing all backend heads, while only the checkpoint's trained heads
are checked. In
**Training**, select a dataset, optimizer, image loss and heads and click **Train
head weights**. Live charts update *after every training sample*, not after
every denoising step. A completed run creates `history.json`, `loss_curve.png`,
`head_weights.csv` and `learned_heads.json` in `outputs/train/<timestamp>/`.

**Attention catalog fix:** FLUX.2's former UI code used `_load('base')` as if it
returned a pipeline; it actually returned `None`. The new attention selector
calls `backend.attention_catalog()`, which reads **only** the component's small
`config.json` rather than loading model weights. FLUX head counts are derived
from its transformer config; InstructPix2Pix instantiates a weightless U-Net
on the PyTorch `meta` device from the UNet config to discover exact module
names. This may fetch a tiny JSON config once, not a checkpoint.

The inference/training CLI scripts remain:

```bash
uv run --extra flux2 python examples/train_heads.py --backend flux2 \
  --dataset data/compbench-vpg/add/manifest_scribble.json

uv run --extra flux2 python examples/generate.py --backend flux2 \
  --dataset data/compbench-vpg/add/manifest_scribble.json \
  --checkpoint outputs/train/YOUR_RUN/learned_heads.json
```

## Shared CLI / Gradio configuration (v0.6.0)

**One project settings file:** `configs/project.json` (or pass `--config` / set `VPG_CONFIG`).
All default values live in `src/visual_prompt_guidance/config.py`. CLI, Gradio and Python hosts use the same `resolve_settings` and `runtime.build_engine` functions. The project file contains **two independent profiles**, `generate` and `train`; omitted values use the central defaults. Example: `configs/project.example.json`.

```bash
# Start UI with the same settings file that scripts read
uv run --extra ui --extra pix2pix python examples/ui_inference.py --config configs/project.json

# Train using train profile; explicit arguments override it for this run only
uv run --extra pix2pix python examples/train_heads.py \
  --dataset data/compbench-vpg/add/manifest_scribble.json \
  --backend pix2pix --steps 20 --guided-steps 3 --epochs 5

# Persist CLI overrides to train profile for subsequent UI/CLI runs
uv run --extra pix2pix python examples/train_heads.py \
  --dataset data/compbench-vpg/add/manifest_scribble.json --steps 20 --save-config

# Generate with saved Generate profile, selecting one dataset example
uv run --extra pix2pix python examples/generate.py \
  --dataset data/compbench-vpg/add/manifest_scribble.json --sample-index 0

# Inference with trained head weights; attention layers are auto-restored
uv run --extra pix2pix python examples/generate.py \
  --dataset data/compbench-vpg/add/manifest_scribble.json \
  --checkpoint outputs/train/YOUR_RUN/learned_heads.json
```

With a standalone source image, use `examples/generate.py --image input.png --prompt "Add a dog" --object dog --point 100 150`, or `--box x0 y0 x1 y1`, `--scribble-mask mask.png`, or `--scribble-path path.json`. The legacy `examples/instruct_pix2pix.py` is now a wrapper around the same Generate CLI.

Each dedicated UI automatically restores its own saved profile on startup. Inference offers **Use checkpoint** for trained weights. Saved weights are referenced by `learned_checkpoint`; inference loads them with `Learned.load`, restoring the *exact trained attention layer order* and rejecting incompatible explicit selections. To avoid confusing comparisons, match the guidance loss, strengths, and denoising schedule of training when applying weights at inference.

The app will read existing `configs/ui_settings.json` from v0.5.x if the canonical file does not exist, then write `configs/project.json` at the next save. Uploaded images and ZIP bytes are not persisted as profile settings. Run folders still contain the effective `config.json` snapshot.

### CompBench datasets without manual filtering

```bash
# Using already downloaded CompBench data: no network, no re-download
uv run --extra dataset python examples/prepare_compbench.py \
  --split-only --out data/compbench-vpg

# Or download a new random sample and write task/prompt manifests automatically
uv run --extra dataset python examples/prepare_compbench.py \
  --per-task 5 --out data/compbench-vpg --seed 42
```

A task folder (e.g. `data/compbench-vpg/add/`) contains `manifest.json` with all variants and `manifest_point.json`, `manifest_box.json`, and `manifest_scribble.json` with a single visual-prompt type. The root folder also contains these combined-across-task manifests. **Pass the specific manifest path** to training when learning task-specific scribble weights; don't train on the mixed manifest unless intended. Input/edited images, masks, instructions and centerline paths keep the existing `EditingDataset` format. CompBench edit-region masks are not guaranteed clean object silhouettes.

## Design

Core has **five independent contracts**:

| Interface | Supplies | Framework depends on |
|---|---|---|
| `ModelAdapter` | Steps, latent access, attention, prediction, advance, decode | The published protocol, not Diffusers |
| `VisualPrompt` | `Target` projected onto a model's spatial grid | One `project(space)` method |
| `Objective` | Arbitrary weighted differentiable loss functions | Each loss returns `[batch, heads]` |
| `HeadPolicy` | Uniform, manual, adaptive, or custom weights | Opaque head IDs; no assumptions about names |
| `TargetTransform` | Optional attention-based mask expansion or other transforms | The target + attention |

Observers are optional. The **engine has no prompt names, no loss names, no network block names, no CFG, and no inpaint branches**. Those details are supplied by the host application or model adapter. There is deliberately **no global registration system**: just pass objects. The optional `visual_prompt_guidance.config` and `visual_prompt_guidance.runtime` modules supply shared, versioned JSON configuration and composition for the CLI and Gradio application.

**Scope:** The current engine targets models that expose a *differentiable latent state* and *spatial, head-resolved attention during stepwise sampling*. Other generator families can reuse the prompt/loss layers but may require a different engine contract. An abstraction should not pretend GANs and autoregressive models implement denoising steps.

## Install (uv)

```bash
uv sync --extra all
uv run --extra all python examples/ui_inference.py --host 0.0.0.0 --port 7861
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
  ui_common.py           # shared widget/metadata support
  ui_inference.py        # inference-only Gradio app
  ui_training.py         # training-only Gradio app with live plots
  ui_visualize.py        # saved inference and training run explorer
  gradio_app.py          # compatibility launcher for inference
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
uv run --extra pix2pix --extra ui python examples/ui_inference.py --host 0.0.0.0 --port 7861
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

Launch `examples/ui_training.py` to train weights. Then launch `examples/ui_inference.py`, choose **Learned** under Head policy, and load the saved checkpoint.
Check heads directly in the compact table; no separate layer selector is needed.
In the Training UI, supply a dataset ZIP/manifest and configure the optimizer,
image loss, epochs and guidance. Click **Train head weights**. In Inference,
choose the resulting JSON checkpoint under **Guidance > Attention policy** and
click **Use checkpoint**; its learned weights are then used for inference.

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

## Persistent configuration and experiment runs

The canonical shared file is `configs/project.json`. Settings from both the
Inference and Training apps are populated on launch; Save or Run persists the
edited workflow profile. Model metadata is fetched automatically, without
loading model weights. The `generate` and `train` profiles remain independent;
CLI scripts use those exact same profiles. `VPG_CONFIG` or `--config` can select
a different settings file. v0.5.x `configs/ui_settings.json` is read as a
migration source if the new file does not exist.

All heads of the selected backend appear in a single compact table. Checking
or unchecking rows determines the active guidance layers and head subset.
Training checkpoints restore their captured layer IDs and selected head weights.
If a checkpoint is incompatible with the selected model, the UI rejects it.

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
Specify `VPG_CONFIG=/path/to/project.json` to choose the shared settings file.

CLI training also creates a timestamped run by default:

```bash
uv run --extra pix2pix python examples/train_heads.py \
  --backend pix2pix --dataset ./my_training_data \
  --max-side 128 --steps 4 --guided-steps 1 --epochs 1
```

To use a particular location, specify `--output-dir outputs/train/my_experiment`.
The legacy `--output checkpoint.json` option is retained for existing scripts;
it writes the other training artifacts next to the specified checkpoint.


## Small CompBench local-edit dataset sampler (v0.5.1)

`examples/prepare_compbench.py` randomly samples source/ground-truth image pairs from
[BohanJia/CompBench](https://huggingface.co/datasets/BohanJia/CompBench)
(CVPR 2026, dataset license CC BY 4.0) for **add**, **remove**, and **replace**.
It uses Hugging Face's **dataset-viewer `/filter` API** to request individual
rows and their images, instead of downloading the approximately 3.8 GB full
Parquet dataset. Internet access and a working dataset viewer are required.

```bash
uv run --extra dataset python examples/prepare_compbench.py \
  --per-task 5 --seed 42 --out data/compbench-vpg
```

For each selected pair, the script exports **three training entries** (same
input/output/mask, but point, box, or scribble target). The annotation rules:

- Point = object-mask centroid snapped to a foreground pixel.
- Box = tight bounding rectangle of the largest mask component.
- Scribble = skeletonize the largest component using scikit-image, extract
  an approximate longest centerline using weighted graph search, and resample
  evenly spaced points along its length. Stored as an ordered **path**.
- Edit mask = the original dataset binary edit-region mask for image loss.

```text
data/compbench-vpg/
  manifest.json                    # all three tasks combined; valid trainer input
  download_report.json             # source IDs and skip reasons
  add/
    manifest.json                  # add only; valid trainer input
    images/*_input.png, *_target.png
    masks/*_edit.png, *_region.png
    previews/*.png                 # verify point, box, scribble visually
  remove/
    manifest.json
    images/ ...
    masks/ ...
    previews/ ...
  replace/
    manifest.json
    images/ ...
    masks/ ...
    previews/ ...
```

Run training by task:

```bash
uv run --extra pix2pix python examples/train_heads.py \
  --backend pix2pix --dataset data/compbench-vpg/add \
  --steps 4 --guided-steps 1 --max-side 128 --epochs 2
```

**Check the generated `manifest.json` and preview PNGs before training.**
CompBench's mask is documented as the *edit region*, which may not be a clean
object silhouette in every example; generated centerlines are approximate.
The `object_phrase` is heuristically extracted from the original instruction
and may need manual correction. Rows for which no phrase is detected are skipped
and reported. In removal tasks the phrase refers to the object being removed;
in replacement tasks the script tries to select the new object after "with".
Treat this as dataset preparation, not validated supervision quality. Multiple
visual variants of one source/target pair must stay in the same train/validation
split to prevent leakage.


### v0.8.4: training stability and attention-table startup

- RMS-normalized guidance clamps the *squared* RMS before taking its square root.
  This prevents NaN higher-order derivatives when the inner gradient is zero,
  without detaching the normalization term.
- Save/Train before asynchronous head metadata has loaded selects all heads of
  the selected backend, including all layers. Explicitly clearing every row
  still raises an error. Only lightweight model metadata is read.
- A non-finite learned-head gradient error now includes NaN/Inf counts and
  suggests reducing denoising/guided steps for numerical troubleshooting.

Start with low denoising/guided step counts when debugging differentiable
backpropagation on MPS. The full 30-step GPU workload is not validated here.


### Diagnosing NaN head-weight gradients (v0.8.5)

Higher-order training differentiates through a guidance gradient and therefore
needs stable second derivatives. On Apple MPS, **training** with `dtype=auto`
now selects FP32 instead of the FP16 used by InstructPix2Pix inference. Explicit
`--dtype` choices are unchanged; run `config.json` records the effective dtype.
For a short diagnostic test, use `--steps 4 --guided-steps 1 --max-side 128`.

When gradients are still NaN, enable **Diagnose NaN gradients** in Training UI
or pass `--detect-anomaly` to `examples/train_heads.py`. PyTorch then reports
the first autograd operation that emits NaN gradients, with its forward stack
trace. This is slower, so leave it disabled for normal training. The trainer
continues to refuse optimizer steps on non-finite gradients (no silent fixes).

## Memory-efficient FLUX head training (v0.8.8)

The **Training UI** automatically enables three *exact-gradient* memory
optimizations for FLUX.2 Klein; they are also part of the shared
`configs/project.json` **train** profile and apply to CLI training. They do
not alter inference or the training objective:

- `train_checkpoint_transformer`: non-reentrant activation checkpoint of the
  native FLUX transformer, explicitly returning captured attention maps.
- `train_checkpoint_attention`: recompute per-chunk QK/logsumexp attention mass
  rather than retaining intermediate attention logits for every block.
- `train_checkpoint_vae`: recompute VAE decoder activations in image-loss backward.
- `train_query_chunk`: attention query chunk width (default `16`).

The inner guidance forward still uses math SDPA for second-order derivatives,
but *normal denoising* may use PyTorch's default fused SDPA. The complete
denoising graph remains differentiable: **no detach, truncated BPTT, removal
of layers, or approximation of head contributions** is introduced.
Recomputation makes training slower in exchange for reduced retained memory.

Example:

```bash
uv run --extra flux2 python examples/train_heads.py \
  --backend flux2 --dataset data/compbench-vpg/add/manifest_scribble.json \
  --device cuda --dtype bf16 --max-side 512 \
  --steps 3 --guided-steps 1 --text-cfg 1 --epochs 1 \
  --layer D2 --head D2:H00 --head D2:H01 --no-propagation
```

To compare with the old behavior, pass all three flags
`--no-transformer-checkpoint --no-attention-checkpoint --no-vae-checkpoint`.
When CUDA is available, each optimizer update logs peak allocated/reserved GiB.

**Validation limits:** CPU equivalence tests compare inner attention guidance,
full multi-step differentiable denoising and learned-logit gradients against
the uncheckpointed path. Full FLUX.2 GPU memory and numerical behavior must
still be checked on the target CUDA environment; the optimizations do not
guarantee a particular resolution or number of steps will fit.

To train **all 600 FLUX heads**, select all 25 layers and explicitly clear a
previously saved head filter with `--all-heads`:

```bash
layers=(); for layer in D{0..4} S{0..19}; do layers+=(--layer "$layer"); done
uv run --extra flux2 python examples/train_heads.py \
  --backend flux2 --dataset data/compbench-vpg/add/manifest_scribble.json \
  --device cuda --dtype bf16 --max-side 512 --steps 3 --guided-steps 1 \
  --text-cfg 1 --epochs 1 --no-propagation --all-heads "${layers[@]}"
```

This is a **memory feasibility test**, not a guarantee that all 600 heads fit
on a 96 GB GPU.
