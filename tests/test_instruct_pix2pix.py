"""Offline integration tests that exercise real hook/engine wiring using tiny mocks.

No Hugging Face checkpoint download is necessary for these tests.
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from visual_prompt_guidance import (
    Engine, Guidance, Request, Point, Scribble, Objective, Loss,
    Adaptive, Manual, AffinityGrow, losses,
)
from visual_prompt_guidance.backends import InstructPix2Pix
from visual_prompt_guidance.backends.instruct_pix2pix_attention import (
    object_tokens, UNetAttentionCapture, default_layers,
)


CROSS_NAME = "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
SELF_NAME = CROSS_NAME.replace("attn2", "attn1")


class MockTokenizer:
    model_max_length = 77

    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            offsets = [(0, 0)]  # BOS
            offsets.extend((i, i + 1) for i in range(len(text)))
            offsets.extend([(0, 0)] * (77 - len(offsets)))
            return {"offset_mapping": offsets}
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": [ord(c) for c in text]}
        ids = [1] + [ord(c) for c in text] + [2]
        ids.extend([0] * (77 - len(ids)))
        return {"input_ids": ids[:77]}


class TinyAttention(nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.heads = 2
        self.to_q = nn.Linear(4, 8, bias=False)
        self.to_k = nn.Linear(8 if cross else 4, 8, bias=False)
        self.scale = 1 / math.sqrt(4)

    def forward(self, hidden_states, encoder_hidden_states=None):
        return hidden_states


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = TinyAttention()
        self.attn2 = TinyAttention(cross=True)
        self.config = type("Config", (), {"in_channels": 8})()
        self.dtype = torch.float32

    def named_modules(self, *args, **kwargs):
        yield "", self
        yield SELF_NAME, self.attn1
        yield CROSS_NAME, self.attn2

    def forward(self, input, t, encoder_hidden_states, return_dict=False):
        # Input is [B,8,32,32] at 256px. Cross attention at 16x16.
        h = F.avg_pool2d(input[:, :4], 2).flatten(2).transpose(1, 2)
        h = self.attn1(h)
        h = self.attn2(h, encoder_hidden_states=encoder_hidden_states)
        return (input[:, :4] * .1,)


class TinyScheduler:
    init_noise_sigma = 1.
    def __init__(self):
        self.timesteps = torch.tensor([2., 1.])

    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.arange(steps, 0, -1, dtype=torch.float32, device=device)

    def scale_model_input(self, x, step):
        return x

    def step(self, pred, step, latents, **kwargs):
        return (latents - .2 * pred,)


class TinyVAE:
    def __init__(self):
        self.config = type("Config", (), {"latent_channels": 4, "scaling_factor": .18215})()
        self.dtype = torch.float32

    def decode(self, latents, return_dict=False):
        return (F.interpolate(latents[:, :3], scale_factor=8, mode="nearest"),)


class TinyImageProcessor:
    def preprocess(self, image):
        return torch.zeros((1, 3, image.height, image.width))

    def postprocess(self, pixels, output_type="pil", do_denormalize=None):
        # Stub conversion to make decode observable.
        return [Image.new("RGB", (pixels.shape[-1], pixels.shape[-2]), (255, 0, 0))]


class TinyPipeline:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.unet = TinyUNet()
        self.vae = TinyVAE()
        self.scheduler = TinyScheduler()
        self.image_processor = TinyImageProcessor()
        self.vae_scale_factor = 8
        self._execution_device = torch.device("cpu")

    def _encode_prompt(self, prompt, device, n, do_cfg, negative_prompt=None):
        b = 3 if do_cfg else 1
        return torch.ones(b, 77, 8, device=device)

    def prepare_image_latents(self, image, bs, n, dtype, device, do_cfg):
        return torch.zeros((3 if do_cfg else 1, 4, image.shape[-2] // 8, image.shape[-1] // 8), device=device, dtype=dtype)

    def prepare_latents(self, bs, channels, height, width, dtype, device, generator, latents):
        return torch.randn((bs, channels, height // 8, width // 8), generator=generator, device=device, dtype=dtype)

    def prepare_extra_step_kwargs(self, generator, eta):
        return {}

    def run_safety_checker(self, pixels, device, dtype):
        return pixels, None

    def maybe_free_model_hooks(self):
        pass


def make_request():
    return Request(
        {"image": Image.new("RGB", (300, 260)), "prompt": "make a bird blue", "object_phrase": "bird"},
        options={"max_side": 256, "steps": 3, "seed": 3, "cfg": 7.5, "image_cfg": 1.5},
    )


def test_object_token_offsets():
    assert object_tokens(MockTokenizer(), "make a bird blue", "bird") == (8, 9, 10, 11)


def test_head_capture_gradient_and_rectangular_grid():
    unet = TinyUNet()
    # Fake a rectangular native latent size: 32x24 -> attention 16x12.
    latent = torch.randn((1, 4, 32, 24), requires_grad=True)
    collector = UNetAttentionCapture(unet, (8, 9, 10, 11), (32, 24), (16, 12), (CROSS_NAME,),
                                     affinity_layers=(CROSS_NAME,), need_affinity=True)
    with collector.capture():
        out = unet(torch.cat([latent, torch.zeros_like(latent)], dim=1), torch.tensor(1.),
                   encoder_hidden_states=torch.randn(1, 77, 8))
    result = collector.result()
    assert result.maps.shape == (1, 2, 16, 12)
    assert result.affinity.shape == (1, 192, 192)
    assert result.labels == (f"{CROSS_NAME}:H00", f"{CROSS_NAME}:H01")
    grad = torch.autograd.grad(result.maps.square().sum(), latent)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_engine_pix2pix_uniform_and_adaptive():
    for policy in (None, Adaptive(temperature=.5)):
        adapter = InstructPix2Pix(pipeline=TinyPipeline())
        engine = Engine(
            adapter, Objective(Loss("coverage", losses.coverage)),
            guidance=Guidance(.02, max_steps=2), policy=policy,
        )
        result = engine.run(make_request(), Point(120, 100))
        assert result.output.size == (300, 260)
        assert len(result.events) == 3
        assert result.events[0].losses["coverage"] > 0
        if policy is not None:
            assert len(result.events[0].scores_after) == 2


def test_manual_head_and_scribble():
    adapter = InstructPix2Pix(pipeline=TinyPipeline())
    assert adapter.attention_layers() == (CROSS_NAME,)
    engine = Engine(adapter, Objective(Loss("coverage", losses.coverage)),
                    guidance=Guidance(.01, 1, layers=(CROSS_NAME,)),
                    policy=Manual({f"{CROSS_NAME}:H01": 1.}))
    result = engine.run(make_request(), Scribble(path=[(40, 40), (100, 60)]))
    assert result.events[0].weights == (0., 1.)


def test_affinity_propagation_runs():
    adapter = InstructPix2Pix(pipeline=TinyPipeline())
    engine = Engine(
        adapter, Objective(Loss("coverage", losses.coverage)),
        guidance=Guidance(.01, max_steps=2),
        transform=AffinityGrow(start=0, stop=2, threshold=20., limit=2),
    )
    result = engine.run(make_request(), Point(120, 100))
    assert len(result.events) == 3


def test_native_three_way_cfg():
    adapter = InstructPix2Pix(pipeline=TinyPipeline())
    state = adapter.prepare(make_request(), Point(120, 100).project(adapter.space(make_request())))
    prediction = adapter.forward(state, state.timesteps[0], attention=None)
    assert prediction.output.shape == state.latents.shape


def test_bad_options():
    adapter = InstructPix2Pix(pipeline=TinyPipeline())
    req = make_request()
    with pytest.raises(ValueError, match="not a native inpaint"):
        adapter.prepare(Request(req.data, {**req.options, "mode": "inpaint"}), Point(10, 20).project(adapter.space(req)))
