import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visual_prompt_guidance import (
    Request, Space, AttentionBatch, Prediction, Point, Engine, Guidance, Objective, Loss,
    Learned, HeadSubset, Adaptive, HeadTrainer, TrainConfig, ImageLossConfig, losses,
)


class DifferentiableTinyAdapter:
    """Analytic fake generator with a nontrivial second-order training gradient."""
    def space(self, request):
        return Space((16, 16), (4, 4))

    def prepare(self, request, target):
        return {"latent": torch.full((1, 1, 4, 4), .2, dtype=torch.float32), "steps": (2, 1)}

    def steps(self, state):
        return state["steps"]

    def latent(self, state):
        return state["latent"]

    def set_latent(self, state, latent):
        state["latent"] = latent

    def forward(self, state, step, *, attention):
        x = state["latent"]
        maps = None
        if attention is not None:
            yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
            coords = (yy + xx).float().to(x.device)[None, None]
            a = torch.sigmoid(1.5 * x + coords * .2)
            b = torch.sigmoid(-.7 * x + (6 - coords) * .2)
            maps = AttentionBatch(torch.cat([a, b], 1), ("L0:H00", "L0:H01"))
        return Prediction(.1 * x + .03 * torch.tanh(x), maps)

    def advance(self, state, step, prediction):
        state["latent"] = state["latent"] - .25 * prediction.output

    def decode_tensor(self, state):
        from torch.nn.functional import interpolate
        x = torch.sigmoid(state["latent"])
        return interpolate(x.repeat(1, 3, 1, 1), size=(16, 16), mode="bilinear", align_corners=False)

    def decode(self, state):
        return self.decode_tensor(state).detach().cpu()


def engine(policy):
    return Engine(DifferentiableTinyAdapter(), Objective(Loss("coverage", losses.coverage)),
                  policy=policy, guidance=Guidance(strength=.02, max_steps=2))


def test_second_order_gradient_nonzero():
    policy = Learned()
    example = Request(data={})
    output = engine(policy).run_differentiable(example, Point(4, 4)).output
    scalar = output.square().mean()
    grad = torch.autograd.grad(scalar, policy.logits)[0]
    assert grad.shape == (2,)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 1e-9


def test_second_order_matches_finite_difference():
    policy = Learned()
    def objective():
        return engine(policy).run_differentiable(Request({}), Point(4, 4)).output.square().mean()
    y = objective()
    analytical = torch.autograd.grad(y, policy.logits)[0][0].item()
    with torch.no_grad():
        policy.logits[0] += 1e-3
    a = objective().item()
    with torch.no_grad():
        policy.logits[0] -= 2e-3
    b = objective().item()
    numeric = (a - b) / .002
    assert analytical == pytest.approx(numeric, rel=.15, abs=2e-5)


def test_training_dataset_end_to_end_and_checkpoint(tmp_path):
    Image.fromarray(np.full((16,16,3),90,dtype=np.uint8)).save(tmp_path / "input.png")
    Image.fromarray(np.full((16,16,3),160,dtype=np.uint8)).save(tmp_path / "target.png")
    Image.fromarray(np.full((16,16),255,dtype=np.uint8)).save(tmp_path / "mask.png")
    (tmp_path/"manifest.json").write_text(json.dumps([{
        "input_image":"input.png", "target_image":"target.png", "edit_mask":"mask.png",
        "prompt":"add a dog", "object_phrase":"dog", "visual":{"type":"point", "x":4, "y":4},
    }]))
    policy=Learned(selected=("L0:H01",))
    ckpt=tmp_path/"weights.json"; hist=tmp_path/"history.json"
    trainer=HeadTrainer(engine(policy),config=TrainConfig(epochs=2,learning_rate=.005,
        checkpoint=str(ckpt), history=str(hist)), image_loss=ImageLossConfig(pixel="mse"))
    result=trainer.fit(tmp_path)
    assert len(result.history)==2
    assert ckpt.is_file() and hist.is_file()
    saved=Learned.load(ckpt)
    assert saved.labels == ("L0:H00","L0:H01")
    assert saved.weights(saved.labels,device=torch.device("cpu")).tolist()==pytest.approx([0,1])
    assert saved.logits.grad is None


def test_normal_inference_remains_detached():
    policy=Learned()
    result=engine(policy).run(Request({}),Point(4,4))
    assert not result.output.requires_grad
    assert policy.logits is not None


def test_adaptive_subset_updates_only_chosen():
    subset=HeadSubset(Adaptive(temperature=.1,ema=1,mix=0), selected=("a","c"))
    labels=("a","b","c")
    assert subset.weights(labels,device=torch.device("cpu")).tolist()==pytest.approx([.5,0,.5])
    subset.update(labels,torch.tensor([[2.,99.,2.]]),torch.tensor([[1.,0.,1.5]]))
    w=subset.weights(labels,device=torch.device("cpu"))
    assert w[0]>w[2]>0 and w[1]==0


def test_reject_training_detached_backend():
    class Bad(DifferentiableTinyAdapter):
        def decode_tensor(self, state):
            return super().decode_tensor(state).detach()
    policy=Learned()
    run=Engine(Bad(),Objective(Loss('coverage',losses.coverage)),policy=policy,
               guidance=Guidance(strength=.01,max_steps=1))
    with pytest.raises(RuntimeError, match="detached"):
        run.run_differentiable(Request({}),Point(4,4))


def test_training_updates_both_active_head_logits(tmp_path):
    Image.fromarray(np.full((16,16,3),110,dtype=np.uint8)).save(tmp_path / "source.png")
    Image.fromarray(np.full((16,16,3),215,dtype=np.uint8)).save(tmp_path / "reference.png")
    (tmp_path / "manifest.json").write_text(json.dumps([{
        "input_image": "source.png", "target_image": "reference.png",
        "prompt": "add a dog", "object_phrase": "dog",
        "visual": {"type": "point", "x":4, "y":4},
    }]))
    policy = Learned()
    trainer = HeadTrainer(engine(policy),
        config=TrainConfig(epochs=3, learning_rate=0.05,
                           checkpoint=str(tmp_path / "learned.json"),
                           history=str(tmp_path / "train.json")))
    history = trainer.fit(tmp_path).history
    assert all(event["grad_norm"] > 0 for event in history)
    weights = policy.weights(policy.labels, device=torch.device("cpu"))
    assert abs(float((weights[0] - weights[1]).detach())) > 1e-6
