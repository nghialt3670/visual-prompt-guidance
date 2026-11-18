import math
from dataclasses import dataclass
import numpy as np
import pytest
import torch
from PIL import Image
from visual_prompt_guidance import (
    Request, Space, Target, AttentionBatch, Prediction, Engine, Guidance,
    Point, Box, Scribble, Loss, Objective, Uniform, Manual, Adaptive,
    FileObserver, AffinityGrow, losses,
)


class FakeAdapter:
    """Implements the public backend contract without model-specific dependencies."""
    def space(self, request):
        return Space((128,96),(12,16))

    def prepare(self, request, target):
        return {"latents": torch.zeros(1,12*16,requires_grad=False),
                "steps": [2.,1.,0.], "target":target, "step":0}

    def steps(self,state):
        return state["steps"]

    def latent(self,state):
        return state["latents"]

    def set_latent(self,state,latent):
        state["latents"] = latent

    def forward(self,state,step,*,attention):
        z=state["latents"]
        maps=None
        if attention is not None:
            base=torch.sigmoid(z.reshape(1,1,12,16))
            head_maps=torch.cat((base,.5*base+0.1),dim=1)
            affinity=None
            if attention.need_affinity:
                n=12*16
                affinity=torch.eye(n).unsqueeze(0).to(z.device)*.9+torch.ones(1,n,n,device=z.device)*(.1/n)
            maps=AttentionBatch(head_maps,("layer-one:head-a","layer-two:head-b"),affinity)
        return Prediction(output=-z,attention=maps)

    def advance(self,state,step,prediction):
        state["latents"] = (state["latents"] + .05 * prediction.output).detach()

    def decode(self,state):
        return state["latents"].detach().clone()


def test_point_box_scribble():
    space=Space((128,96),(12,16))
    for prompt in [Point(30,20),Box(10,15,80,70),Scribble(path=[(2,3),(50,20),(80,60)]),
                   Scribble(mask=Image.fromarray(np.pad(np.ones((20,20),dtype=np.uint8)*255,((2,74),(2,106)))))]:
        t=prompt.project(space)
        assert t.density.shape==(1,1,12,16)
        assert t.density.sum().item()==pytest.approx(1.)
        assert t.support.max()==1
        assert t.edit_mask.max()==1


def test_loss_shapes_and_gradients():
    target=Scribble(path=[(15,5),(60,80)]).project(Space((128,96),(12,16)))
    maps=torch.rand(2,3,12,16,requires_grad=True)
    # repeat target across batch
    target=Target(*(x.repeat(2,1,1,1) for x in [target.density,target.support,target.edit_mask]),target.features)
    obj=Objective(*[Loss(name,fn) for name,fn in losses.BUILTIN_LOSSES.items()])
    total,parts=obj.evaluate(maps,target)
    assert total.shape==(2,3)
    assert all(x.shape==(2,3) for x in parts.values())
    total.mean().backward()
    assert maps.grad is not None and torch.isfinite(maps.grad).all()


def test_injected_custom_loss():
    def custom(maps,target,offset):
        return (maps-offset).square().mean((-2,-1))
    obj=Objective(Loss("my-loss",custom,1.25,{"offset":.2}))
    t=Point(50,20).project(Space((128,96),(12,16)))
    assert obj(torch.zeros(1,2,12,16),t).shape==(1,2)


def test_manual_arbitrary_labels():
    m=Manual({"layer-two:head-b":2.})
    weights=m.weights(("layer-one:head-a","layer-two:head-b"),device=torch.device("cpu"))
    assert weights.tolist()==[0.,1.]
    with pytest.raises(ValueError,match="Unknown head"):
        m.weights(("unknown",),device=torch.device("cpu"))


def test_adaptive_updates():
    policy=Adaptive(temperature=.01,ema=1.,mix=0)
    ids=("alpha","beta")
    assert policy.weights(ids,device=torch.device("cpu")).tolist()==pytest.approx([.5,.5])
    policy.update(ids,torch.tensor([[2.,2.]]),torch.tensor([[1.,1.9]]))
    assert policy.weights(ids,device=torch.device("cpu"))[0]>.99


def test_engine_uniform(tmp_path):
    observer=FileObserver(tmp_path,maps=True,images=True,interval=1)
    run=Engine(FakeAdapter(),Objective(Loss("cov",losses.coverage)),
               guidance=Guidance(.02,max_steps=2),observers=[observer])
    result=run.run(Request({"unused":True}),Box(20,10,70,50))
    assert len(result.events)==3
    assert result.events[0].losses["cov"] > 0
    assert len(result.events[0].scores_before)==2
    assert result.events[-1].losses=={}
    assert (tmp_path/"heads.csv").exists()
    assert (tmp_path/"head_maps"/"step_000"/"head_000.npy").exists()
    assert (tmp_path/"head_maps"/"step_000"/"head_001.png").exists()


def test_engine_adaptive_custom_score():
    score=Objective(Loss("score",losses.coverage))
    obj=Objective(Loss("custom",losses.centroid))
    policy=Adaptive(temperature=.1)
    run=Engine(FakeAdapter(),obj,scoring=score,policy=policy,guidance=Guidance(.04,max_steps=2))
    result=run.run(Request(None),Point(60,40))
    assert len(result.events[0].scores_before)==2
    assert len(result.events[0].scores_after)==2
    assert sum(result.events[0].extras["weights_next"])==pytest.approx(1.)


def test_engine_uses_transform_without_prompt_branch():
    run=Engine(FakeAdapter(),Objective(Loss("cov",losses.coverage)),
               transform=AffinityGrow(start=0,stop=2,limit=3,threshold=2.,max_fraction=.3),
               guidance=Guidance(.03,max_steps=2))
    result=run.run(Request(None),Point(35,30))
    assert len(result.events)==3


def test_no_guidance_runs_without_attention():
    result=Engine(FakeAdapter(),Objective(Loss("cov",losses.coverage)),
                  guidance=Guidance(strength=0)).run(Request(None),Point(35,30))
    assert all(not e.losses for e in result.events)
