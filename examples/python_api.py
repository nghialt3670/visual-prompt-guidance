"""Example host application: all components composed via the public API."""
from PIL import Image
from visual_prompt_guidance import (
    Engine, Guidance, Request, Point, Loss, Objective, Adaptive, FileObserver, losses,
)
from visual_prompt_guidance.backends import Flux2Klein

backend=Flux2Klein(device="auto")
engine=Engine(
    backend,
    Objective(
        Loss("coverage", losses.coverage, weight=1.),
        Loss("centroid", losses.centroid, weight=.5),
    ),
    guidance=Guidance(strength=.01,max_steps=10,layers=("D2","D3")),
    policy=Adaptive(temperature=.01),
    # A scoring objective can be different from guidance objectives:
    scoring=Objective(Loss("coverage",losses.coverage)),
    observers=[FileObserver("outputs/python",maps=False)],
)
result=engine.run(
    Request({"image":Image.open("input.png"),"prompt":"Insert a bird","object_phrase":"bird"},
            options={"steps":30,"mode":"base","max_side":512,"seed":42}),
    Point(x=200,y=130,sigma=12),
)
result.output.save("outputs/python/image.png")
