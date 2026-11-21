"""Backwards-compatible launcher for the standalone Inference UI.

Train in the dedicated app instead: python examples/ui_training.py --port 7862
"""
from ui_inference import build_demo
from ui_common import serve

if __name__ == '__main__':
    serve(build_demo, default_port=7861)
