"""Optional model adapters. Importing does not load model weights."""
from .flux2 import Flux2Klein
from .instruct_pix2pix import InstructPix2Pix

__all__ = ["Flux2Klein", "InstructPix2Pix"]
