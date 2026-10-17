"""Stage 1 - background removal and white compositing.

Uses rembg (ONNX Runtime, GPU-accelerated when onnxruntime-gpu is installed).
Deterministic and very light on VRAM.
"""

from __future__ import annotations

from PIL import Image
from rembg import remove


def remove_background_to_white(image: Image.Image, session) -> Image.Image:
    """Return ``image`` with the subject composited onto a pure-white background."""
    cutout = remove(image, session=session)  # RGBA with alpha
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    background = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    background.alpha_composite(cutout)
    return background.convert("RGB")
