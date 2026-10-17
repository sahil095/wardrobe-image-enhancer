"""Chains the three pipeline stages together.

This runs synchronously (heavy GPU work) and is called from the API inside a
thread-pool executor, guarded by an asyncio lock so only one job uses the GPU
at a time.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from PIL import Image

from .background import remove_background_to_white
from .enhance import generate
from .upscale import upscale_image

logger = logging.getLogger("wardrobe.pipeline")


def _round8(value: float) -> int:
    return max(8, int(round(value / 8)) * 8)


def resize_for_sd(image: Image.Image, work_size: int) -> Image.Image:
    """Scale so the longest side is ~work_size, snapped to a multiple of 8."""
    w, h = image.size
    scale = work_size / max(w, h)
    return image.resize((_round8(w * scale), _round8(h * scale)), Image.LANCZOS)


def run_pipeline(manager, settings, image: Image.Image, tags: Dict, params: Dict) -> Dict:
    # 1. Remove background -> white.
    clean = remove_background_to_white(image, manager.get_rembg())

    # 2. Downscale to a working resolution for diffusion.
    work = resize_for_sd(clean, settings.work_size)

    # 3. Soft-edge control image (locks garment shape).
    detector = manager.get_softedge()
    control = detector(work, safe=True).resize(work.size, Image.LANCZOS).convert("RGB")

    # 4. Generative de-wrinkle / straighten.
    result = generate(manager, work, control, clean, tags, params)

    # 5. Upscale for a crisp final image.
    if settings.upscale_enabled:
        result = upscale_image(
            manager.get_upscaler(),
            result,
            manager.device,
            tile=settings.upscale_tile,
            overlap=settings.upscale_tile_overlap,
            target_max=settings.upscale_target_max,
        )

    if manager.device == "cuda":
        torch.cuda.empty_cache()

    return {"clean": clean, "result": result}
