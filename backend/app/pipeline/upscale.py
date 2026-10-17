"""Stage 3 - super-resolution upscaling.

Uses a Real-ESRGAN model loaded via ``spandrel`` (modern, basicsr-free). Runs
tiled inference to keep peak VRAM low. If the weights are missing or loading
fails, falls back to high-quality Lanczos resampling so the app still works.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


@torch.inference_mode()
def _tiled_sr(model, image_t: torch.Tensor, scale: int, tile: int, overlap: int) -> torch.Tensor:
    """Run a super-resolution model over ``image_t`` (B,C,H,W, range [0,1]) in tiles."""
    _, _, h, w = image_t.shape
    out_h, out_w = h * scale, w * scale
    out = torch.zeros((1, 3, out_h, out_w), device=image_t.device, dtype=image_t.dtype)
    weight = torch.zeros_like(out)

    stride = max(tile - overlap, 1)
    ys = list(range(0, max(h - overlap, 1), stride)) or [0]
    xs = list(range(0, max(w - overlap, 1), stride)) or [0]

    for y in ys:
        for x in xs:
            y2 = min(y + tile, h)
            x2 = min(x + tile, w)
            y1 = max(y2 - tile, 0)
            x1 = max(x2 - tile, 0)

            patch = image_t[:, :, y1:y2, x1:x2]
            sr_patch = model(patch).clamp(0, 1)

            out[:, :, y1 * scale : y2 * scale, x1 * scale : x2 * scale] += sr_patch
            weight[:, :, y1 * scale : y2 * scale, x1 * scale : x2 * scale] += 1.0

    return out / weight.clamp(min=1.0)


def _cap_to_max(image: Image.Image, target_max: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if target_max and longest > target_max:
        ratio = target_max / longest
        image = image.resize((round(w * ratio), round(h * ratio)), Image.LANCZOS)
    return image


def _lanczos_fallback(image: Image.Image, target_max: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if target_max and longest < target_max:
        ratio = target_max / longest
        image = image.resize((round(w * ratio), round(h * ratio)), Image.LANCZOS)
    return image


def upscale_image(
    model,
    image: Image.Image,
    device: str,
    tile: int = 256,
    overlap: int = 16,
    target_max: int = 1536,
) -> Image.Image:
    if model is None:
        return _lanczos_fallback(image, target_max)

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    scale = int(getattr(model, "scale", 4) or 4)
    sr = _tiled_sr(model, tensor, scale, tile, overlap)

    sr_np = (
        sr.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
    result = Image.fromarray(sr_np)
    return _cap_to_max(result, target_max)
