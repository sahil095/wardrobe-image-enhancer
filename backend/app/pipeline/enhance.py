"""Stage 2 - the generative de-wrinkle / straighten step.

Stable Diffusion img2img guided by:
  * ControlNet (soft-edge) -> locks the garment's overall shape.
  * IP-Adapter               -> locks color / pattern / appearance.
  * A tag-driven prompt      -> steers toward a clean product photo.

Keeping ``strength`` low preserves the exact garment; raising it removes more
wrinkles at the cost of fidelity. This is exposed in the UI.
"""

from __future__ import annotations

from typing import Dict

import torch
from PIL import Image


def build_prompt(tags: Dict[str, str]) -> tuple[str, str]:
    category = (tags.get("category") or "").strip() or "clothing garment"
    color = (tags.get("color") or "").strip()
    pattern = (tags.get("pattern") or "").strip()
    notes = (tags.get("notes") or "").strip()

    descriptor = " ".join(p for p in (color, pattern, category) if p)
    prompt = (
        f"professional product photoshoot of a {descriptor}, "
        "e-commerce catalog product photography, neatly steamed and ironed, "
        "perfectly straightened and symmetrical, smooth wrinkle-free fabric, "
        "clean product presentation, studio lighting, soft even light, subtle soft shadow, "
        "pure white seamless background, crisp detailed fabric texture, "
        "8k, ultra high quality, enhanced, highly detailed, sharp focus, photorealistic"
    )
    if notes:
        prompt += f", {notes}"

    negative = (
        "wrinkles, creases, folds, crumpled, messy, bunched up, shadows, dark, dull, "
        "flat lighting, cluttered background, props, hanger, person, mannequin, hands, body, "
        "deformed, distorted, asymmetrical, extra sleeves, extra fabric, missing parts, "
        "text, logo artifacts, watermark, lowres, low quality, blurry, out of focus, "
        "jpeg artifacts, noise, oversaturated, cartoon, illustration"
    )
    return prompt, negative


@torch.inference_mode()
def generate(
    manager,
    work_image: Image.Image,
    control_image: Image.Image,
    reference_image: Image.Image,
    tags: Dict[str, str],
    params: Dict[str, float],
) -> Image.Image:
    pipe = manager.get_img2img()
    prompt, negative = build_prompt(tags)

    pipe.set_ip_adapter_scale(float(params["ip_scale"]))

    gen_device = "cuda" if manager.device == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(params["seed"]))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=work_image,
        control_image=control_image,
        ip_adapter_image=reference_image,
        strength=float(params["strength"]),
        num_inference_steps=int(params["steps"]),
        guidance_scale=float(params["guidance"]),
        controlnet_conditioning_scale=float(params["controlnet_scale"]),
        generator=generator,
    ).images[0]

    return result
