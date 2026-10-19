"""Compare hosted garment-enhancement APIs on YOUR OWN image.

Runs one input photo through whichever providers you have keys for, saves each
result, and builds a side-by-side `compare.html` so you can judge quality before
committing to one provider.

Providers (all optional - only those with a key set are run):
  - Photoroom   (specialized apparel: AI ironing + white bg + soft shadow,
                 and a separate ghost-mannequin variant)   env: PHOTOROOM_API_KEY
  - fal.ai Flux Kontext   (generic identity-preserving edit) env: FAL_KEY
  - fal.ai Qwen-Image-Edit (cheapest generic edit)           env: FAL_KEY

Get free credits to test:
  - Photoroom: https://www.photoroom.com/api  (free trial key)
  - fal.ai:    https://fal.ai                  ($20 free signup credit)

Usage (Anaconda Prompt):
  set PHOTOROOM_API_KEY=xxxx
  set FAL_KEY=xxxx
  python scripts\\compare_providers.py --image path\\to\\garment.jpg ^
      --category "t-shirt" --color "navy blue" --pattern "plain"

Then open outputs\\compare\\compare.html
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "compare"

TIMEOUT = 240


def build_prompt(category: str, color: str, pattern: str) -> str:
    desc = " ".join(p for p in (color, pattern, category) if p) or "clothing garment"
    return (
        f"commercial e-commerce product photo of this exact {desc}, "
        "perfectly ironed and completely wrinkle-free, neatly flat-laid and straightened, "
        "pure white studio background, soft even studio lighting, subtle soft shadow, "
        "keep the exact same garment, same colours, same fabric, same logos and text, "
        "do not change the design, 8k, ultra sharp, high-end catalog photography"
    )


def to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# Providers. Each returns (ok: bool, note: str). Saves PNG to OUT_DIR on success.
# --------------------------------------------------------------------------- #
def run_photoroom(image_path: Path, label: str, extra: dict) -> tuple[bool, str]:
    key = os.environ.get("PHOTOROOM_API_KEY")
    if not key:
        return False, "skipped (no PHOTOROOM_API_KEY)"
    t0 = time.time()
    try:
        with open(image_path, "rb") as fh:
            resp = requests.post(
                "https://image-api.photoroom.com/v2/edit",
                headers={
                    "x-api-key": key,
                    # Required to use advanced shadow presets (ai.preset-soft).
                    "pr-ai-shadows-model-version": "2026-04-15",
                },
                files={"imageFile": (image_path.name, fh)},
                data={"export.format": "png", **extra},
                timeout=TIMEOUT,
            )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        (OUT_DIR / f"{label}.png").write_bytes(resp.content)
        return True, f"ok in {time.time() - t0:.1f}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"error: {exc}"


def run_fal(image_path: Path, label: str, model_slug: str, prompt: str) -> tuple[bool, str]:
    key = os.environ.get("FAL_KEY")
    if not key:
        return False, "skipped (no FAL_KEY)"
    t0 = time.time()
    try:
        resp = requests.post(
            f"https://fal.run/{model_slug}",
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "image_url": to_data_uri(image_path)},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        images = data.get("images") or data.get("image")
        if isinstance(images, dict):
            images = [images]
        if not images:
            return False, f"no image in response: {str(data)[:200]}"
        url = images[0]["url"]
        img = requests.get(url, timeout=TIMEOUT).content
        (OUT_DIR / f"{label}.png").write_bytes(img)
        return True, f"ok in {time.time() - t0:.1f}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"error: {exc}"


def build_html(image_path: Path, results: list[tuple[str, str, str]]):
    # results: list of (label, filename_or_empty, note)
    cards = []
    # original
    orig_rel = "original" + image_path.suffix
    (OUT_DIR / orig_rel).write_bytes(image_path.read_bytes())
    cards.append(
        f'<figure><figcaption>Original</figcaption>'
        f'<img src="{orig_rel}" /></figure>'
    )
    for label, fname, note in results:
        if fname:
            body = f'<img src="{fname}" />'
        else:
            body = f'<div class="fail">{note}</div>'
        cards.append(
            f'<figure><figcaption>{label}<br><small>{note}</small></figcaption>{body}</figure>'
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Provider comparison</title>
<style>
 body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:20px}}
 h1{{font-size:18px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
 figure{{margin:0;background:#1b1b1b;border:1px solid #333;border-radius:10px;padding:10px}}
 figcaption{{font-size:13px;margin-bottom:8px;color:#bbb}}
 img{{width:100%;background:#fff;border-radius:6px;display:block}}
 .fail{{padding:40px 10px;text-align:center;color:#f88;font-size:12px}}
 small{{color:#888}}
</style></head><body>
<h1>Garment enhancement - provider comparison</h1>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    (OUT_DIR / "compare.html").write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--category", default="")
    ap.add_argument("--color", default="")
    ap.add_argument("--pattern", default="")
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt or build_prompt(args.category, args.color, args.pattern)
    print(f"Prompt (generic editors): {prompt}\n")

    jobs = [
        (
            "photoroom_ironed_white",
            lambda: run_photoroom(
                args.image,
                "photoroom_ironed_white",
                {
                    "ironing.mode": "ai.auto",
                    "background.color": "FFFFFF",
                    "shadow.mode": "ai.preset-soft",
                    "lighting.mode": "ai.preserve-hue-and-saturation",
                    "padding": "0.08",
                },
            ),
        ),
        (
            "photoroom_ghost_mannequin",
            lambda: run_photoroom(
                args.image,
                "photoroom_ghost_mannequin",
                {"ghostMannequin.mode": "ai.auto", "background.color": "FFFFFF"},
            ),
        ),
        (
            "fal_flux_kontext",
            lambda: run_fal(args.image, "fal_flux_kontext", "fal-ai/flux-pro/kontext", prompt),
        ),
        (
            "fal_qwen_image_edit",
            lambda: run_fal(args.image, "fal_qwen_image_edit", "fal-ai/qwen-image-edit", prompt),
        ),
    ]

    results = []
    for label, fn in jobs:
        print(f"-> {label} ...", end=" ", flush=True)
        ok, note = fn()
        print(note)
        results.append((label, f"{label}.png" if ok else "", note))

    build_html(args.image, results)
    print(f"\nDone. Open: {OUT_DIR / 'compare.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
