"""Pre-download all model weights so the first request is fast.

Run:
    python scripts/download_models.py

This is optional - the app will also download weights on demand at first use.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings  # noqa: E402

UPSCALE_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)


def download_hf():
    from huggingface_hub import snapshot_download

    is_sdxl = settings.base_model_type == "sdxl"
    base_id = settings.sdxl_model_id if is_sdxl else settings.sd15_model_id
    cn_id = settings.sdxl_controlnet_id if is_sdxl else settings.sd15_controlnet_id

    print(f"[1/4] Base model: {base_id}")
    snapshot_download(base_id)

    print(f"[2/4] ControlNet: {cn_id}")
    snapshot_download(cn_id)

    print(f"[3/4] IP-Adapter: {settings.ip_adapter_repo}")
    if is_sdxl:
        patterns = ["sdxl_models/ip-adapter_sdxl.bin", "sdxl_models/image_encoder/*"]
    else:
        patterns = ["models/ip-adapter_sd15.bin", "models/image_encoder/*"]
    snapshot_download(settings.ip_adapter_repo, allow_patterns=patterns)

    print(f"[3b] Soft-edge annotator: {settings.annotator_id}")
    snapshot_download(settings.annotator_id, allow_patterns=["*pidinet*", "*.json", "*.txt"])


def download_upscaler():
    import urllib.request

    dest = settings.upscale_weights
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[4/4] Upscaler already present: {dest}")
        return
    print(f"[4/4] Downloading Real-ESRGAN -> {dest}")
    urllib.request.urlretrieve(UPSCALE_URL, dest)


def download_rembg():
    # rembg downloads its model to ~/.u2net on first session creation.
    from rembg import new_session

    print(f"[+] rembg model: {settings.rembg_model}")
    new_session(settings.rembg_model)


def main() -> int:
    print("Downloading models (this can take several minutes / GBs)...")
    try:
        download_hf()
    except Exception as exc:
        print(f"  ! HuggingFace download issue: {exc}")
    try:
        download_rembg()
    except Exception as exc:
        print(f"  ! rembg download issue: {exc}")
    try:
        download_upscaler()
    except Exception as exc:
        print(f"  ! Upscaler download issue: {exc}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
