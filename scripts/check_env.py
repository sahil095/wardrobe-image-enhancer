"""Verify the local environment before running the app.

Run:
    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=" * 60)
    print(" Wardrobe Image Enhancer - environment check")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")

    ok = True

    # --- PyTorch / CUDA ---
    try:
        import torch

        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA build  : {torch.version.cuda}")
        cuda = torch.cuda.is_available()
        print(f"CUDA avail  : {cuda}")
        if cuda:
            print(f"GPU         : {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            print(f"VRAM        : {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB total")
            if total / 1e9 < 7.0:
                print("  ! Less than ~7 GB VRAM detected - use WORK_SIZE<=640 / sd15.")
        else:
            print("  ! CUDA not available. The app will run on CPU (very slow).")
            print("  ! Install the CUDA build: ")
            print("    pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu124")
            ok = False
    except Exception as exc:
        print(f"PyTorch     : NOT INSTALLED ({exc})")
        ok = False

    # --- key libraries ---
    for mod in ("diffusers", "transformers", "accelerate", "controlnet_aux", "rembg", "spandrel"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            print(f"{mod:<12}: MISSING ({exc})")
            ok = False

    # --- onnxruntime GPU provider (used by rembg) ---
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        gpu = "CUDAExecutionProvider" in providers
        print(f"onnxruntime : {ort.__version__} (CUDA provider: {gpu})")
        if not gpu:
            print("  ! rembg will run on CPU. Install onnxruntime-gpu for acceleration.")
    except Exception as exc:
        print(f"onnxruntime : MISSING ({exc})")

    print("=" * 60)
    print("RESULT:", "READY" if ok else "ISSUES FOUND (see above)")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
