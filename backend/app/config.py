"""Central configuration.

All values can be overridden with environment variables so you can tune the
pipeline without editing code (handy for experimenting on an 8 GB GPU).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root so cwd doesn't matter)
#   this file:  <root>/backend/app/config.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _b(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


@dataclass
class Settings:
    # --- base model selection -------------------------------------------------
    # "sd15" (default, fits 8 GB comfortably) or "sdxl" (higher quality, slower,
    # auto-uses CPU offload).
    base_model_type: str = field(default_factory=lambda: _s("BASE_MODEL_TYPE", "sd15").lower())

    sd15_model_id: str = field(default_factory=lambda: _s("SD15_MODEL_ID", "Lykon/dreamshaper-8"))
    sd15_controlnet_id: str = field(
        default_factory=lambda: _s("SD15_CONTROLNET_ID", "lllyasviel/control_v11p_sd15_softedge")
    )
    sdxl_model_id: str = field(
        default_factory=lambda: _s("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    )
    sdxl_controlnet_id: str = field(
        default_factory=lambda: _s("SDXL_CONTROLNET_ID", "diffusers/controlnet-canny-sdxl-1.0")
    )

    annotator_id: str = field(default_factory=lambda: _s("ANNOTATOR_ID", "lllyasviel/Annotators"))
    ip_adapter_repo: str = field(default_factory=lambda: _s("IP_ADAPTER_REPO", "h94/IP-Adapter"))

    # --- background removal ---------------------------------------------------
    rembg_model: str = field(default_factory=lambda: _s("REMBG_MODEL", "isnet-general-use"))
    # rembg runs on CPU by default: it is fast enough for a single image and
    # avoids the onnxruntime-gpu CUDA DLL dependency mess on Windows. Set
    # REMBG_USE_GPU=true only if you have a matching CUDA/cuDNN runtime for
    # onnxruntime-gpu on your PATH.
    rembg_use_gpu: bool = field(default_factory=lambda: _b("REMBG_USE_GPU", False))

    # --- upscaling ------------------------------------------------------------
    upscale_enabled: bool = field(default_factory=lambda: _b("UPSCALE_ENABLED", True))
    upscale_weights: Path = field(
        default_factory=lambda: MODELS_DIR / "upscale" / "RealESRGAN_x4plus.pth"
    )
    upscale_tile: int = field(default_factory=lambda: _i("UPSCALE_TILE", 256))
    upscale_tile_overlap: int = field(default_factory=lambda: _i("UPSCALE_TILE_OVERLAP", 16))
    upscale_target_max: int = field(default_factory=lambda: _i("UPSCALE_TARGET_MAX", 1536))

    # --- generation defaults (all overridable per-request from the UI) --------
    work_size: int = field(default_factory=lambda: _i("WORK_SIZE", 768))
    # Defaults tuned for a real "product photoshoot" regeneration (not just a
    # background removal). Higher strength = more transformation; IP-Adapter
    # keeps it the SAME garment; lower ControlNet lets it straighten/de-wrinkle
    # instead of copying the original creases.
    strength: float = field(default_factory=lambda: _f("STRENGTH", 0.6))
    steps: int = field(default_factory=lambda: _i("STEPS", 30))
    guidance: float = field(default_factory=lambda: _f("GUIDANCE", 7.0))
    controlnet_scale: float = field(default_factory=lambda: _f("CONTROLNET_SCALE", 0.4))
    ip_adapter_scale: float = field(default_factory=lambda: _f("IP_ADAPTER_SCALE", 0.8))

    # --- runtime --------------------------------------------------------------
    warmup_on_startup: bool = field(default_factory=lambda: _b("WARMUP_ON_STARTUP", True))
    use_cpu_offload: bool = field(default_factory=lambda: _b("USE_CPU_OFFLOAD", False))
    max_upload_mb: int = field(default_factory=lambda: _i("MAX_UPLOAD_MB", 25))

    def summary(self) -> dict:
        return {
            "base_model_type": self.base_model_type,
            "model_id": self.sdxl_model_id if self.base_model_type == "sdxl" else self.sd15_model_id,
            "work_size": self.work_size,
            "strength": self.strength,
            "steps": self.steps,
            "guidance": self.guidance,
            "controlnet_scale": self.controlnet_scale,
            "ip_adapter_scale": self.ip_adapter_scale,
            "upscale_enabled": self.upscale_enabled,
        }


settings = Settings()
