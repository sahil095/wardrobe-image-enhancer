"""Model loading, caching and VRAM-aware optimization.

A single :class:`ModelManager` owns every heavy model. Models are built lazily
and cached, so they are only loaded once (on startup warmup or first request)
and then kept resident for fast reruns.
"""

from __future__ import annotations

import logging
import threading

import torch

from .config import Settings

logger = logging.getLogger("wardrobe.models")


class ModelManager:
    def __init__(self, settings: Settings):
        self.s = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._img2img = None
        self._rembg = None
        self._softedge = None
        self._upscaler = None
        self._upscaler_tried = False

        # Guards model *construction* so two concurrent requests don't build the
        # same pipeline twice. Actual GPU inference is serialized at the request
        # layer with an asyncio lock.
        self._load_lock = threading.RLock()

        logger.info("ModelManager device=%s dtype=%s", self.device, self.dtype)

    # ------------------------------------------------------------------ rembg
    def get_rembg(self):
        if self._rembg is None:
            with self._load_lock:
                if self._rembg is None:
                    from rembg import new_session

                    if self.s.rembg_use_gpu:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        providers = ["CPUExecutionProvider"]
                    logger.info(
                        "Loading rembg session: %s (providers=%s)",
                        self.s.rembg_model,
                        providers,
                    )
                    self._rembg = new_session(self.s.rembg_model, providers=providers)
        return self._rembg

    # -------------------------------------------------------------- softedge
    def get_softedge(self):
        if self._softedge is None:
            with self._load_lock:
                if self._softedge is None:
                    from controlnet_aux import PidiNetDetector

                    logger.info("Loading PidiNet soft-edge detector")
                    self._softedge = PidiNetDetector.from_pretrained(self.s.annotator_id)
        return self._softedge

    # -------------------------------------------------------------- img2img
    def get_img2img(self):
        if self._img2img is None:
            with self._load_lock:
                if self._img2img is None:
                    self._img2img = self._build_img2img()
        return self._img2img

    def _build_img2img(self):
        from diffusers import ControlNetModel, DPMSolverMultistepScheduler

        is_sdxl = self.s.base_model_type == "sdxl"
        if is_sdxl:
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline as Pipe

            model_id = self.s.sdxl_model_id
            cn_id = self.s.sdxl_controlnet_id
            ip_subfolder, ip_weight = "sdxl_models", "ip-adapter_sdxl.bin"
        else:
            from diffusers import StableDiffusionControlNetImg2ImgPipeline as Pipe

            model_id = self.s.sd15_model_id
            cn_id = self.s.sd15_controlnet_id
            ip_subfolder, ip_weight = "models", "ip-adapter_sd15.bin"

        logger.info("Loading ControlNet: %s", cn_id)
        controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=self.dtype)

        logger.info("Loading base model: %s", model_id)
        kwargs = dict(controlnet=controlnet, torch_dtype=self.dtype)
        if not is_sdxl:
            kwargs.update(safety_checker=None, requires_safety_checker=False)
        pipe = Pipe.from_pretrained(model_id, **kwargs)

        # Fast, high-quality scheduler.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )

        # IP-Adapter: conditions generation on the reference garment so color /
        # pattern / overall appearance are preserved.
        logger.info("Loading IP-Adapter (%s/%s)", ip_subfolder, ip_weight)
        pipe.load_ip_adapter(self.s.ip_adapter_repo, subfolder=ip_subfolder, weight_name=ip_weight)
        pipe.set_ip_adapter_scale(self.s.ip_adapter_scale)

        self._apply_optimizations(pipe, is_sdxl)
        pipe.set_progress_bar_config(disable=True)
        logger.info("img2img pipeline ready")
        return pipe

    def _apply_optimizations(self, pipe, is_sdxl: bool):
        offload = self.s.use_cpu_offload or is_sdxl
        if self.device == "cuda" and offload:
            logger.info("Enabling model CPU offload (frees VRAM between stages)")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
            if self.device == "cuda":
                try:
                    pipe.unet.to(memory_format=torch.channels_last)
                except Exception:  # pragma: no cover - best effort
                    pass

        # IMPORTANT: do NOT enable attention slicing or xformers here. Both
        # replace the UNet attention processors and would clobber the IP-Adapter
        # processors installed by load_ip_adapter(), causing
        # "'tuple' object has no attribute 'shape'" at inference time.
        # PyTorch 2.x SDPA is used automatically and is already memory-efficient.
        #
        # VAE tiling/slicing only affect the VAE (not attention), so they are
        # safe and give big VRAM savings at decode time.
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    # -------------------------------------------------------------- upscaler
    def get_upscaler(self):
        if not self.s.upscale_enabled:
            return None
        if self._upscaler is None and not self._upscaler_tried:
            with self._load_lock:
                if self._upscaler is None and not self._upscaler_tried:
                    self._upscaler = self._build_upscaler()
                    self._upscaler_tried = True
        return self._upscaler

    def _build_upscaler(self):
        path = self.s.upscale_weights
        if not path.exists():
            logger.warning(
                "Upscale weights not found at %s - falling back to Lanczos. "
                "Run scripts/download_models.py to fetch Real-ESRGAN.",
                path,
            )
            return None
        try:
            from spandrel import ModelLoader

            logger.info("Loading upscaler: %s", path.name)
            model = ModelLoader().load_from_file(str(path))
            model = model.to(self.device).eval()
            return model
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to load upscaler (%s) - falling back to Lanczos", exc)
            return None

    # ------------------------------------------------------------------ util
    def warmup(self):
        """Best-effort eager load so the first request is fast."""
        for name, fn in (
            ("rembg", self.get_rembg),
            ("softedge", self.get_softedge),
            ("img2img", self.get_img2img),
            ("upscaler", self.get_upscaler),
        ):
            try:
                fn()
            except Exception as exc:
                logger.warning("Warmup for %s failed: %s", name, exc)

    def status(self) -> dict:
        return {
            "img2img": self._img2img is not None,
            "rembg": self._rembg is not None,
            "softedge": self._softedge is not None,
            "upscaler": self._upscaler is not None,
        }
