"""FastAPI application: serves the web UI and the /api/enhance endpoint."""

from __future__ import annotations

import asyncio
import io
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .config import FRONTEND_DIR, OUTPUTS_DIR, settings
from .models import ModelManager
from .pipeline.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wardrobe")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = ModelManager(settings)
    app.state.gpu_lock = asyncio.Lock()

    if settings.warmup_on_startup:
        logger.info("Warming up models (first start downloads weights)...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, app.state.manager.warmup)
        logger.info("Warmup complete: %s", app.state.manager.status())

    yield

    app.state.manager = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Wardrobe Image Enhancer", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    manager: ModelManager = app.state.manager
    info = {
        "status": "ok",
        "device": manager.device,
        "cuda": torch.cuda.is_available(),
        "models": manager.status(),
        "settings": settings.summary(),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["vram_gb"] = {"free": round(free / 1e9, 2), "total": round(total / 1e9, 2)}
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


@app.post("/api/enhance")
async def enhance(
    file: UploadFile = File(...),
    category: str = Form(""),
    color: str = Form(""),
    pattern: str = Form(""),
    notes: str = Form(""),
    strength: Optional[float] = Form(None),
    steps: Optional[int] = Form(None),
    guidance: Optional[float] = Form(None),
    controlnet_scale: Optional[float] = Form(None),
    ip_scale: Optional[float] = Form(None),
    seed: int = Form(-1),
):
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB limit")
    try:
        # exif_transpose honors the camera orientation flag so phone photos are
        # not saved / processed sideways.
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    used_seed = seed if (seed is not None and seed >= 0) else random.randint(0, 2**31 - 1)
    params = {
        "strength": strength if strength is not None else settings.strength,
        "steps": steps if steps else settings.steps,
        "guidance": guidance if guidance is not None else settings.guidance,
        "controlnet_scale": controlnet_scale if controlnet_scale is not None else settings.controlnet_scale,
        "ip_scale": ip_scale if ip_scale is not None else settings.ip_adapter_scale,
        "seed": used_seed,
    }
    tags = {"category": category, "color": color, "pattern": pattern, "notes": notes}
    job_id = uuid.uuid4().hex[:12]

    manager: ModelManager = app.state.manager
    loop = asyncio.get_event_loop()

    async with app.state.gpu_lock:  # serialize GPU access
        start = time.time()
        try:
            out = await loop.run_in_executor(
                None, run_pipeline, manager, settings, image, tags, params
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(
                status_code=507,
                detail="GPU out of memory. Lower WORK_SIZE, enable USE_CPU_OFFLOAD, or disable upscaling.",
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Pipeline failed")
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")
        elapsed = time.time() - start

    original_name = f"{job_id}_original.png"
    result_name = f"{job_id}_result.png"
    image.save(OUTPUTS_DIR / original_name)
    out["result"].save(OUTPUTS_DIR / result_name)

    logger.info("Job %s done in %.1fs (seed=%s)", job_id, elapsed, used_seed)
    return {
        "job_id": job_id,
        "original_url": f"/outputs/{original_name}",
        "result_url": f"/outputs/{result_name}",
        "seed": used_seed,
        "params": params,
        "tags": tags,
        "elapsed_sec": round(elapsed, 2),
    }


# Static mounts (registered AFTER the API routes so /api and /outputs win).
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
