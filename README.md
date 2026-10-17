# Wardrobe Image Enhancer

Local, GPU-powered web app that takes a photo of a clothing item (jeans, shirt,
pullover, …) plus optional tags, and returns the **same garment** cleaned up:
white background, de-wrinkled / straightened, and upscaled. The UI shows a
side-by-side comparison and a **Rerun** button.

Everything runs **locally** on your NVIDIA GPU. Nothing is uploaded anywhere.

---

## How it works (hybrid pipeline)

```
Upload + tags
   -> Stage 1  Background removal (rembg)          -> pure white background
   -> Stage 2  De-wrinkle / straighten             -> SD img2img + ControlNet (shape)
                                                        + IP-Adapter (appearance)
   -> Stage 3  Upscale (Real-ESRGAN, tiled)        -> crisp final image
```

- **Stage 1** is deterministic and preserves the garment exactly.
- **Stage 2** is the only generative step. `strength` controls the trade-off:
  low = keep the garment exact (fewer wrinkles removed), high = smoother fabric
  (more risk of altering prints/logos). Tags (`category`, `color`, `pattern`,
  `notes`) are turned into the prompt, and IP-Adapter conditions on your actual
  photo so color/pattern are kept.
- **Stage 3** sharpens and upscales with low VRAM via tiling (falls back to
  Lanczos if the weights aren't downloaded).

Default base model is **SD 1.5** (`Lykon/dreamshaper-8`), which fits comfortably
in 8 GB VRAM. An **SDXL** path exists via config for higher quality (slower,
auto CPU-offload).

---

## Requirements

- Windows + **NVIDIA GPU** (tested target: RTX 4060 8 GB, 16 GB RAM).
- **NVIDIA driver** installed. You do **not** need to install the CUDA Toolkit
  separately — the PyTorch CUDA wheels bundle the CUDA runtime. (`nvcc` is only
  needed if you compile CUDA code, which this project does not.)
- Python 3.11 (via Anaconda recommended).

Check your driver / GPU anytime with:

```bat
nvidia-smi
```

---

## Quick start (scripts)

From an **Anaconda Prompt** in this folder:

```bat
setup.bat   :: creates the "wardrobe" env + installs everything + checks GPU
run.bat     :: starts the server at http://127.0.0.1:8000
```

Prefer manual steps? See below.

## Setup (Anaconda Prompt)

Open **Anaconda Prompt** (not PowerShell) and run:

```bat
:: 1. Create and activate an environment
conda create -n wardrobe python=3.11 -y
conda activate wardrobe

:: 2. Go to the project
cd /d C:\Users\Sahil\Projects\wardrobe-image-enhancer

:: 3. Install PyTorch with CUDA 12.4 FIRST (important - gets the GPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: 4. Install the rest
pip install -r backend\requirements.txt

:: 5. Verify everything (should say READY and show your GPU)
python scripts\check_env.py

:: 6. (Optional) pre-download model weights
python scripts\download_models.py
```

> If `check_env.py` says CUDA is not available, your driver may be too old or
> the CPU build of torch got installed. Re-run step 3 and confirm with
> `nvidia-smi`.

---

## Run

```bat
conda activate wardrobe
cd /d C:\Users\Sahil\Projects\wardrobe-image-enhancer\backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open: **http://127.0.0.1:8000**

The first request (or startup warmup) downloads the models — this can take a few
minutes and several GB. Subsequent runs are fast because models stay in VRAM.

To disable the startup warmup (start instantly, load on first request):

```bat
set WARMUP_ON_STARTUP=false
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Using the app

1. Click / drag a clothing image into the upload box.
2. (Optional) fill in **Category / Color / Pattern / Notes** — these improve the
   result.
3. Click **Enhance**. Compare **Original** vs **Enhanced** side by side.
4. Not happy? Click **Rerun (new seed)** for a fresh variation, or tweak
   **Advanced settings**.
5. Click **Download enhanced** to save the result.

---

## Tuning (Advanced settings / env vars)

| Control        | Env var             | Default | Meaning                                        |
| -------------- | ------------------- | ------- | ---------------------------------------------- |
| Strength       | `STRENGTH`          | `0.35`  | Higher removes more wrinkles, less fidelity    |
| Steps          | `STEPS`             | `28`    | More steps = slightly better, slower           |
| Guidance       | `GUIDANCE`          | `6.5`   | Prompt adherence                               |
| ControlNet     | `CONTROLNET_SCALE`  | `0.55`  | Shape lock strength                            |
| IP-Adapter     | `IP_ADAPTER_SCALE`  | `0.6`   | Appearance (color/pattern) lock strength       |
| Work size      | `WORK_SIZE`         | `768`   | Diffusion resolution (lower = less VRAM)       |
| Base model     | `BASE_MODEL_TYPE`   | `sd15`  | `sd15` or `sdxl`                               |
| CPU offload    | `USE_CPU_OFFLOAD`   | `false` | Force offload to save VRAM (auto-on for sdxl)  |
| Upscaling      | `UPSCALE_ENABLED`   | `true`  | Toggle Real-ESRGAN                             |
| Upscale cap    | `UPSCALE_TARGET_MAX`| `1536`  | Max long side of final image                   |

**Out of memory?** Lower `WORK_SIZE` (e.g. 640), set `USE_CPU_OFFLOAD=true`, or
set `UPSCALE_ENABLED=false`.

---

## Optimized inference (built in)

- Models loaded once and cached in a `ModelManager` (startup warmup).
- fp16, `torch.inference_mode()`, SDPA attention (memory-efficient, and
  IP-Adapter compatible), VAE tiling + slicing, `channels_last`.
- Optional model CPU offload for the SDXL path.
- Single-flight asyncio GPU lock + thread-pool execution so the server stays
  responsive and only one job uses the GPU at a time.
- Fast DPM++ 2M Karras scheduler (~28 steps).
- Tiled Real-ESRGAN to bound upscaler VRAM.

---

## Project layout

```
wardrobe-image-enhancer/
  backend/
    app/
      main.py            FastAPI app, routes, lifespan
      config.py          settings (env-overridable)
      models.py          ModelManager (load/cache/optimize)
      schemas.py         response models
      pipeline/
        background.py    Stage 1: rembg -> white bg
        enhance.py       Stage 2: SD img2img + ControlNet + IP-Adapter
        upscale.py       Stage 3: Real-ESRGAN (tiled) + Lanczos fallback
        orchestrator.py  chains the stages
    requirements.txt
  frontend/              index.html, app.js, style.css
  scripts/
    check_env.py         verify CUDA / libs
    download_models.py   pre-fetch weights
  models/                downloaded weights (gitignored)
  outputs/               generated images (gitignored)
```

---

## API

`POST /api/enhance` (multipart form): `file`, `category`, `color`, `pattern`,
`notes`, `strength`, `steps`, `guidance`, `controlnet_scale`, `ip_scale`,
`seed` (-1 = random). Returns JSON with `original_url`, `result_url`, `seed`,
`params`, `elapsed_sec`.

`GET /api/health` — device, CUDA, VRAM, loaded models, settings.

This endpoint can be called by your separate wardrobe-suggestions app to batch
enhance its catalog images.
