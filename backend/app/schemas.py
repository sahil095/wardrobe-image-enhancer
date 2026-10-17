"""Pydantic response models for the API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EnhanceParams(BaseModel):
    strength: float
    steps: int
    guidance: float
    controlnet_scale: float
    ip_scale: float
    seed: int


class TagInfo(BaseModel):
    category: str = ""
    color: str = ""
    pattern: str = ""
    notes: str = ""


class EnhanceResponse(BaseModel):
    job_id: str
    original_url: str
    result_url: str
    seed: int
    params: EnhanceParams
    tags: TagInfo
    elapsed_sec: float


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda: bool
    models: dict
    settings: dict
    vram_gb: Optional[dict] = None
