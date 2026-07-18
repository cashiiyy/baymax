"""
BAYMAX AI – Health Endpoint
=============================
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.utils.gpu import gpu_info

router = APIRouter(tags=["System"])


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    vram_free_gb: float


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and GPU status."""
    from config import settings

    info = gpu_info()
    vram = 0.0
    if info.has_gpu and info.primary_device:
        vram = info.primary_device.free_vram_gb

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        gpu_available=info.has_gpu,
        vram_free_gb=vram,
    )
