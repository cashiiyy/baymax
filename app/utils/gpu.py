"""
BAYMAX AI – GPU Detection & Device Management
==============================================
Detects available GPU(s), reports VRAM, and provides a unified
device selector used across all ML modules.

Usage:
    from app.utils.gpu import gpu_info, get_device
    info = gpu_info()
    device = get_device()
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import List, Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GPUDevice:
    """Information about a single GPU device."""
    index: int
    name: str
    total_vram_mb: float
    free_vram_mb: float
    compute_capability: Optional[str] = None

    @property
    def total_vram_gb(self) -> float:
        return round(self.total_vram_mb / 1024, 2)

    @property
    def free_vram_gb(self) -> float:
        return round(self.free_vram_mb / 1024, 2)

    def supports_fp16(self) -> bool:
        """Check if the GPU supports FP16 (CUDA compute ≥ 5.3)."""
        if self.compute_capability is None:
            return False
        major, minor = map(int, self.compute_capability.split("."))
        return (major, minor) >= (5, 3)

    def supports_bf16(self) -> bool:
        """Check if the GPU supports BF16 (Ampere+, compute ≥ 8.0)."""
        if self.compute_capability is None:
            return False
        major, _ = map(int, self.compute_capability.split("."))
        return major >= 8

    def recommend_dtype(self) -> str:
        """Return recommended torch dtype based on GPU capability."""
        if self.supports_bf16():
            return "bfloat16"
        elif self.supports_fp16():
            return "float16"
        return "float32"


@dataclass
class SystemGPUInfo:
    """Aggregated GPU status for the host system."""
    cuda_available: bool
    cuda_version: Optional[str]
    devices: List[GPUDevice] = field(default_factory=list)
    python_platform: str = platform.platform()

    @property
    def has_gpu(self) -> bool:
        return self.cuda_available and len(self.devices) > 0

    @property
    def primary_device(self) -> Optional[GPUDevice]:
        """Return the GPU with most VRAM."""
        if not self.devices:
            return None
        return max(self.devices, key=lambda d: d.total_vram_mb)

    def recommend_quantization(self) -> str:
        """
        Suggest a quantization strategy for Qwen3-8B based on VRAM.
        - ≥ 16 GB → float16 (full precision)
        - 8–16 GB → 8-bit quantization
        - < 8 GB  → 4-bit quantization (default)
        """
        if not self.has_gpu:
            return "cpu"
        vram = self.primary_device.total_vram_gb  # type: ignore[union-attr]
        if vram >= 16:
            return "float16"
        elif vram >= 8:
            return "8bit"
        else:
            return "4bit"


def gpu_info() -> SystemGPUInfo:
    """
    Query the system for GPU information.

    Returns:
        SystemGPUInfo dataclass with full GPU details.
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        devices: List[GPUDevice] = []

        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                cap = f"{props.major}.{props.minor}"
                devices.append(
                    GPUDevice(
                        index=i,
                        name=props.name,
                        total_vram_mb=total_mem / (1024 ** 2),
                        free_vram_mb=free_mem / (1024 ** 2),
                        compute_capability=cap,
                    )
                )

        info = SystemGPUInfo(
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            devices=devices,
        )
        _log_gpu_info(info)
        return info

    except ImportError:
        log.warning("PyTorch not installed – GPU detection unavailable")
        return SystemGPUInfo(cuda_available=False, cuda_version=None)


def get_device(prefer_gpu: bool = True) -> str:
    """
    Return the best available compute device string ('cuda' or 'cpu').

    Args:
        prefer_gpu: If False, always return 'cpu'.

    Returns:
        'cuda' or 'cpu'
    """
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.debug("Selected compute device: {}", device)
        return device
    except ImportError:
        log.warning("PyTorch not available, defaulting to CPU")
        return "cpu"


def _log_gpu_info(info: SystemGPUInfo) -> None:
    """Log a friendly summary of detected GPU(s)."""
    if not info.has_gpu:
        log.warning(
            "No CUDA GPU detected – running on CPU. "
            "Performance will be significantly degraded."
        )
        return

    log.info(
        "GPU detected | CUDA {} | {} device(s)",
        info.cuda_version,
        len(info.devices),
    )
    for dev in info.devices:
        log.info(
            "  [GPU {}] {} | VRAM: {:.1f} GB total / {:.1f} GB free | "
            "Compute: {} | Recommended dtype: {}",
            dev.index,
            dev.name,
            dev.total_vram_gb,
            dev.free_vram_gb,
            dev.compute_capability,
            dev.recommend_dtype(),
        )
    quant = info.recommend_quantization()
    log.info("Recommended quantization strategy: {}", quant)
