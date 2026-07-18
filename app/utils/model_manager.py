"""
BAYMAX AI – Model Directory Manager
=====================================
Centralizes model path resolution, cache validation, and download helpers.
Ensures all modules load models from a consistent, configurable location.

Usage:
    from app.utils.model_manager import ModelManager
    mgr = ModelManager()
    path = mgr.get_model_path("whisper", "medium")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


class ModelManager:
    """
    Manages model file locations, download status, and metadata.

    All models are stored under the configured `models_dir` with
    sub-directories per model type.
    """

    MODEL_REGISTRY: dict[str, dict[str, str]] = {
        "whisper": {
            "tiny":     "Systran/faster-whisper-tiny",
            "base":     "Systran/faster-whisper-base",
            "small":    "Systran/faster-whisper-small",
            "medium":   "Systran/faster-whisper-medium",
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3",
        },
        "llm": {
            "qwen3-8b": "Qwen/Qwen3-8B-Instruct",
        },
        "embedding": {
            "minilm":    "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet":     "sentence-transformers/all-mpnet-base-v2",
        },
        "tts": {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
        },
    }

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        from config import settings

        self.models_dir = models_dir or settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.models_dir / ".registry.json"
        self._registry: dict[str, dict] = self._load_registry()
        log.info("ModelManager initialized | models_dir={}", self.models_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_model_path(self, model_type: str, variant: str) -> Path:
        """
        Return the local filesystem path for a registered model.

        Args:
            model_type: Category such as 'whisper', 'llm', 'embedding', 'tts'.
            variant:    Model variant key, e.g. 'medium', 'qwen3-8b'.

        Returns:
            Resolved Path to the model directory.
        """
        model_dir = self.models_dir / model_type / variant
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_hf_model_id(self, model_type: str, variant: str) -> str:
        """
        Return the HuggingFace Hub model ID for a given type/variant.

        Args:
            model_type: Category key.
            variant:    Variant key.

        Returns:
            HuggingFace model ID string.

        Raises:
            KeyError: If the type/variant is not in the registry.
        """
        try:
            return self.MODEL_REGISTRY[model_type][variant]
        except KeyError as exc:
            log.error(
                "Model not found in registry | type={} variant={}",
                model_type,
                variant,
            )
            raise KeyError(
                f"Unknown model: type='{model_type}', variant='{variant}'"
            ) from exc

    def is_downloaded(self, model_type: str, variant: str) -> bool:
        """
        Check whether a model has been downloaded to disk.

        Args:
            model_type: Category key.
            variant:    Variant key.

        Returns:
            True if the model directory is non-empty.
        """
        model_path = self.get_model_path(model_type, variant)
        files = list(model_path.glob("**/*"))
        downloaded = any(f.is_file() for f in files)
        log.debug(
            "Model download check | type={} variant={} downloaded={}",
            model_type,
            variant,
            downloaded,
        )
        return downloaded

    def register_model(
        self,
        model_type: str,
        variant: str,
        metadata: dict,
    ) -> None:
        """
        Record metadata about a downloaded model in the registry JSON.

        Args:
            model_type: Category key.
            variant:    Variant key.
            metadata:   Arbitrary dict (e.g., version, checksum, timestamp).
        """
        key = f"{model_type}/{variant}"
        self._registry[key] = metadata
        self._save_registry()
        log.info("Model registered | key={}", key)

    def get_registry_entry(self, model_type: str, variant: str) -> Optional[dict]:
        """
        Retrieve a registry entry for a model.

        Args:
            model_type: Category key.
            variant:    Variant key.

        Returns:
            Registry dict or None if not registered.
        """
        return self._registry.get(f"{model_type}/{variant}")

    def compute_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Compute a hex-digest checksum for a file (for integrity checks).

        Args:
            file_path:  Absolute path to the file.
            algorithm:  Hash algorithm name (default: sha256).

        Returns:
            Hex digest string.
        """
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def list_downloaded_models(self) -> list[dict]:
        """
        Return a list of all models currently downloaded to disk.

        Returns:
            List of dicts with keys: type, variant, path, size_mb.
        """
        result = []
        for model_type in self.MODEL_REGISTRY:
            for variant in self.MODEL_REGISTRY[model_type]:
                path = self.get_model_path(model_type, variant)
                if self.is_downloaded(model_type, variant):
                    size = sum(
                        f.stat().st_size
                        for f in path.rglob("*")
                        if f.is_file()
                    )
                    result.append(
                        {
                            "type": model_type,
                            "variant": variant,
                            "path": str(path),
                            "size_mb": round(size / (1024 ** 2), 2),
                        }
                    )
        return result

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text())
            except json.JSONDecodeError:
                log.warning("Corrupted model registry – starting fresh")
        return {}

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2), encoding="utf-8"
        )
