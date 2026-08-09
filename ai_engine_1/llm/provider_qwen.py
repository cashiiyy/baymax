"""
AI Engine 1 – Qwen 2.5 Local Provider (4-bit Quantized)
=========================================================
High-performance local inference provider using HuggingFace Transformers
with BitsAndBytes NF4 quantization to run Qwen2.5-7B-Instruct entirely
within 8GB VRAM on an RTX 5050 Laptop GPU.

This replaces AirLLM layer-streaming with native VRAM-resident inference
for ultra-low latency text generation.

Usage:
    provider = QwenLocalProvider(model_id="Qwen/Qwen2.5-7B-Instruct")
    if provider.is_available():
        response = await provider.generate("What is diabetes?")

Configuration (via environment / ai_engine_1.config):
    LLM_PROVIDER=qwen_local
    QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
    QWEN_MAX_NEW_TOKENS=512
    QWEN_TEMPERATURE=0.3
"""

import asyncio
import gc
import json
import re
import time
import threading
from typing import Optional, List

from ai_engine_1.llm.provider_base import LLMProvider
from ai_engine_1.llm.llm_engine import LLMResponse

# Lazy import — backend.logging may not exist in all test contexts
try:
    from backend.logging.logger import get_logger
    logger = get_logger("ai-engine-1-qwen-local")
except Exception:
    import logging
    logger = logging.getLogger("ai-engine-1-qwen-local")


class QwenLocalProvider(LLMProvider):
    """4-bit quantized Qwen 2.5 local inference provider.

    Uses HuggingFace Transformers + BitsAndBytes NF4 quantization to fit
    Qwen2.5-7B-Instruct entirely in 8GB VRAM. Provides fast, native
    inference without layer-streaming overhead.

    Features:
        - 4-bit NF4 quantization (~4GB VRAM for 7B model)
        - Lazy model initialization (loaded only on first call)
        - Singleton: model persists across requests (never reloaded)
        - Resource cleanup via unload()
        - Qwen 2.5 ChatML prompt formatting
        - Structured output parsing with safe fallback
        - Thread-safe model loading with lock
    """

    _instance_lock = threading.Lock()

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self._model = None
        self._tokenizer = None
        self._available: Optional[bool] = None  # None = not yet checked
        self._load_error: Optional[str] = None

        logger.info(
            f"[LLM] QwenLocalProvider configured | model={self.model_id} "
            f"max_tokens={self.max_new_tokens} temp={self.temperature} "
            f"quantization=4bit-NF4"
        )

    # ── LLMProvider Interface ─────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Generate a response using Qwen 2.5 (4-bit quantized).

        Runs the blocking inference in a thread pool to keep the
        async event loop responsive.
        """
        if not self.is_available():
            raise RuntimeError(
                f"QwenLocalProvider is not available: {self._load_error or 'unknown error'}"
            )

        return await asyncio.to_thread(
            self._generate_sync,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def is_available(self) -> bool:
        """Check if torch, transformers, and bitsandbytes are importable."""
        if self._available is not None:
            return self._available

        try:
            import torch
            if not torch.cuda.is_available():
                self._available = False
                self._load_error = "CUDA not available — 4-bit quantization requires GPU"
                logger.warning(f"[LLM] QwenLocalProvider unavailable: {self._load_error}")
                return False

            import transformers  # noqa: F401
            import bitsandbytes  # noqa: F401
            self._available = True
        except ImportError as e:
            self._available = False
            self._load_error = f"Missing dependency: {e}"
            logger.warning(f"[LLM] QwenLocalProvider unavailable: {self._load_error}")

        return self._available

    def provider_name(self) -> str:
        return f"QwenLocal-4bit/{self.model_id}"

    def unload(self) -> None:
        """Release model and tokenizer from GPU/RAM."""
        if self._model is not None:
            logger.info("[LLM] Unloading Qwen 2.5 local model...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("[LLM] Qwen 2.5 local model unloaded successfully")

    # ── Private Implementation ────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model with 4-bit quantization on first use.

        Thread-safe: uses a lock to prevent concurrent loading.
        """
        if self._model is not None:
            return

        with self._instance_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            logger.info(f"[LLM] Loading model: {self.model_id} (4-bit NF4 quantization)...")
            load_start = time.time()

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                # Tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )

                # 4-bit NF4 quantization config — optimal for 8GB VRAM
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                # Load model
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._model.eval()

                load_elapsed = time.time() - load_start

                # Log device and memory info
                device = next(self._model.parameters()).device
                dtype = next(self._model.parameters()).dtype
                vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

                logger.info(
                    f"[LLM] Model loaded successfully | "
                    f"model={self.model_id} | device={device} | dtype={dtype} | "
                    f"load_time={load_elapsed:.1f}s | "
                    f"VRAM allocated={vram_allocated:.2f}GB reserved={vram_reserved:.2f}GB"
                )

            except Exception as e:
                self._model = None
                self._tokenizer = None
                self._available = False
                self._load_error = f"Model load failed: {e}"
                logger.error(f"[LLM] Failed to load Qwen 2.5 model: {e}")
                raise RuntimeError(self._load_error)

    def _generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Synchronous generation (called from thread pool)."""
        self._ensure_model_loaded()

        import torch

        # Build chat messages and apply Qwen 2.5 chat template
        messages = self._build_messages(prompt, system_prompt)
        text_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            [text_input],
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]

        logger.info(f"[LLM] Inference started | input_tokens={input_len}")
        start_time = time.time()

        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=self.top_p if temperature > 0 else None,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=temperature > 0,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Slice off prompt tokens
            new_tokens = output_ids[0][input_len:]
            generated_text = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            elapsed = time.time() - start_time
            output_token_count = len(new_tokens)
            tokens_per_sec = output_token_count / elapsed if elapsed > 0 else 0

            logger.info(
                f"[LLM] Inference completed | "
                f"latency={elapsed:.2f}s | "
                f"tokens={output_token_count} | "
                f"tok/s={tokens_per_sec:.1f}"
            )

            return LLMResponse(
                text=generated_text,
                model_used=self.provider_name(),
                tokens_used=output_token_count,
                latency_ms=round(elapsed * 1000, 2),
                fallback_triggered=False,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[LLM] Inference failed after {elapsed:.2f}s: {e}")
            raise

    @staticmethod
    def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> List[dict]:
        """Build chat message list for Qwen 2.5 Instruct."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ── Structured Output Helpers ─────────────────────────────────────────────

    @staticmethod
    def parse_structured_response(raw_text: str) -> Optional[dict]:
        """Attempt to extract a JSON object from the model's raw output.

        Returns None if no valid JSON is found (caller should use raw text).
        """
        # Try direct JSON parse
        try:
            return json.loads(raw_text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding first { ... } block
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return None
