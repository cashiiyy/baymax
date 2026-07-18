"""
BAYMAX AI – Qwen3 8B Instruct LLM Engine
==========================================
Loads and runs Qwen/Qwen3-8B-Instruct with HuggingFace Transformers.
Supports 4-bit quantization (bitsandbytes), streaming token generation,
and RAG-grounded response generation.

Usage:
    from app.llm.qwen_engine import QwenEngine
    engine = QwenEngine()
    response = engine.generate(chat_messages)
    print(response.text)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Result from a single LLM generation call.

    Attributes:
        text:           Generated response text.
        input_tokens:   Number of tokens in the prompt.
        output_tokens:  Number of tokens generated.
        elapsed_s:      Generation time in seconds.
        finish_reason:  Why generation stopped ('length', 'eos', etc.)
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    finish_reason: str = "eos"

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s == 0:
            return 0.0
        return round(self.output_tokens / self.elapsed_s, 2)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class QwenEngine:
    """
    HuggingFace Transformers-based inference engine for Qwen3-8B-Instruct.

    Features:
        - Lazy model loading (only loaded on first call)
        - 4-bit quantization with bitsandbytes for low-VRAM systems
        - Streaming token generation
        - HuggingFace chat template support

    Attributes:
        model_id:       HuggingFace model identifier.
        load_in_4bit:   Use 4-bit quantization (BnB).
        device_map:     HF device_map ('auto', 'cuda', 'cpu').
        max_new_tokens: Maximum tokens to generate per call.
        temperature:    Sampling temperature (0 = greedy).
        top_p:          Nucleus sampling threshold.
        repetition_penalty: Penalize repeated tokens.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        load_in_8bit: Optional[bool] = None,
        device_map: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        from config import settings

        self.model_id = model_id or settings.LLM_MODEL_ID
        self.load_in_4bit = load_in_4bit if load_in_4bit is not None else settings.LLM_LOAD_IN_4BIT
        self.load_in_8bit = load_in_8bit if load_in_8bit is not None else getattr(settings, 'LLM_LOAD_IN_8BIT', False)
        self.device_map = device_map or settings.LLM_DEVICE_MAP
        self.max_new_tokens = max_new_tokens or settings.LLM_MAX_NEW_TOKENS
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.top_p = top_p if top_p is not None else settings.LLM_TOP_P
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None
            else settings.LLM_REPETITION_PENALTY
        )

        self._model = None     # Lazy load
        self._tokenizer = None

        log.info(
            "QwenEngine configured | model={} 8bit={} 4bit={} device={}",
            self.model_id,
            self.load_in_8bit,
            self.load_in_4bit,
            self.device_map,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: List[dict],
        stream: bool = False,
    ) -> LLMResponse:
        """
        Generate a response from the LLM given a chat messages list.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
                      Must include a 'system' role message at minimum.
            stream:   If True, use streaming generation (text still collected).

        Returns:
            LLMResponse with generated text and statistics.
        """
        self._ensure_model_loaded()

        t_start = time.time()

        # Apply Qwen3 chat template
        text_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text_input], return_tensors="pt").to(
            self._model.device
        )
        input_len = inputs["input_ids"].shape[1]

        log.debug("Generating | input_tokens={}", input_len)

        if stream:
            generated_text, output_len = self._generate_streaming(inputs)
        else:
            generated_text, output_len = self._generate_standard(inputs)

        elapsed = time.time() - t_start
        response = LLMResponse(
            text=generated_text,
            input_tokens=input_len,
            output_tokens=output_len,
            elapsed_s=round(elapsed, 3),
        )

        log.info(
            "LLM generated | {}/{} tokens | {:.2f}s | {:.1f} tok/s",
            output_len,
            self.max_new_tokens,
            elapsed,
            response.tokens_per_second,
        )
        return response

    def generate_streaming(
        self,
        messages: List[dict],
    ) -> Generator[str, None, None]:
        """
        Streaming token generator — yields text chunks as they are produced.

        Args:
            messages: Chat messages list.

        Yields:
            Text chunks (one or more tokens at a time).
        """
        self._ensure_model_loaded()
        from transformers import TextIteratorStreamer
        import threading

        text_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text_input], return_tensors="pt").to(
            self._model.device
        )

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
        }

        # Run generation in a background thread
        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        for chunk in streamer:
            yield chunk

    def is_loaded(self) -> bool:
        """Return True if the model is loaded in memory."""
        return self._model is not None

    def unload(self) -> None:
        """Free model and tokenizer from memory."""
        import gc
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("QwenEngine unloaded")

    # ── Private Methods ───────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        log.info(
            "Loading Qwen3 model: {} (8bit={} 4bit={})",
            self.model_id,
            self.load_in_8bit,
            self.load_in_4bit,
        )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Quantization config — prefer 8-bit for RTX 5050 8GB
        bnb_config = None
        if self.load_in_8bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                log.info("Using 8-bit BnB quantization (optimal for RTX 5050 8GB VRAM)")
            except Exception as exc:
                log.warning("8-bit BnB config failed ({}). Falling back to 4-bit.", exc)
                bnb_config = None
                self.load_in_4bit = True

        if not self.load_in_8bit and self.load_in_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                log.info("Using 4-bit BnB quantization (NF4 + FP16)")
            except Exception as exc:
                log.warning("4-bit BnB config failed: {}", exc)
                bnb_config = None

        # Model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if bnb_config is None else None,
        )
        self._model.eval()

        log.info(
            "Qwen3 loaded | device={} | dtype={}",
            next(self._model.parameters()).device,
            next(self._model.parameters()).dtype,
        )

    def _generate_standard(
        self,
        inputs: dict,
    ) -> tuple[str, int]:
        """Standard (non-streaming) generation."""
        import torch

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.temperature > 0,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), len(new_tokens)

    def _generate_streaming(
        self,
        inputs: dict,
    ) -> tuple[str, int]:
        """Streaming generation that collects all output."""
        collected_text = ""
        for chunk in self.generate_streaming(
            messages=[]  # Already have encoded inputs
        ):
            collected_text += chunk
        # Count tokens
        tokens = self._tokenizer.encode(collected_text, add_special_tokens=False)
        return collected_text.strip(), len(tokens)
