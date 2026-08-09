import os
import time
import asyncio
import json
import httpx
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from ai_engine_1.config import settings
from backend.logging.logger import get_logger

logger = get_logger("ai-engine-1-llm")


class LLMResponse(BaseModel):
    text: str
    model_used: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    fallback_triggered: bool = False


class ProductionLLMEngine:
    """Production LLM Engine with local Qwen 2.5 primary, OpenRouter API, and Gemini fallbacks."""

    def __init__(self, config=None):
        if config is None:
            config = settings
        self.config = config
        self.primary_model = config.primary_llm_model
        self.fallback_model = config.fallback_llm_model
        self.openrouter_api_key = config.openrouter_api_key
        self.openrouter_url = f"{config.openrouter_base_url}/chat/completions"
        self.ollama_url = f"{config.ollama_base_url}/api/generate"

        # Local Qwen 2.5 provider (lazy-initialized, only when configured)
        self._qwen_local_provider = None
        self._llm_provider_type = getattr(config, 'llm_provider', 'openrouter')
        if self._llm_provider_type == 'qwen_local':
            logger.info("[LLM] Provider selected: QwenLocal (4-bit quantized)")
        else:
            logger.info(f"[LLM] Provider selected: {self._llm_provider_type}")

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        json_mode: bool = False,
        max_tokens: int = 1500
    ) -> LLMResponse:
        start_time = time.time()

        # Context length validation: >4096 tokens is routed to gateway failover immediately
        is_context_overflow = False
        try:
            # Quick estimation of tokens based on character count / 4
            approx_tokens = len(prompt) // 4
            if approx_tokens > 4096:
                is_context_overflow = True
                logger.warning(f"[LLM] Context window overflow detected (approx {approx_tokens} tokens). Directing to gateway backup.")
        except Exception:
            pass

        # Try Local Qwen 2.5 (4-bit quantized) if configured as primary and context length is okay
        if self._llm_provider_type == 'qwen_local' and not is_context_overflow:
            try:
                provider = self._get_qwen_local_provider()
                if provider.is_available():
                    # We wrap the local generation to execute failover on timeout or OOM/Cuda exceptions
                    # using an internal timeout or try/except block.
                    res = await asyncio.wait_for(
                        provider.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        timeout=30.0 # 30s timeout trigger metric
                    )
                    return res
            except Exception as err:
                logger.warning(f"[LLM] Local Qwen provider failed (possible OOM/Timeout/Error), falling back to OmniRoute gateway: {err}")

        # 1. Try Backup: OmniRoute API gateway (or OpenRouter if configured as backup)
        if self.openrouter_api_key:
            try:
                res = await self._call_openrouter(prompt, system_prompt, temperature, json_mode, max_tokens)
                latency = (time.time() - start_time) * 1000
                return LLMResponse(
                    text=res,
                    model_used=f"OmniRoute/{self.primary_model}",
                    latency_ms=round(latency, 2),
                    fallback_triggered=True
                )
            except Exception as err:
                logger.warning(f"OmniRoute gateway API call failed, triggering secondary fallback: {err}")

        # 2. Try Fallback: Gemini API
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDJxwOTPaNZw-p3glKSfYZyhxa5w9cxtzE")
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel("gemini-flash-latest")
            full_prompt = f"{system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
            
            res = await model.generate_content_async(full_prompt)
            response_text = res.text
            
            latency = (time.time() - start_time) * 1000
            return LLMResponse(
                text=response_text,
                model_used="Gemini/gemini-flash-latest",
                latency_ms=round(latency, 2),
                fallback_triggered=True
            )
        except Exception as err:
            logger.warning(f"Gemini fallback failed: {err}")

        # 3. Safe offline fallback — dynamic context-aware response
        latency = (time.time() - start_time) * 1000
        fallback_text = self._build_offline_fallback(prompt)
        return LLMResponse(
            text=fallback_text,
            model_used="OfflineFallback",
            latency_ms=round(latency, 2),
            fallback_triggered=True
        )

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.generate_async(prompt, **kwargs))
            else:
                return loop.run_until_complete(self.generate_async(prompt, **kwargs))
        except Exception:
            return asyncio.run(self.generate_async(prompt, **kwargs))

    async def _call_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        json_mode: bool,
        max_tokens: int
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://baymax.local",
            "X-Title": "BAYMAX Medical Intelligence Engine",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.primary_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.openrouter_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Local gateways like OmniRoute might stream responses or return SSE payloads (data: chunks)
            content_type = response.headers.get("content-type", "")
            response_text = response.text.strip()
            
            if "text/event-stream" in content_type or response_text.startswith("data:"):
                # Parse Server-Sent Events (SSE) data stream chunks
                full_text = ""
                for line in response_text.splitlines():
                    line = line.strip()
                    if line.startswith("data:") and not line.endswith("[DONE]"):
                        try:
                            json_str = line[5:].strip()
                            chunk_data = json.loads(json_str)
                            # Check standard delta choices
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "content" in delta:
                                    full_text += delta["content"]
                        except Exception:
                            pass
                if full_text:
                    return full_text.strip()
            
            data = response.json()
            # If standard JSON response
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            return response_text

    async def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {
            "model": self.fallback_model,
            "prompt": full_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

    def _get_qwen_local_provider(self):
        """Lazy-initialize the local Qwen 2.5 provider singleton."""
        if self._qwen_local_provider is None:
            from ai_engine_1.llm.provider_qwen import QwenLocalProvider
            self._qwen_local_provider = QwenLocalProvider(
                model_id=getattr(self.config, 'qwen_model', 'Qwen/Qwen2.5-7B-Instruct'),
                max_new_tokens=getattr(self.config, 'qwen_max_new_tokens', 512),
                temperature=getattr(self.config, 'qwen_temperature', 0.3),
                top_p=getattr(self.config, 'qwen_top_p', 0.9),
                repetition_penalty=getattr(self.config, 'qwen_repetition_penalty', 1.1),
            )
        return self._qwen_local_provider

    @property
    def active_provider_name(self) -> str:
        """Return the name of the currently configured primary provider."""
        if self._llm_provider_type == 'qwen_local':
            provider = self._get_qwen_local_provider()
            return provider.provider_name()
        return f"OmniRoute/{self.primary_model}"

    def _build_offline_fallback(self, prompt: str) -> str:
        """Builds a context-aware offline response when all LLM backends are unavailable.
        
        Extracts the actual user query from the assembled prompt and matches keywords
        to provide relevant offline guidance.
        """
        import re

        # Extract the user query from the assembled prompt (avoids false-positive matches in template text)
        query_match = re.search(r"User Query:\s*(.+?)(?:\n\n|Patient Context|$)", prompt, re.IGNORECASE | re.DOTALL)
        if query_match:
            user_query_raw = query_match.group(1).strip()
        else:
            user_query_raw = prompt

        p = user_query_raw.lower()

        def has_word(text: str, *words: str) -> bool:
            """Word-boundary aware keyword matching."""
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', text):
                    return True
            return False

        if has_word(p, "burn", "scald", "burns"):
            return (
                "### First Aid for Burns\n\n"
                "1. Cool under running water for 10-20 minutes. Never use ice or butter.\n"
                "2. Cover loosely with a sterile, non-stick bandage.\n"
                "3. Take paracetamol or ibuprofen for pain relief. Do NOT pop blisters.\n\n"
                "EMERGENCY: Seek emergency care for large burns, burns on face/hands/genitals, or chemical/electrical burns.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "cpr", "resuscitation", "cardiac arrest"):
            return (
                "### CPR Protocol\n\n"
                "1. Check surroundings and responsiveness. Call 108/911 immediately.\n"
                "2. Place heel of hand on center of chest. Push hard and fast (100-120 compressions/min).\n"
                "3. Allow full chest recoil between compressions. Continue until help arrives.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "dengue"):
            return (
                "### Dengue Fever Guidance\n\n"
                "Key symptoms: high fever, severe headache, joint pain, rash.\n"
                "Care: rest, oral rehydration, paracetamol (avoid NSAIDs/aspirin).\n"
                "Seek hospital care if you notice bleeding, persistent vomiting, or severe fatigue.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "hypertension") or "blood pressure" in p:
            return (
                "### Hypertension Management\n\n"
                "1. Reduce sodium intake (<2000mg/day). Follow DASH diet.\n"
                "2. Exercise 30 minutes daily. Manage stress with meditation or breathing.\n"
                "3. Monitor BP regularly. Take prescribed medications as directed.\n\n"
                "BP >180/120 with symptoms = hypertensive crisis. Seek emergency care.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "snake", "snakebite", "venom"):
            return (
                "### Snake Bite Emergency\n\n"
                "1. Keep victim calm and still. Do NOT cut, suck, or tourniquet the wound.\n"
                "2. Immobilize the bitten limb at or below heart level.\n"
                "3. Remove rings/watches before swelling. Wash wound gently.\n\n"
                "EMERGENCY: Transport immediately to hospital with antivenom capability.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "asthma", "wheezing", "inhaler") or "difficulty breathing" in p:
            return (
                "### Breathing Difficulty / Asthma\n\n"
                "1. Sit upright. Use rescue inhaler (1-2 puffs Albuterol).\n"
                "2. Breathe slowly. Move away from dust, smoke, or cold air.\n\n"
                "EMERGENCY: Call emergency services if lips turn blue or inhaler gives no relief.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "fever", "temperature"):
            return (
                "### Fever Management\n\n"
                "1. Stay hydrated with water, broth, or electrolyte drinks.\n"
                "2. Rest, wear light clothing, and keep room temperature comfortable.\n"
                "3. Paracetamol or ibuprofen can reduce fever.\n\n"
                "Consult a doctor if fever exceeds 39.4 degrees C (103 F) or lasts more than 3 days.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "diabetes", "blood sugar", "glucose", "insulin"):
            return (
                "### Diabetes Overview & Management\n\n"
                "**Type 1 Diabetes:** Autoimmune condition where the body produces no insulin.\n"
                "**Type 2 Diabetes:** Body doesn't use insulin effectively. Most common form.\n\n"
                "**Common Symptoms:** Increased thirst, frequent urination, fatigue, blurred vision, slow healing wounds.\n\n"
                "**Management:**\n"
                "1. Monitor blood glucose levels regularly.\n"
                "2. Follow a low-glycemic, balanced diet (limit sugars and refined carbs).\n"
                "3. Exercise regularly to improve insulin sensitivity.\n"
                "4. Take prescribed medications (metformin, insulin, etc.) as directed.\n"
                "5. Attend regular check-ups for HbA1c, kidney, eye, and foot health.\n\n"
                "Consult your endocrinologist or physician for a personalized diabetes management plan.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "headache", "migraine"):
            return (
                "### Headache & Migraine Relief\n\n"
                "1. Rest in a quiet, dark room. Apply cold or warm compress to forehead/neck.\n"
                "2. Stay hydrated. Dehydration is a common trigger.\n"
                "3. Over-the-counter pain relievers (ibuprofen, paracetamol) can help mild headaches.\n"
                "4. For migraines: avoid triggers (bright lights, strong smells, stress).\n\n"
                "EMERGENCY: Seek immediate care if headache is sudden/severe ('thunderclap'), accompanied by fever, stiff neck, vision loss, or confusion.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )
        if has_word(p, "allergy", "allergic", "anaphylaxis", "hives"):
            return (
                "### Allergy & Anaphylaxis Response\n\n"
                "**Mild Allergy:** Antihistamines (cetirizine, loratadine), avoid allergen.\n"
                "**Severe Anaphylaxis Symptoms:** Throat swelling, difficulty breathing, sudden BP drop.\n\n"
                "EMERGENCY for Anaphylaxis:\n"
                "1. Administer epinephrine auto-injector (EpiPen) immediately.\n"
                "2. Call 108/911. Lay flat with legs elevated.\n"
                "3. Second EpiPen dose may be needed after 5-15 minutes.\n\n"
                "*Note: LLM backends are currently offline. This is a static fallback response.*"
            )

        # Generic medical fallback
        clean_query = re.sub(r'[^a-zA-Z0-9\s,.-]', '', user_query_raw[:150]).strip()

        return (
            f"### Health Information: {clean_query}\n\n"
            f"Regarding your query: **{clean_query}**\n\n"
            "1. Monitor symptoms carefully, noting onset, duration, and severity.\n"
            "2. Ensure adequate rest, hydration, and a balanced diet.\n"
            "3. Avoid self-medicating with prescription drugs without professional guidance.\n\n"
            "For personalized medical advice, please consult a qualified healthcare professional.\n\n"
            "Note: The AI inference backend (OpenRouter / Ollama) is currently offline. "
            "To enable full dynamic AI responses, add your OPENROUTER_API_KEY to the .env file "
            "or run Ollama locally (port 11434) with: ollama pull qwen:1.5b"
        )


llm_engine = ProductionLLMEngine()
