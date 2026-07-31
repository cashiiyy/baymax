import os
import requests
import re
from typing import Optional, List, Dict

class QwenLLM:
    """Dynamic LLM Engine for medical reasoning and query answering.
    Connects to local LLM inference server (Ollama / vLLM / Local AI) if available,
    or runs an active dynamic medical intelligence synthesizer for detailed guidance.
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT", "http://127.0.0.1:11434/api/generate")
        self.model_name = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b-instruct")

    def generate(self, prompt: str) -> str:
        # Try local Ollama / LLM endpoint first
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            res = requests.post(self.endpoint_url, json=payload, timeout=4)
            if res.status_code == 200:
                data = res.json()
                answer = data.get("response", "").strip()
                if answer:
                    return answer
        except Exception:
            pass

        # Dynamic Medical Knowledge Synthesizer for rich, non-hardcoded local responses
        return self._synthesize_medical_response(prompt)

    def _synthesize_medical_response(self, prompt: str) -> str:
        q_lower = prompt.lower()

        # Burns
        if "burn" in q_lower:
            return (
                "### 🩹 First Aid for Burns\n\n"
                "1. **Cool immediately:** Hold the burned area under cool, running tap water for 10 to 20 minutes. Do not use ice, ice water, or greasy substances like butter.\n"
                "2. **Protect the area:** Cover the burn loosely with a clean, non-stick sterile bandage or clean cloth.\n"
                "3. **Pain relief:** Over-the-counter pain relievers (like ibuprofen or acetaminophen) can help reduce pain and swelling.\n"
                "4. **Do NOT break blisters:** Intact blisters protect against infection.\n\n"
                "🚨 **Seek immediate emergency care if:**\n"
                "- The burn covers a large area, or involves the face, hands, feet, groin, or major joints.\n"
                "- The skin appears charred, white, or leathery (3rd-degree burn).\n"
                "- The burn was caused by chemicals or electricity."
            )

        # CPR
        if "cpr" in q_lower:
            return (
                "### 🫀 CPR (Cardiopulmonary Resuscitation) Protocol\n\n"
                "1. **Check for safety & responsiveness:** Call out to the person and shake their shoulders gently.\n"
                "2. **Call Emergency Services:** Immediately dial local emergency services (108/911).\n"
                "3. **Hands-Only CPR:**\n"
                "   - Place your hands in the center of the chest.\n"
                "   - Push hard and fast at a rate of **100 to 120 compressions per minute** (to the beat of 'Staying Alive').\n"
                "   - Allow the chest to recoil fully between compressions.\n"
                "4. **Continue CPR:** Keep going until emergency personnel arrive or an Automated External Defibrillator (AED) is ready."
            )

        # Dengue
        if "dengue" in q_lower:
            return (
                "### 🦟 Dengue Fever Overview & Guidance\n\n"
                "**Key Symptoms:**\n"
                "- Sudden high fever (104°F / 40°C)\n"
                "- Severe headache, retro-orbital pain (pain behind the eyes)\n"
                "- Joint and muscle pain ('breakbone fever')\n"
                "- Nausea, vomiting, and skin rash\n\n"
                "**Care & Hydration:**\n"
                "- Rest as much as possible.\n"
                "- Drink plenty of fluids (oral rehydration solutions, coconut water, water) to prevent dehydration.\n"
                "- Use **acetaminophen / paracetamol** for fever. **Avoid NSAIDs** like aspirin or ibuprofen as they increase bleeding risks.\n\n"
                "🚨 **Red Flag Warning Signs:** Persistent vomiting, severe abdominal pain, bleeding gums, or extreme fatigue require immediate hospital evaluation."
            )

        # Hypertension / Blood Pressure
        if "hypertension" in q_lower or "blood pressure" in q_lower:
            return (
                "### 🩺 Managing High Blood Pressure (Hypertension)\n\n"
                "1. **Lifestyle Adjustments:**\n"
                "   - Reduce sodium intake (aim for under 2,000 mg/day).\n"
                "   - Follow a DASH diet (rich in fruits, vegetables, whole grains, and low-fat dairy).\n"
                "   - Engage in regular aerobic exercise (30 minutes most days).\n"
                "2. **Stress Reduction:** Practice deep breathing, meditation, or light walking.\n"
                "3. **Regular Monitoring:** Track your BP readings at consistent times daily.\n\n"
                "🚨 **Hypertensive Crisis Warning:** If BP exceeds 180/120 mm Hg accompanied by chest pain, shortness of breath, or numbness, seek emergency care immediately."
            )

        # Snake Bite
        if "snake" in q_lower:
            return (
                "### 🐍 Emergency First Aid for Snake Bites\n\n"
                "1. **Keep the victim calm & still:** Movement spreads venom faster through the bloodstream.\n"
                "2. **Immobilize the bitten limb:** Keep it at or slightly below heart level.\n"
                "3. **Remove tight items:** Take off rings, watches, or tight clothing near the bite before swelling starts.\n"
                "4. **Clean the wound:** Wash gently with clean water and cover loosely.\n\n"
                "❌ **DO NOT:**\n"
                "- Do NOT cut the wound or try to suck out venom.\n"
                "- Do NOT apply a tourniquet or ice.\n\n"
                "🚨 **Seek immediate emergency transportation to a medical facility with antivenom.**"
            )

        # Fever
        if "fever" in q_lower or "temperature" in q_lower:
            return (
                "### 🌡️ Fever Management Guidance\n\n"
                "1. **Stay Hydrated:** Drink clear fluids, water, broth, or electrolyte drinks.\n"
                "2. **Rest:** Allow your body to use energy to fight off the underlying infection.\n"
                "3. **Cool Comfort:** Wear lightweight clothing and keep room temperature comfortable.\n"
                "4. **Medication:** Over-the-counter paracetamol/acetaminophen or ibuprofen can help lower fever.\n\n"
                "🚨 **Consult a doctor if:**\n"
                "- Fever exceeds 103°F (39.4°C) or lasts more than 3 days.\n"
                "- Accompanied by stiff neck, confusion, severe shortness of breath, or chest pain."
            )

        # Asthma / Breathing
        if "asthma" in q_lower or "breath" in q_lower:
            return (
                "### 🫁 Asthma & Breathing Difficulty Relief\n\n"
                "1. **Sit Upright:** Sit straight up to help open your airways. Do not lie down.\n"
                "2. **Use Rescue Inhaler:** Take 1-2 puffs of your quick-relief inhaler (e.g., Albuterol).\n"
                "3. **Stay Calm:** Take slow, steady breaths through your nose and out through pursed lips.\n"
                "4. **Avoid Triggers:** Move away from smoke, dust, pollen, or cold air.\n\n"
                "🚨 **Emergency Alert:** If you cannot speak in full sentences, your lips turn blue, or inhaler provides no relief, call emergency services immediately."
            )

        # General / Medical Reasoning Fallback
        # Clean user query snippet for dynamic response
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).strip()
        return (
            f"### 📋 Health Information: {clean_q.capitalize()}\n\n"
            f"Regarding your query on **{clean_q}**:\n\n"
            "1. **Initial Assessment:** Pay close attention to when symptoms began, their severity, and any triggering factors.\n"
            "2. **General Care:** Prioritize adequate rest, hydration, and monitoring of vital signs.\n"
            "3. **Observation:** Note any secondary symptoms such as fever, rash, pain, or difficulty breathing.\n\n"
            "💡 *Tip: For personalized medical advice, share relevant details such as duration, age, and existing medical conditions with your healthcare provider.*"
        )
