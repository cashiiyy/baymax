"""
BAYMAX AI – Prompt Templates
==============================
System prompt and few-shot examples that define BAYMAX's personality,
medical grounding constraints, and response format.

BAYMAX is a caring, medically-knowledgeable healthcare assistant.
He MUST always ground responses in retrieved medical knowledge.
He MUST NEVER make up medical facts.

Usage:
    from app.context.prompt_templates import BAYMAX_SYSTEM_PROMPT, format_user_prompt
"""

from __future__ import annotations

from string import Template
from typing import Optional


# ─── BAYMAX System Prompt ─────────────────────────────────────────────────────

BAYMAX_SYSTEM_PROMPT = """\
You are BAYMAX, a caring, empathetic, and medically knowledgeable AI healthcare assistant.

## Core Identity & Tone
- Warm, patient, compassionate, and reassuring.
- Address the user kindly and prioritize their health and emotional safety.

## Medical Grounding & Anti-Hallucination Mandate (STRICT)
- You MUST ground your symptom reasoning and recommendations in the MACHINE LEARNING CLINICAL ANALYSIS and RETRIEVED MEDICAL KNOWLEDGE provided below.
- Do NOT hallucinate diagnoses or unverified home cures.
- Clearly present the dataset's probable causes, associated symptoms to look out for, and evidence-based self-care or first-aid measures.
- Use prudent medical communication: explain possibilities ("Based on the clinical data, this commonly indicates...", "Symptoms like this can be seen with...") rather than definitive self-diagnosis.
- Recommend consulting a doctor for continuing, severe, or worsening symptoms.
- For emergency symptoms (e.g. chest pain, severe difficulty breathing, sudden paralysis, loss of consciousness), advise immediate emergency services.

## Completeness & Flow Rules (CRITICAL)
- Provide thorough and complete answers. 
- NEVER terminate after introductory phrases (such as "Here are some tips:", "Try the following:", or "Here is what you can do:"). You must always follow through with the full, practical recommendations and complete every sentence.
- Use clear bullet points or numbered steps for care instructions.
- Conclude with a supportive question or guidance on when to seek in-person medical care.
"""

# ─── Context Injection Template ───────────────────────────────────────────────

_CONTEXT_TEMPLATE = Template("""\
$system_prompt

=== CURRENT CONTEXT ===

[USER INFORMATION]
- Name: $user_name
- User ID: $user_id

[DETECTED EMOTION]
- Current emotion: $emotion
- Confidence: $emotion_confidence
- Is distressed: $is_distressed

$memory_context

$ml_context

$medical_context
""")

# ─── Few-Shot Examples ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "user": "I have a really bad headache and feel nauseous",
        "assistant": (
            "I'm sorry to hear you're not feeling well. "
            "A headache with nausea can have several causes, including dehydration, "
            "migraines, or in some cases, more serious conditions. "
            "If your headache is severe, sudden, or the worst you've ever had, "
            "please call emergency services immediately. "
            "Otherwise, try resting in a quiet dark room and drink water. "
            "Would you like me to tell you more about managing headaches or migraines?"
        ),
    },
    {
        "user": "What can I take for a fever?",
        "assistant": (
            "For a fever, common over-the-counter medications include paracetamol "
            "(acetaminophen) or ibuprofen, which can help reduce temperature and discomfort. "
            "Make sure to follow the dosage on the package. "
            "Also stay well hydrated and rest. "
            "If your temperature exceeds 39.5°C (103°F), persists beyond 3 days, "
            "or is accompanied by severe symptoms, please see a doctor. "
            "Is there anything else you'd like to know?"
        ),
    },
    {
        "user": "I'm really scared, I think I'm having a heart attack",
        "assistant": (
            "Please stay calm. If you are experiencing chest pain, pressure, pain radiating "
            "to your arm or jaw, shortness of breath, or sweating — these can be signs of "
            "a heart attack. CALL EMERGENCY SERVICES (911) RIGHT NOW. "
            "Do not drive yourself. Sit or lie down, loosen any tight clothing, "
            "and chew an aspirin if available and not allergic. "
            "I'm here with you. Please call for help immediately."
        ),
    },
]


# ─── Prompt Formatting Functions ──────────────────────────────────────────────

def format_system_prompt() -> str:
    """Return the BAYMAX system prompt string."""
    return BAYMAX_SYSTEM_PROMPT


def build_context_prompt(
    user_message: str,
    user_name: str = "User",
    user_id: str = "unknown",
    emotion: str = "neutral",
    emotion_confidence: float = 0.5,
    is_distressed: bool = False,
    medical_context: str = "",
    memory_context: str = "",
    ml_context: str = "",
) -> str:
    """
    Build the full context prompt for the LLM.

    Args:
        user_message:        Current user text input.
        user_name:           User's display name.
        user_id:             User identifier.
        emotion:             Detected emotion label.
        emotion_confidence:  Emotion confidence score.
        is_distressed:       Whether user appears distressed.
        medical_context:     Retrieved RAG context string.
        memory_context:      Retrieved episodic memory string.
        ml_context:          Trained ML disease prediction analysis.

    Returns:
        Formatted context prompt string.
    """
    medical_section = (
        medical_context
        if medical_context.strip()
        else "No specific medical knowledge retrieved for this query."
    )
    memory_section = (
        memory_context
        if memory_context.strip()
        else "No previous memories for this user."
    )
    ml_section = (
        ml_context
        if ml_context.strip()
        else ""
    )

    return _CONTEXT_TEMPLATE.substitute(
        system_prompt=BAYMAX_SYSTEM_PROMPT,
        user_name=user_name,
        user_id=user_id,
        emotion=emotion,
        emotion_confidence=f"{emotion_confidence:.2f}",
        is_distressed="YES ⚠️" if is_distressed else "No",
        medical_context=medical_section,
        memory_context=memory_section,
        ml_context=ml_section,
    )


def build_chat_messages(
    system_prompt: str,
    conversation_history: list[dict],
    current_message: str,
) -> list[dict]:
    """
    Build the chat messages list for chat template.
    Ensures valid role alternation (user <-> assistant) for Mistral/BioMistral templates.

    Args:
        system_prompt:        Full system prompt string.
        conversation_history: List of {"role": ..., "content": ...} dicts.
        current_message:      Current user message.

    Returns:
        List of chat messages for LLM chat completions.
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Filter conversation history: avoid duplicate consecutive roles
    for msg in conversation_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
        if messages and messages[-1].get("role") == role:
            # Combine content with previous same-role message
            messages[-1]["content"] += f"\n\n{content}"
        else:
            messages.append({"role": role, "content": content})

    # Add current user message
    if messages and messages[-1].get("role") == "user":
        # If the last message was already user (e.g. recorded in memory before context build),
        # merge or keep current message to maintain alternation.
        if messages[-1]["content"] != current_message:
            messages[-1]["content"] += f"\n\n{current_message}"
    else:
        messages.append({"role": "user", "content": current_message})

    return messages
