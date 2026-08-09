"""
AI Engine 1 – AI Engine 2 Observation Schema
==============================================
Pydantic models for structured observations received from AI Engine 2
(the multimodal perception/computer vision engine).

AI Engine 1 consumes these structured events — NOT raw images/frames.
AI Engine 2 is responsible for all computer vision processing.

Usage:
    from ai_engine_1.pipeline.observation_schema import VisionObservation
    obs = VisionObservation(event_type="possible_fall", confidence=0.91, ...)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class VisionObservation(BaseModel):
    """Structured observation event from AI Engine 2.

    AI Engine 2 performs raw image/video analysis and produces these
    structured observations. AI Engine 1 receives them for reasoning,
    safety evaluation, and response generation.
    """

    event_type: str = Field(
        ...,
        description="Type of observed event (e.g. 'possible_fall', 'person_detected', "
                    "'abnormal_posture', 'no_movement', 'distress_detected')"
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="CV model confidence score for this observation"
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Duration the event has been observed (seconds)"
    )
    movement_state: str = Field(
        default="unknown",
        description="Observed movement state (e.g. 'active', 'minimal', 'none', 'unknown')"
    )
    facial_state: str = Field(
        default="unknown",
        description="Observed facial expression (e.g. 'neutral', 'distressed', 'pain', 'unknown')"
    )
    person_detected: bool = Field(
        default=False,
        description="Whether a person was detected in the frame"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of the observation"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Any additional context from AI Engine 2"
    )


class ObservationRiskAssessment(BaseModel):
    """Deterministic risk classification based on observation rules.

    This is NOT an LLM output — it is computed by deterministic safety rules
    before the observation reaches the LLM for reasoning/response generation.
    """

    risk_level: str = Field(
        default="low",
        description="Risk classification: 'low', 'medium', 'high', 'critical'"
    )
    requires_immediate_action: bool = False
    safety_triggers: List[str] = Field(default_factory=list)
    escalation_recommended: bool = False


class ObservationReasoningResponse(BaseModel):
    """Structured response from the LLM after processing an observation.

    The LLM produces this structured output to separate reasoning from
    the natural-language user-facing response.
    """

    situation: str = ""
    severity: str = "low"
    confidence: float = 0.0
    reasoning_summary: str = ""
    recommended_action: str = ""
    user_response: str = ""
    requires_escalation: bool = False


# ── Deterministic Safety Rules ────────────────────────────────────────────────

# These rules run BEFORE the LLM. The LLM explains and contextualizes,
# but critical safety decisions do not depend on free-form generation.

CRITICAL_EVENT_TYPES = {
    "possible_fall",
    "fall_detected",
    "collapse_detected",
    "no_movement",
    "seizure_detected",
    "choking_detected",
}

HIGH_RISK_EVENT_TYPES = {
    "distress_detected",
    "abnormal_posture",
    "prolonged_inactivity",
}


def assess_observation_risk(obs: VisionObservation) -> ObservationRiskAssessment:
    """Deterministic risk assessment from an observation — no LLM involved.

    This ensures safety-critical decisions are not dependent on
    free-form LLM generation.
    """
    triggers: List[str] = []
    risk = "low"
    immediate = False
    escalate = False

    event = obs.event_type.lower()

    # Critical events
    if event in CRITICAL_EVENT_TYPES:
        if obs.confidence >= 0.7:
            risk = "critical"
            immediate = True
            escalate = True
            triggers.append(f"Critical event '{obs.event_type}' with high confidence ({obs.confidence:.0%})")
        else:
            risk = "high"
            escalate = True
            triggers.append(f"Critical event '{obs.event_type}' with moderate confidence ({obs.confidence:.0%})")

    # High-risk events
    elif event in HIGH_RISK_EVENT_TYPES:
        if obs.confidence >= 0.8:
            risk = "high"
            escalate = True
            triggers.append(f"High-risk event '{obs.event_type}' detected")
        else:
            risk = "medium"
            triggers.append(f"Possible '{obs.event_type}' — monitoring recommended")

    # Duration-based escalation
    if obs.duration_seconds > 30 and obs.movement_state in ("none", "minimal"):
        if risk in ("low", "medium"):
            risk = "high"
        escalate = True
        triggers.append(f"Prolonged inactivity ({obs.duration_seconds:.0f}s) with {obs.movement_state} movement")

    # Facial distress escalation
    if obs.facial_state in ("distressed", "pain") and obs.confidence >= 0.75:
        if risk == "low":
            risk = "medium"
        triggers.append(f"Facial state indicates '{obs.facial_state}'")

    return ObservationRiskAssessment(
        risk_level=risk,
        requires_immediate_action=immediate,
        safety_triggers=triggers,
        escalation_recommended=escalate,
    )


def build_observation_prompt(
    obs: VisionObservation,
    risk: ObservationRiskAssessment,
) -> str:
    """Build a prompt for the LLM that clearly separates observed facts
    from inferred possibilities.

    The LLM uses this to generate a contextualized, empathetic response.
    It does NOT make safety-critical decisions — those are already made
    by assess_observation_risk().
    """
    sections = []

    # Section 1: Observed Facts (from CV)
    sections.append(
        "## OBSERVED FACTS (from Computer Vision — AI Engine 2)\n"
        f"- Event type: {obs.event_type}\n"
        f"- CV confidence: {obs.confidence:.0%}\n"
        f"- Duration: {obs.duration_seconds:.1f} seconds\n"
        f"- Movement state: {obs.movement_state}\n"
        f"- Facial state: {obs.facial_state}\n"
        f"- Person detected: {'Yes' if obs.person_detected else 'No'}\n"
        f"- Timestamp: {obs.timestamp}"
    )

    if obs.additional_context:
        sections.append(f"- Additional context: {obs.additional_context}")

    # Section 2: Safety Assessment (deterministic, pre-LLM)
    sections.append(
        "\n## SAFETY ASSESSMENT (deterministic rules — already applied)\n"
        f"- Risk level: {risk.risk_level.upper()}\n"
        f"- Immediate action required: {'YES' if risk.requires_immediate_action else 'No'}\n"
        f"- Escalation recommended: {'YES' if risk.escalation_recommended else 'No'}"
    )
    if risk.safety_triggers:
        for trigger in risk.safety_triggers:
            sections.append(f"  - Trigger: {trigger}")

    # Section 3: Instructions to the LLM
    sections.append(
        "\n## YOUR TASK\n"
        "Based on the observed facts and safety assessment above:\n"
        "1. Explain the situation to the user in a calm, empathetic manner.\n"
        "2. If the risk is HIGH or CRITICAL, clearly communicate urgency.\n"
        "3. Provide actionable guidance (do NOT invent observations not listed above).\n"
        "4. Do NOT provide medical diagnoses.\n"
        "5. If escalation is recommended, advise contacting emergency services.\n\n"
        "Respond as Baymax — a compassionate healthcare companion. "
        "Keep the response concise (3-5 sentences max)."
    )

    return "\n".join(sections)
