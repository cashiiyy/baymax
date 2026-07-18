"""
BAYMAX AI – LLM Response Parser
=================================
Post-processes raw LLM output to:
  - Clean up artifacts and formatting issues
  - Extract suggested emotion from response content
  - Detect safety-critical phrases (emergency advice)
  - Assess whether the response is medically grounded

Usage:
    from app.llm.response_parser import ResponseParser
    parser = ResponseParser()
    parsed = parser.parse(raw_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# Patterns indicating emergency advice
_EMERGENCY_PATTERNS = [
    r"call\s+(911|emergency|ambulance)",
    r"emergency\s+service",
    r"immediately\s+seek\s+medical",
    r"go\s+to\s+(the\s+)?(er|emergency\s+room|hospital)",
    r"heart\s+attack",
    r"stroke",
    r"call\s+for\s+help",
]

# Emotion keywords for response-based emotion suggestion
_EMOTION_KEYWORDS: dict[str, List[str]] = {
    "empathetic": ["sorry", "understand", "feel", "difficult", "hard", "tough"],
    "reassuring": ["don't worry", "you'll be ok", "manageable", "treatable", "help"],
    "urgent": ["immediately", "emergency", "right now", "call 911"],
    "informative": ["the cause", "this is", "this means", "typically", "usually"],
}


@dataclass
class ParsedResponse:
    """
    Structured output from the response parser.

    Attributes:
        text:               Cleaned response text.
        suggested_emotion:  Avatar emotion to display ('empathetic', 'reassuring', etc.)
        is_emergency:       Whether emergency services were recommended.
        is_grounded:        Whether the response references medical information.
        contains_disclaimer: Whether a medical disclaimer is present.
        bullet_points:      Any extracted list items.
    """
    text: str
    suggested_emotion: str = "neutral"
    is_emergency: bool = False
    is_grounded: bool = False
    contains_disclaimer: bool = False
    bullet_points: List[str] = field(default_factory=list)

    @property
    def avatar_emotion(self) -> str:
        """Map suggested emotion to avatar expression name."""
        mapping = {
            "empathetic": "sad",
            "reassuring": "happy",
            "urgent": "surprise",
            "informative": "neutral",
            "neutral": "neutral",
        }
        return mapping.get(self.suggested_emotion, "neutral")


class ResponseParser:
    """
    Post-processes raw Qwen LLM output into structured ParsedResponse.
    """

    # Phrases that shouldn't appear at the start of BAYMAX's response
    _PREAMBLE_PATTERNS = [
        r"^(assistant|baymax|ai):\s*",
        r"^(sure|certainly|absolutely|of course)[,!.]?\s*",
        r"^i would be happy to help[.!]?\s*",
    ]

    def parse(self, raw_text: str) -> ParsedResponse:
        """
        Parse and clean the raw LLM output.

        Args:
            raw_text: Raw text output from Qwen.

        Returns:
            ParsedResponse with cleaned text and metadata.
        """
        # Clean basic artifacts
        text = self._clean_text(raw_text)

        # Detect features
        is_emergency = self._detect_emergency(text)
        is_grounded = self._detect_grounding(text)
        contains_disclaimer = self._detect_disclaimer(text)
        suggested_emotion = self._suggest_emotion(text, is_emergency)
        bullet_points = self._extract_bullet_points(text)

        return ParsedResponse(
            text=text,
            suggested_emotion=suggested_emotion,
            is_emergency=is_emergency,
            is_grounded=is_grounded,
            contains_disclaimer=contains_disclaimer,
            bullet_points=bullet_points,
        )

    # ── Private Methods ───────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Remove preamble patterns and normalize whitespace."""
        text = text.strip()

        # Remove common AI preamble phrases
        for pattern in self._PREAMBLE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        # Remove any trailing "</s>" or other EOS tokens
        text = re.sub(r"</s>$", "", text).strip()

        return text

    @staticmethod
    def _detect_emergency(text: str) -> bool:
        """Return True if the response contains emergency advice."""
        text_lower = text.lower()
        return any(
            re.search(p, text_lower) for p in _EMERGENCY_PATTERNS
        )

    @staticmethod
    def _detect_grounding(text: str) -> bool:
        """
        Check if the response references specific medical information
        (indicates RAG context was used).
        """
        grounding_signals = [
            "according to", "research shows", "studies indicate",
            "symptoms include", "commonly associated", "medical knowledge",
            "treatment involves", "medication", "diagnosis",
        ]
        text_lower = text.lower()
        return any(sig in text_lower for sig in grounding_signals)

    @staticmethod
    def _detect_disclaimer(text: str) -> bool:
        """Check if response contains a medical disclaimer."""
        disclaimer_signals = [
            "not a replacement", "consult a doctor", "see a healthcare",
            "medical professional", "ai assistant", "not medical advice",
        ]
        text_lower = text.lower()
        return any(sig in text_lower for sig in disclaimer_signals)

    @staticmethod
    def _suggest_emotion(text: str, is_emergency: bool) -> str:
        """
        Suggest an avatar emotion based on response content.

        Args:
            text:         Response text.
            is_emergency: Whether emergency advice is present.

        Returns:
            Emotion label string.
        """
        if is_emergency:
            return "urgent"

        text_lower = text.lower()
        scores: dict[str, int] = {e: 0 for e in _EMOTION_KEYWORDS}

        for emotion, keywords in _EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[emotion] += 1

        if all(v == 0 for v in scores.values()):
            return "informative"

        return max(scores, key=lambda k: scores[k])

    @staticmethod
    def _extract_bullet_points(text: str) -> List[str]:
        """
        Extract ordered or unordered list items from the response.

        Args:
            text: Response text.

        Returns:
            List of extracted items (stripped).
        """
        items = []
        patterns = [
            r"^\s*[-•*]\s+(.+)$",        # Unordered bullets
            r"^\s*\d+[.)]\s+(.+)$",       # Ordered lists
            r"^\s*Step\s+\d+[.:]\s+(.+)$", # Step N: format
        ]
        for line in text.splitlines():
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    items.append(match.group(1).strip())
                    break
        return items
