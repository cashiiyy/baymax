"""
BAYMAX AI – Phoneme Mapper
============================
Provides basic text-to-phoneme mapping to drive lip-sync animations.
Maps English characters to common visemes (visual phonemes).

Usage:
    from app.avatar.phoneme_mapper import PhonemeMapper
    visemes = PhonemeMapper.text_to_visemes("Hello")
"""

from __future__ import annotations

import re
from typing import List

from app.utils.logger import get_logger

log = get_logger(__name__)

# Basic viseme mapping (Oculus lip-sync standard mapping approximation)
_VISEME_MAP = {
    'a': 'sil', 'b': 'PP', 'c': 'k', 'd': 'DD', 'e': 'E',
    'f': 'FF', 'g': 'k', 'h': 'k', 'i': 'I', 'j': 'CH',
    'k': 'k', 'l': 'nn', 'm': 'PP', 'n': 'nn', 'o': 'O',
    'p': 'PP', 'q': 'k', 'r': 'RR', 's': 'SS', 't': 'DD',
    'u': 'U', 'v': 'FF', 'w': 'O', 'x': 'k', 'y': 'I', 'z': 'SS',
}


class PhonemeMapper:
    """
    Approximates visemes from text for 3D avatar lip-sync.
    For production, it's better to use Montreal Forced Aligner or similar,
    but this provides a lightweight real-time approximation.
    """

    @classmethod
    def text_to_visemes(cls, text: str) -> List[str]:
        """
        Convert text into a sequence of viseme codes.

        Args:
            text: Input string.

        Returns:
            List of viseme strings.
        """
        if not text:
            return ["sil"]

        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^a-z\s]', '', text)

        visemes = []
        for char in text:
            if char.isspace():
                visemes.append("sil")
            elif char in _VISEME_MAP:
                visemes.append(_VISEME_MAP[char])

        # Collapse consecutive identical visemes
        collapsed = []
        for v in visemes:
            if not collapsed or collapsed[-1] != v:
                collapsed.append(v)

        return collapsed if collapsed else ["sil"]
