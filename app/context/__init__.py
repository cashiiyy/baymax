"""BAYMAX AI – context package."""
from app.context.prompt_templates import (
    BAYMAX_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES,
    build_context_prompt, build_chat_messages, format_system_prompt,
)
from app.context.builder import ContextBuilder, BuiltContext

__all__ = [
    "BAYMAX_SYSTEM_PROMPT", "FEW_SHOT_EXAMPLES",
    "build_context_prompt", "build_chat_messages", "format_system_prompt",
    "ContextBuilder", "BuiltContext",
]
