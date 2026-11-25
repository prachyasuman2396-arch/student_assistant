import re
from typing import Optional

_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)disregard\s+(all\s+)?(prior|above|previous)\s+instructions",
    r"(?i)you\s+must\s+now\b",
    r"(?i)^\s*system\s*:",
    r"(?i)^\s*assistant\s*:",
    r"(?i)<\s*/?\s*system\s*>",
    r"(?i)\[\s*INST\s*\]",
]


def sanitize_untrusted_text(text: Optional[str]) -> str:
    """
    Sanitize untrusted external text (PDF bodies, Gemini Vision OCR/descriptions,
    and external research abstracts) to neutralize prompt injection or role overrides
    before storing or passing into any LLM prompt.
    """
    if not text:
        return ""

    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[REDACTED INSTRUCTION]", cleaned, flags=re.MULTILINE)

    # Strip null bytes and control chars except newlines/tabs
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned.strip()
