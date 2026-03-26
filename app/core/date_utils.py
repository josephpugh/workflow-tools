from __future__ import annotations

import re
from datetime import date, timedelta


_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def extract_date_expression(text: str) -> str | None:
    absolute_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},\s*\d{4})\b", text)
    if absolute_match:
        return absolute_match.group(1)

    relative_patterns = [
        r"\b(today|tomorrow|yesterday)\b",
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    ]
    for pattern in relative_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def resolve_date_expression(value: str, reference_date: date) -> str | None:
    normalized = value.strip()
    lowered = normalized.lower()

    if lowered == "today":
        return reference_date.isoformat()
    if lowered == "tomorrow":
        return (reference_date + timedelta(days=1)).isoformat()
    if lowered == "yesterday":
        return (reference_date - timedelta(days=1)).isoformat()

    next_match = re.fullmatch(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", lowered)
    if next_match:
        target = _WEEKDAY_INDEX[next_match.group(1)]
        delta = (target - reference_date.weekday()) % 7
        delta = 7 if delta == 0 else delta
        return (reference_date + timedelta(days=delta)).isoformat()

    this_match = re.fullmatch(r"this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", lowered)
    if this_match:
        target = _WEEKDAY_INDEX[this_match.group(1)]
        delta = (target - reference_date.weekday()) % 7
        return (reference_date + timedelta(days=delta)).isoformat()

    return None
