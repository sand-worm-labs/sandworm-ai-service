from __future__ import annotations

import json
import re


class LLMJSONError(ValueError):
    """Raised when a model's response can't be turned into a JSON object,
    either because it isn't valid JSON or because it isn't shaped as one."""


def extract_json_candidate(raw: str) -> str:
    """Pre-parse step: strip markdown code fences and any leading/trailing
    prose around a JSON object, so a response the model wrapped in commentary
    (despite being told not to) still has a shot at parsing."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def parse_json_object(raw: str) -> dict:
    """Pre-parse + validate a model response as a JSON object.

    Raises LLMJSONError (never a bare JSONDecodeError) on anything that isn't
    a well-formed JSON object — empty output, malformed JSON, or valid JSON
    that isn't a dict (e.g. a bare list or string) — so every caller has one
    exception type to catch regardless of how the response was broken.
    """
    candidate = extract_json_candidate(raw)
    if not candidate:
        raise LLMJSONError("empty response after stripping fences/whitespace")

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMJSONError(f"not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMJSONError(f"expected a JSON object, got {type(data).__name__}")

    return data
