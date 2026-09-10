"""Extract tool calls from raw model text.

Qwen3 (and most Hermes-style templates) emit tool calls wrapped in
``<tool_call> ... </tool_call>`` blocks containing JSON. We parse those first,
then fall back to the first bare JSON object that looks like a call so that a
model which forgets the wrapper is not scored as a hard failure.
"""

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _try_json(blob: str) -> dict | None:
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _first_json_object(text: str) -> dict | None:
    """Scan for the first balanced ``{...}`` that parses as a dict."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    obj = _try_json(text[start : i + 1])
                    if obj is not None:
                        return obj
    return None


def _normalize_call(obj: dict) -> dict:
    """Coerce a parsed object into ``{"name": str, "arguments": dict}``."""
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):  # some models double-encode the arguments
        args = _try_json(args) or {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Return every tool call found in ``text`` as normalized dicts."""
    calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(text):
        obj = _try_json(match.group(1))
        if obj is not None and "name" in obj:
            calls.append(_normalize_call(obj))

    if not calls:
        obj = _first_json_object(text)
        if obj is not None and "name" in obj:
            calls.append(_normalize_call(obj))

    return calls
