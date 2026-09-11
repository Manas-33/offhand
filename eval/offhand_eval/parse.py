"""Extract tool calls from raw model text.

Two parsers, and the gap between them is a headline metric:

* ``parse_tool_calls`` (strict) accepts only a well-formed
  ``<tool_call>{"name":..., "arguments":...}</tool_call>`` block, with a JSON
  fallback for a bare object that still carries a "name". This is what a phone
  app's runtime can actually execute, so it is the honest reliability number.
* ``parse_tool_calls_lenient`` also recovers near-miss formats where the model
  had the right intent but broke the structure, e.g. ``draft_email {...}`` or
  ``.create_note {...}`` (function name leaked outside the JSON, wrapper
  dropped). The strict-to-lenient gap is the "format brittleness" that
  grammar-constrained decoding is meant to remove.
"""

import json
import re
from typing import Any, Iterable

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _try_json(blob: str) -> dict | None:
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _balanced_object_at(text: str, start: int) -> dict | None:
    """Parse the balanced ``{...}`` beginning at index ``start``."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _try_json(text[start : i + 1])
    return None


def _first_json_object(text: str) -> dict | None:
    """Scan for the first balanced ``{...}`` that parses as a dict."""
    for i, ch in enumerate(text):
        if ch == "{":
            obj = _balanced_object_at(text, i)
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
    """Strict parse: well-formed <tool_call> blocks, or a bare object with a name."""
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


def parse_tool_calls_lenient(text: str, valid_names: Iterable[str]) -> list[dict[str, Any]]:
    """Lenient parse: strict, plus name-prefixed calls like ``.create_note {...}``.

    Recovers the common quantization failure mode where the model emits the
    right function name and arguments but drops the ``<tool_call>`` wrapper and
    leaves the name outside the JSON object. Only names in ``valid_names`` are
    considered, and the name must be immediately followed by a ``{``, so tool
    names merely mentioned in prose are not turned into calls.
    """
    calls = parse_tool_calls(text)
    if calls:
        return calls

    best: tuple[int, dict] | None = None
    for name in sorted(valid_names, key=len, reverse=True):  # prefer longer names
        for m in re.finditer(rf"(?<!\w){re.escape(name)}\s*\{{", text):
            brace = m.end() - 1
            obj = _balanced_object_at(text, brace)
            if obj is not None and (best is None or m.start() < best[0]):
                best = (m.start(), {"name": name, "arguments": obj})
    return [_normalize_call(best[1])] if best is not None else []
