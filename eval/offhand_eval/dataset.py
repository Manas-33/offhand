"""Load tool schemas and eval items from disk."""

from __future__ import annotations

import json
from pathlib import Path


def load_tools(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        tools = json.load(fh)
    if not isinstance(tools, list):
        raise ValueError(f"{path} must contain a JSON list of tool schemas")
    return tools


def tools_by_name(tools: list[dict]) -> dict[str, dict]:
    return {t["function"]["name"]: t for t in tools}


def load_items(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return items
