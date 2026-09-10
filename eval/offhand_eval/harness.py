"""Run a runner over the dataset and produce per-item + aggregate scores."""

from __future__ import annotations

from typing import Any, Callable

from .parse import parse_tool_calls
from .runners import ModelRunner
from .scoring import ItemScore, aggregate, score_item


def run_eval(
    items: list[dict],
    tools_by_name: dict[str, dict],
    runner: ModelRunner,
    on_item: Callable[[int, int, ItemScore], None] | None = None,
) -> dict[str, Any]:
    """Evaluate ``runner`` over ``items``.

    Returns ``{"summary": {...}, "results": [ {item, raw, calls, score}, ... ]}``.
    ``on_item(index, total, score)`` is called after each item for progress.
    """
    results: list[dict] = []
    scores: list[ItemScore] = []
    total = len(items)

    for i, item in enumerate(items):
        raw = runner.generate(item["query"])
        calls = parse_tool_calls(raw)
        score = score_item(item, calls, tools_by_name)
        scores.append(score)
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "raw": raw,
                "calls": calls,
                "score": score.as_dict(),
            }
        )
        if on_item is not None:
            on_item(i + 1, total, score)

    return {"summary": aggregate(scores), "results": results}
