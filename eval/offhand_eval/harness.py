"""Run a runner over the dataset and produce strict + lenient scores.

Every item is scored twice: strict (what an app runtime can execute) and lenient
(right intent even if the tool-call format is broken). The gap between them,
``format_recoverable``, is the reliability that grammar-constrained decoding
should be able to reclaim.
"""

from __future__ import annotations

from typing import Any, Callable

from .parse import parse_tool_calls, parse_tool_calls_lenient
from .scoring import ItemScore, aggregate, score_item


def run_eval(
    items: list[dict],
    tools_by_name: dict[str, dict],
    runner,
    on_item: Callable[[int, int, ItemScore], None] | None = None,
) -> dict[str, Any]:
    """Evaluate ``runner`` over ``items``.

    Returns ``{"summary": {"strict": {...}, "lenient": {...},
    "format_recoverable": float}, "results": [...]}``. ``on_item(index, total,
    strict_score)`` is called after each item for progress.
    """
    valid_names = list(tools_by_name.keys())
    results: list[dict] = []
    strict_scores: list[ItemScore] = []
    lenient_scores: list[ItemScore] = []
    total = len(items)

    for i, item in enumerate(items):
        raw = runner.generate(item["query"])
        strict_calls = parse_tool_calls(raw)
        lenient_calls = parse_tool_calls_lenient(raw, valid_names)
        strict_score = score_item(item, strict_calls, tools_by_name)
        lenient_score = score_item(item, lenient_calls, tools_by_name)
        strict_scores.append(strict_score)
        lenient_scores.append(lenient_score)
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "raw": raw,
                "strict_calls": strict_calls,
                "lenient_calls": lenient_calls,
                "strict_score": strict_score.as_dict(),
                "lenient_score": lenient_score.as_dict(),
            }
        )
        if on_item is not None:
            on_item(i + 1, total, strict_score)

    strict_agg = aggregate(strict_scores)
    lenient_agg = aggregate(lenient_scores)
    gap = None
    if strict_agg["call_accuracy"] is not None and lenient_agg["call_accuracy"] is not None:
        gap = round(lenient_agg["call_accuracy"] - strict_agg["call_accuracy"], 4)

    return {
        "summary": {"strict": strict_agg, "lenient": lenient_agg, "format_recoverable": gap},
        "results": results,
    }
