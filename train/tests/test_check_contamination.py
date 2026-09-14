"""Contamination check tests, no downloads: `python train/tests/test_check_contamination.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_contamination as cc

BFCL = [
    {"query": "Find the area of a triangle with a base of 10 units and height of 5 units.",
     "tools": [{"type": "function", "function": {"name": "calculate_triangle_area"}}]},
    {"query": "What is the capital of Brazil?",
     "tools": [{"type": "function", "function": {"name": "get_capital"}}]},
]


def test_flags_copied_questions():
    train = [
        {"query": "find the area of a triangle with a base of 10 units and height of 5 units"},  # exact
        {"query": "Find the area of a triangle with a base of 10 units"},                          # contained
        {"query": "what's the weather like in Paris right now?"},                                  # clean
        {"query": "what is the capital of Brazil?"},                                               # exact
    ]
    flagged = cc.query_overlaps(train, [b["query"] for b in BFCL])
    assert [item["query"] for item, _ in flagged] == [train[0]["query"], train[1]["query"], train[3]["query"]]


def test_flags_light_rewording():
    q = "please find the area of a triangle with a base of 10 units and a height of 5"
    flagged = cc.query_overlaps([{"query": q}], [BFCL[0]["query"]])
    assert len(flagged) == 1 and "trigrams" in flagged[0][1]


def test_short_common_phrasing_is_not_flagged():
    assert cc.query_overlaps([{"query": "what is the time"}], ["What is the time complexity of quicksort?"]) == []


def test_name_collisions():
    tools = [{"type": "function", "function": {"name": "calculate_triangle_area"}},
             {"type": "function", "function": {"name": "fx_rate"}}]
    assert cc.name_collisions(tools, BFCL) == ["calculate_triangle_area"]


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all check_contamination tests passed.")


if __name__ == "__main__":
    _run_all()
