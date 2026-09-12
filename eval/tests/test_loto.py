"""LOTO reporting test: run the reference-correct mock over mini_eval and check
the seen/unseen/no-call bucketing and rates. No ML deps.
Run: `python eval/tests/test_loto.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/

from offhand_eval.dataset import load_items, load_tools, tools_by_name
from offhand_eval.harness import run_eval
from offhand_eval.loto import HELD_OUT, format_loto, loto_breakdown
from offhand_eval.mock import build_mock_runner

HERE = Path(__file__).resolve().parents[1]
SEEN = {"set_alarm", "set_timer", "create_note", "open_settings", "draft_email", "play_music", "set_reminder"}


def _run():
    tools = load_tools(HERE / "tools.json")
    items = load_items(HERE / "mini_eval.jsonl")
    runner = build_mock_runner(items, tools_by_name(tools))
    return items, run_eval(items, tools, runner)


def test_buckets_and_counts():
    items, outcome = _run()
    bd = loto_breakdown(items, outcome["results"], HELD_OUT, "strict")
    assert set(bd["seen"]["tools"]) == SEEN, bd["seen"]["tools"]
    assert set(bd["unseen"]["tools"]) == set(HELD_OUT), bd["unseen"]["tools"]
    assert all(m["n"] == 4 for m in bd["per_tool"].values())     # mini_eval: 4 per tool
    assert bd["seen"]["n"] == 28 and bd["unseen"]["n"] == 12
    assert bd["no_call"]["n"] == 8


def test_reference_mock_scores_perfect():
    items, outcome = _run()
    bd = loto_breakdown(items, outcome["results"], HELD_OUT, "strict")
    # the reference-correct mock should ace every bucket, including the held-out trio
    assert bd["seen"]["call_accuracy"] == 1.0
    assert bd["unseen"]["call_accuracy"] == 1.0
    assert bd["no_call"]["accuracy"] == 1.0
    assert "Gate signal" in format_loto(bd)  # renders without error


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all loto tests passed.")


if __name__ == "__main__":
    _run_all()
