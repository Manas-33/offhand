"""Device replay unit tests, no device or ML deps: `python eval/tests/test_device_replay.py`.

Checks that recorded device outputs score exactly like a live run, that an
incomplete device run is rejected, and that compare_runs reads the verdicts the
harness stored.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_runs
import run_device_eval
from offhand_eval.dataset import load_items, load_tools, tools_by_name
from offhand_eval.harness import run_eval
from offhand_eval.mock import build_mock_runner

HERE = Path(__file__).resolve().parents[1]
TOOLS = load_tools(HERE / "tools.json")
ITEMS = load_items(HERE / "mini_eval.jsonl")


def _mock():
    return build_mock_runner(ITEMS, tools_by_name(TOOLS))


def _device_rows(runner):
    """What EvalActivity would write if the device produced the runner's outputs."""
    return [{"id": it["id"], "query": it["query"], "raw": runner.generate(it["query"], TOOLS), "ttft_ms": 20.0}
            for it in ITEMS]


def test_replay_scores_like_a_live_run():
    runner = _mock()
    live = run_eval(ITEMS, TOOLS, runner)
    replay = run_device_eval.score_device_rows(ITEMS, TOOLS, _device_rows(runner))
    assert replay["summary"] == live["summary"]
    assert [r["strict_score"] for r in replay["results"]] == [r["strict_score"] for r in live["results"]]
    assert replay["results"][0]["device"] == {"ttft_ms": 20.0}
    assert replay["loto"]["strict"]["seen"]["call_accuracy"] == 1.0


def test_replay_rejects_missing_items():
    rows = _device_rows(_mock())[1:]
    try:
        run_device_eval.score_device_rows(ITEMS, TOOLS, rows)
    except ValueError as exc:
        assert ITEMS[0]["id"] in str(exc)
    else:
        raise AssertionError("an incomplete device run must fail loudly")


def test_parse_device_rows_skips_blank_lines():
    text = '{"id": "a", "raw": "x"}\n\n{"id": "b", "raw": "y"}\n'
    assert [r["id"] for r in run_device_eval.parse_device_rows(text)] == ["a", "b"]


def test_compare_runs_reads_stored_verdicts():
    outcome = run_eval(ITEMS, TOOLS, _mock())
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"label": "specialist_w4a16_seqmse", **outcome}, fh)
        path = fh.name
    try:
        run = compare_runs.load_run(path)
        assert run["name"] == "seqmse"
        assert all(v["correct"] for v in run["items"].values())
        first = ITEMS[0]
        assert run["items"][first["id"]]["tool"] == first["gold"].get("name")
        assert compare_runs.load_run(f"fp16={path}")["name"] == "fp16"
    finally:
        os.unlink(path)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
