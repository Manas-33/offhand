"""Harness unit tests — run with no ML deps: `python eval/tests/test_harness.py`.

Exercises parsing, the four scoring signals, and a full mock run over the real
dataset so the plumbing is trusted before it ever touches a GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offhand_eval.dataset import load_items, load_tools, tools_by_name
from offhand_eval.harness import run_eval
from offhand_eval.mock import build_mock_runner
from offhand_eval.parse import parse_tool_calls
from offhand_eval.scoring import args_match, is_schema_valid, score_item

HERE = Path(__file__).resolve().parents[1]
TOOLS = load_tools(HERE / "tools.json")
BY_NAME = tools_by_name(TOOLS)


def test_parse_wrapped_and_bare():
    wrapped = '<tool_call>\n{"name": "set_alarm", "arguments": {"time": "07:00"}}\n</tool_call>'
    assert parse_tool_calls(wrapped)[0]["name"] == "set_alarm"

    bare = 'Sure! {"name": "toggle_flashlight", "arguments": {"state": "on"}}'
    assert parse_tool_calls(bare)[0]["arguments"]["state"] == "on"

    assert parse_tool_calls("The capital of France is Paris.") == []


def test_parse_double_encoded_arguments():
    text = '<tool_call>{"name": "set_timer", "arguments": "{\\"duration_minutes\\": 10}"}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0]["arguments"]["duration_minutes"] == 10


def test_schema_validity():
    assert is_schema_valid({"name": "set_alarm", "arguments": {"time": "07:00"}}, BY_NAME)
    # missing required param
    assert not is_schema_valid({"name": "set_alarm", "arguments": {"label": "x"}}, BY_NAME)
    # unknown tool
    assert not is_schema_valid({"name": "launch_rocket", "arguments": {}}, BY_NAME)
    # unknown param (strict)
    assert not is_schema_valid(
        {"name": "toggle_flashlight", "arguments": {"state": "on", "color": "red"}}, BY_NAME
    )


def test_args_match_forms():
    assert args_match({"panel": "wifi"}, {"panel": ["wifi"]})
    assert args_match({"time": "7 AM"}, {"time": ["07:00", "7 am"]})
    assert args_match({"content": "Buy milk from the store"}, {"content": {"contains": "milk"}})
    assert not args_match({"content": "call the plumber"}, {"content": {"contains": "milk"}})


def test_score_correct_and_wrong_tool():
    item = {"id": "x", "category": "single",
            "gold": {"name": "open_settings", "arguments": {"panel": ["wifi"]}}}

    good = parse_tool_calls('<tool_call>{"name":"open_settings","arguments":{"panel":"wifi"}}</tool_call>')
    s = score_item(item, good, BY_NAME)
    assert s.schema_valid and s.right_tool and s.right_args

    wrong = parse_tool_calls('<tool_call>{"name":"toggle_flashlight","arguments":{"state":"on"}}</tool_call>')
    s = score_item(item, wrong, BY_NAME)
    assert s.schema_valid and not s.right_tool and not s.right_args


def test_score_irrelevance():
    item = {"id": "irr", "category": "irrelevance", "gold": {"no_call": True}}
    assert score_item(item, [], BY_NAME).no_call_correct is True
    spurious = parse_tool_calls('<tool_call>{"name":"play_music","arguments":{"action":"play"}}</tool_call>')
    assert score_item(item, spurious, BY_NAME).no_call_correct is False


def test_full_mock_run_is_perfect():
    """A mock that echoes the reference-correct call scores 100% on the dataset."""
    items = load_items(HERE / "mini_eval.jsonl")
    outcome = run_eval(items, BY_NAME, build_mock_runner(items, BY_NAME))
    summary = outcome["summary"]
    assert summary["call_accuracy"] == 1.0, summary
    assert summary["schema_valid_rate"] == 1.0, summary
    assert summary["irrelevance_accuracy"] == 1.0, summary


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
