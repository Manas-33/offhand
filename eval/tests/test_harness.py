"""Harness unit tests — run with no ML deps: `python eval/tests/test_harness.py`.

Exercises parsing (strict and lenient), the four scoring signals, and a full
mock run over the real dataset so the plumbing is trusted before it touches a GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offhand_eval.dataset import load_items, load_tools, tools_by_name
from offhand_eval.harness import run_eval
from offhand_eval.mock import build_mock_runner
from offhand_eval.parse import parse_tool_calls, parse_tool_calls_lenient
from offhand_eval.scoring import args_match, is_schema_valid, score_item

HERE = Path(__file__).resolve().parents[1]
TOOLS = load_tools(HERE / "tools.json")
BY_NAME = tools_by_name(TOOLS)
VALID = list(BY_NAME)


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


def test_lenient_recovers_name_prefixed_calls():
    # dropped <tool_call> wrapper, name leaked outside the JSON (the nf4 failure)
    assert parse_tool_calls('.create_note {"content": "buy milk"}') == []  # strict misses it
    lenient = parse_tool_calls_lenient('.create_note {"content": "buy milk"}', VALID)
    assert lenient and lenient[0]["name"] == "create_note"
    assert lenient[0]["arguments"]["content"] == "buy milk"

    lenient2 = parse_tool_calls_lenient('draft_email {"to": "jane", "subject": "Lunch"}', VALID)
    assert lenient2 and lenient2[0]["name"] == "draft_email"


def test_lenient_does_not_invent_calls_from_prose():
    assert parse_tool_calls_lenient("None of the tools can adjust screen brightness.", VALID) == []


def test_schema_validity():
    assert is_schema_valid({"name": "set_alarm", "arguments": {"time": "07:00"}}, BY_NAME)
    assert not is_schema_valid({"name": "set_alarm", "arguments": {"label": "x"}}, BY_NAME)  # missing required
    assert not is_schema_valid({"name": "launch_rocket", "arguments": {}}, BY_NAME)  # unknown tool
    assert not is_schema_valid(
        {"name": "toggle_flashlight", "arguments": {"state": "on", "color": "red"}}, BY_NAME
    )  # unknown param


def test_args_match_forms():
    assert args_match({"panel": "wifi"}, {"panel": ["wifi"]})
    assert args_match({"time": "7 AM"}, {"time": ["07:00", "7 am"]})
    assert args_match({"content": "Buy milk from the store"}, {"content": {"contains": "milk"}})
    assert not args_match({"content": "call the plumber"}, {"content": {"contains": "milk"}})
    # the recipient fix: a resolved address still matches a contains-gold
    assert args_match({"to": "boss@example.com"}, {"to": {"contains": "boss"}})


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
    """A mock that echoes the reference-correct call scores 100% strict."""
    items = load_items(HERE / "mini_eval.jsonl")
    outcome = run_eval(items, TOOLS, build_mock_runner(items, BY_NAME))
    strict = outcome["summary"]["strict"]
    assert strict["call_accuracy"] == 1.0, strict
    assert strict["schema_valid_rate"] == 1.0, strict
    assert strict["irrelevance_accuracy"] == 1.0, strict
    assert outcome["summary"]["format_recoverable"] == 0.0


def test_optional_arg_omission():
    # BFCL marks optional args with "" in the accepted list; omitting them is fine.
    gold = {"base": [10], "height": [5], "unit": ["units", ""]}
    assert args_match({"base": 10, "height": 5}, gold)              # unit omitted -> ok
    assert args_match({"base": 10, "height": 5, "unit": "units"}, gold)
    assert not args_match({"base": 10, "height": 5, "unit": "kg"}, gold)  # present but wrong
    assert not args_match({"height": 5}, gold)                     # required base missing


def test_convert_bfcl_and_score():
    import convert_bfcl

    q = {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Find the area of a triangle base 10 height 5."}]],
        "function": [{
            "name": "calculate_triangle_area",
            "description": "Area of a triangle.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer"},
                    "height": {"type": "integer"},
                    "unit": {"type": "string"},
                },
                "required": ["base", "height"],
            },
        }],
    }
    gt = {"id": "simple_0", "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}

    item = convert_bfcl.convert_call_item(q, gt)
    assert item["gold"]["name"] == "calculate_triangle_area"
    assert item["query"].startswith("Find the area")
    assert item["tools"][0]["type"] == "function"

    by_name = tools_by_name(item["tools"])
    call = {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
    s = score_item(item, [call], by_name)
    assert s.schema_valid and s.right_tool and s.right_args


def test_bootstrap_ci():
    from offhand_eval.stats import bootstrap_ci

    assert bootstrap_ci([]) == (None, None, None)
    m, lo, hi = bootstrap_ci([1, 1, 1, 1], n_boot=200)
    assert m == 1.0 and lo == 1.0 and hi == 1.0
    m, lo, hi = bootstrap_ci([1, 0] * 50, n_boot=500)
    assert 0.35 < m < 0.65 and lo <= m <= hi


def test_parse_label_and_memory():
    import report

    assert report.parse_label("bfcl_Qwen3-0.6B_fp16") == (0.6, "fp16")
    assert report.parse_label("bfcl_Qwen3-4B_nf4") == (4.0, "nf4")
    assert report.memory_gb(0.6, "fp16") == 1.2
    assert report.memory_gb(4.0, "nf4") == 2.0
    assert report.memory_gb(1.7, "fp16") == 3.4
    assert report.memory_gb(None, "fp16") is None


def test_summarize_file_synthetic(tmp_path=None):
    import json
    import tempfile
    import report

    item_call = {"id": "c1", "category": "single", "tools": TOOLS,
                 "gold": {"name": "set_alarm", "arguments": {"time": ["07:00"]}}}
    item_irr = {"id": "i1", "category": "irrelevance", "gold": {"no_call": True}}
    gold_by_id = {"c1": item_call, "i1": item_irr}

    results = {
        "label": "bfcl_Qwen3-0.6B_fp16", "model": "Qwen/Qwen3-0.6B",
        "results": [
            {"id": "c1", "raw": '<tool_call>{"name":"set_alarm","arguments":{"time":"07:00"}}</tool_call>'},
            {"id": "i1", "raw": "The capital of France is Paris."},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(results, fh)
        path = fh.name

    row = report.summarize_file(path, gold_by_id, TOOLS, n_boot=200)
    assert row["n_call"] == 1 and row["strict_acc"] == 1.0
    assert row["n_irr"] == 1 and row["irr_acc"] == 1.0
    assert row["memory_gb"] == 1.2 and row["precision"] == "fp16"
    assert row["format_gap"] == 0.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
