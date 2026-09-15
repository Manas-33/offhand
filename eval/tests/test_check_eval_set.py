"""Eval-set checker tests, no model needed: `python eval/tests/test_check_eval_set.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_eval_set as ces
from offhand_eval.dataset import load_tools, tools_by_name

TOOLS = load_tools(Path(__file__).resolve().parents[1] / "tools.json")
BY_NAME = tools_by_name(TOOLS)


def _call(iid, query, name, arguments):
    return {"id": iid, "category": "single", "query": query, "gold": {"name": name, "arguments": arguments}}


def _no_call(iid, query):
    return {"id": iid, "category": "irrelevance", "query": query, "gold": {"no_call": True}}


GOOD = [
    _call("a1", "wake me at 6:45 am", "set_alarm", {"time": ["06:45", "6:45", "6:45 am"]}),
    _call("t1", "an hour and a half timer please", "set_timer", {"duration_minutes": ["90", "90.0"]}),
    _call("s1", "open bluetooth settings", "open_settings", {"panel": ["bluetooth"]}),
    _call("m1", "play some coldplay", "play_music", {"query": {"contains": "coldplay"}}),
    _call("r1", "text mom that dinner is at 7", "draft_sms", {"recipient": {"contains": "mom"}}),
    _no_call("n1", "how far away is the moon?"),
]


def test_a_clean_set_passes_every_check():
    assert ces.structure_problems(GOOD, BY_NAME) == []
    assert ces.unwinnable(GOOD, TOOLS) == []
    assert ces.repeats(GOOD, []) == []


def test_structure_catches_each_kind_of_mistake():
    bad = [
        _call("x1", "q1", "set_alrm", {"time": ["07:00"]}),                 # unknown tool
        _call("x2", "q2", "set_alarm", {"when": ["07:00"]}),                 # unknown argument
        _call("x3", "q3", "open_settings", {"panel": ["airplane"]}),        # not in the enum
        _call("x4", "q4", "draft_sms", {"recipient": {"contains": "Mom"}}),  # uppercase key
        _call("x5", "q5", "set_timer", {"duration_minutes": ["ten"]}),       # not a number
        _call("x6", "q6", "set_alarm", {"time": ["7am"]}),                   # no HH:MM entry
        {**_no_call("x7", "q7"), "category": "single"},                      # category and key disagree
        _call("x1", "q8", "set_timer", {"duration_minutes": ["5", "5.0"]}),  # repeated id
    ]
    found = ces.structure_problems(bad, BY_NAME)
    for iid in ("x2", "x3", "x4", "x5", "x6", "x7"):
        assert any(line.startswith(iid + ":") for line in found), iid
    assert sum(line.startswith("x1:") for line in found) == 2  # the unknown tool and the repeated id


def test_unwinnable_flags_an_answer_key_nothing_can_match():
    item = _call("u1", "set a timer", "set_timer", {"duration_minutes": []})
    assert [line.split(":")[0] for line in ces.unwinnable([item], TOOLS)] == ["u1"]


def test_repeats_inside_the_set_and_against_another_set():
    a = _no_call("p1", "What's the capital of Peru?")
    b = _no_call("p2", "what's the capital of peru")
    c = _no_call("p3", "tell me a fun fact")
    found = ces.repeats([a, b, c], [_no_call("old", "Tell me a fun fact!")])
    assert found == ["p2: same question as p1", "p3: same question as old in another set"]


def test_overlaps_follow_the_contamination_rule():
    items = [_no_call("o1", "what is the tallest mountain in the whole world")]
    assert ces.overlaps(items, ["What is the tallest mountain in the whole world?"])
    assert not ces.overlaps(items, ["how do plants make food from sunlight"])


def test_summary_counts_held_out_tools():
    text = ces.summary(GOOD)
    assert text.startswith("6 items: 5 call (1 on held-out tools), 1 no call")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all check_eval_set tests passed.")


if __name__ == "__main__":
    _run_all()
