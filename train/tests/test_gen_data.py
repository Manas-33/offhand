"""Verify the generator's hygiene with the mock teacher, no model needed:
`python train/tests/test_gen_data.py`."""

import json
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_data

PHONE = gen_data.load_tools(Path(gen_data.HERE).parent / "eval" / "tools.json")
GENERAL = gen_data.load_tools(Path(gen_data.HERE) / "general_tools.json")
PHONE_NAMES = [t["function"]["name"] for t in PHONE]
GENERAL_NAMES = [t["function"]["name"] for t in GENERAL]
ALL_BY_NAME = gen_data.tools_by_name(PHONE + GENERAL)
HARD_KINDS = {"info_call", "info_unlisted", "near_miss_phone", "near_miss_general", "unserviceable"}


def _generate():
    by_name = gen_data.tools_by_name(gen_data.load_tools(Path(gen_data.HERE).parent / "eval" / "tools.json"))
    rng = random.Random(0)
    train, heldout = [], []
    for i, seed in enumerate(gen_data.build_seeds(30, rng)):
        item = gen_data.label_and_validate(seed, by_name, gen_data.MockTeacher(), rng)
        if item is None:
            continue
        item["id"] = f"g{i}"
        (heldout if seed["intended_tool"] in gen_data.HELD_OUT else train).append(item)
    return by_name, train, heldout


def test_leave_one_tool_out_split():
    _, train, heldout = _generate()
    train_tools = {t["intended_tool"] for t in train if t["intended_tool"]}
    # held-out tools never appear as a training target
    assert not (train_tools & set(gen_data.HELD_OUT)), train_tools
    # the held-out file contains only held-out tools
    assert heldout and all(h["intended_tool"] in gen_data.HELD_OUT for h in heldout)


def test_labels_are_clean():
    by_name, train, heldout = _generate()
    for item in train + heldout:
        if item["intended_tool"] is None:
            assert gen_data.parse_tool_calls(item["target"]) == []  # no-call target has no call
            continue
        calls = gen_data.parse_tool_calls(item["target"])
        assert calls and calls[0]["name"] == item["intended_tool"]
        assert gen_data.is_schema_valid(calls[0], by_name)


def test_deployment_condition_and_varied_lists():
    _, train, heldout = _generate()
    # held-out tools still show up in some eval prompts (a tool added to the app)
    assert any(set(h["tools_listed"]) & set(gen_data.HELD_OUT) for h in heldout)
    # the listed-tool set is not constant across examples
    assert len({tuple(sorted(t["tools_listed"])) for t in train}) > 1
    # a call item always lists its own tool
    for item in train + heldout:
        if item["intended_tool"]:
            assert item["intended_tool"] in item["tools_listed"]


def test_shorten_no_call():
    s = gen_data.shorten_no_call
    # a one-sentence answer is left intact
    assert s("The capital of France is Paris.") == "The capital of France is Paris."
    # only the first sentence is kept
    assert s("Paris is the capital. It is in France.") == "Paris is the capital."
    # newlines never survive into a target
    assert s("First line.\nSecond line.") == "First line."
    # a runaway single sentence is capped at a word boundary
    long_one = "word " * 60 + "end."
    out = s(long_one)
    assert len(out) <= gen_data.NO_CALL_MAX_CHARS and " " in out and "\n" not in out


def test_paraphrase_scales_up_and_preserves_routing():
    by_name = gen_data.tools_by_name(gen_data.load_tools(Path(gen_data.HERE).parent / "eval" / "tools.json"))
    rng = random.Random(0)
    teacher = gen_data.MockTeacher()
    seeds = [s for s in gen_data.build_seeds(10, rng) if s["intended_tool"]]
    expanded = gen_data.expand_with_paraphrases(seeds, teacher, 6, rng)
    assert len(expanded) > len(seeds)  # paraphrasing multiplied the set

    # the mock's lead-ins must not change which tool a query routes to, so
    # (almost) every paraphrase should survive labeling against its seed tool
    kept = [gen_data.label_and_validate(s, by_name, teacher, rng) for s in expanded]
    kept = [k for k in kept if k]
    assert len(kept) >= 0.9 * len(expanded), (len(kept), len(expanded))
    for item in kept:
        calls = gen_data.parse_tool_calls(item["target"])
        assert calls and calls[0]["name"] == item["intended_tool"]


def test_added_tools_are_trained():
    # the four newly-templated tools should show up as training targets
    _, train, _ = _generate()
    train_tools = {t["intended_tool"] for t in train if t["intended_tool"]}
    for tool in ("open_settings", "draft_email", "play_music", "set_reminder"):
        assert tool in train_tools, (tool, train_tools)


# --- hard call/no-call families ------------------------------------------------

class _Oracle:
    """A teacher that labels hard seeds correctly: it calls a seed's intended tool
    (every required argument filled) when that tool is listed, and answers
    everything else in text."""

    def __init__(self, seeds):
        self.intended = {s["query"]: s["intended_tool"] for s in seeds if s["intended_tool"]}

    def generate(self, query, tools):
        name = self.intended.get(query)
        if name and name in {t["function"]["name"] for t in tools}:
            args = gen_data.MockTeacher.fill_required(ALL_BY_NAME[name]["function"]["parameters"])
            return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'
        return "Sorry, I can't do that with the tools I have."


def _hard(info_per_tool=3, unlisted_frac=0.5):
    rng = random.Random(0)
    seeds = gen_data.build_hard_seeds(info_per_tool, unlisted_frac, rng)
    pools = gen_data.hard_pools(PHONE_NAMES, GENERAL_NAMES)
    oracle = _Oracle(seeds)
    items = [gen_data.label_hard(s, ALL_BY_NAME, oracle, rng, pools) for s in seeds]
    return seeds, [i for i in items if i]


def test_general_tools_are_well_formed():
    assert len(GENERAL_NAMES) == len(set(GENERAL_NAMES)), "duplicate informational tool name"
    assert not set(GENERAL_NAMES) & set(PHONE_NAMES), "informational tools must not shadow phone tools"
    for tool in GENERAL:
        params = tool["function"]["parameters"]
        assert set(params["required"]) <= set(params["properties"]), tool["function"]["name"]
    # the templates cover exactly the informational tools
    assert set(gen_data.INFO_TEMPLATES) == set(GENERAL_NAMES)
    assert set(gen_data.GENERAL_NEAR_MISSES) <= set(GENERAL_NAMES)


def test_hard_templates_only_use_known_slots():
    templates = [t for ts in gen_data.INFO_TEMPLATES.values() for t in ts]
    templates += [t for ts in gen_data.GENERAL_NEAR_MISSES.values() for t in ts]
    for template in templates:
        for _, field, _, _ in string.Formatter().parse(template):
            if field:
                assert field in gen_data.INFO_SLOTS, (field, template)


def test_hard_families_never_touch_held_out_tools():
    seeds, items = _hard()
    for seed in seeds:
        assert seed["intended_tool"] not in gen_data.HELD_OUT
        assert seed.get("related_tool") not in gen_data.HELD_OUT
    assert items
    for item in items:
        assert not set(item["tools_listed"]) & set(gen_data.HELD_OUT), item
    # only trained phone tools get near misses
    assert not set(gen_data.PHONE_NEAR_MISSES) & set(gen_data.HELD_OUT)


def test_hard_labels_are_clean():
    _, items = _hard()
    assert {i["kind"] for i in items} == HARD_KINDS
    for item in items:
        calls = gen_data.parse_tool_calls(item["target"])
        if item["kind"] == "info_call":
            assert calls and calls[0]["name"] == item["intended_tool"]
            assert item["intended_tool"] in item["tools_listed"]
            assert gen_data.is_schema_valid(calls[0], ALL_BY_NAME)
        else:
            assert item["intended_tool"] is None and calls == [] and item["target"]
        if item["kind"].startswith("near_miss"):
            assert item["related_tool"] in item["tools_listed"]
        if item["kind"] == "info_unlisted":
            assert not set(item["tools_listed"]) & set(GENERAL_NAMES)


def test_near_miss_dropped_when_teacher_calls():
    rng = random.Random(0)
    pools = gen_data.hard_pools(PHONE_NAMES, GENERAL_NAMES)
    seed = {"kind": "near_miss_phone", "intended_tool": None, "related_tool": "set_alarm",
            "query": "what alarms do I have set?", "pool": "seen_phone"}
    # the mock routes any "alarm" query to set_alarm: a teacher that falls for the near miss
    assert gen_data.label_hard(seed, ALL_BY_NAME, gen_data.MockTeacher(), rng, pools) is None


def test_info_twin_differs_only_in_listing():
    _, items = _hard(unlisted_frac=1.0)
    calls = {i["query"]: i for i in items if i["kind"] == "info_call"}
    twins = [i for i in items if i["kind"] == "info_unlisted" and i["query"] in calls]
    assert twins  # the same question: called with its tool listed, answered in text without it
    for twin in twins:
        assert calls[twin["query"]]["intended_tool"] not in twin["tools_listed"]


def test_paraphrase_keeps_hard_seed_fields():
    rng = random.Random(0)
    seeds = gen_data.build_hard_seeds(1, 0.0, rng)[:10]
    expanded = gen_data.expand_with_paraphrases(seeds, gen_data.MockTeacher(), 3, rng)
    assert len(expanded) > len(seeds)
    for item in expanded:
        assert item["kind"] in HARD_KINDS and item["pool"] in ("info", "seen_phone")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all gen_data tests passed.")


if __name__ == "__main__":
    _run_all()
