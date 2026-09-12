"""Verify the generator's hygiene with the mock teacher, no model needed:
`python train/tests/test_gen_data.py`."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gen_data


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


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all gen_data tests passed.")


if __name__ == "__main__":
    _run_all()
