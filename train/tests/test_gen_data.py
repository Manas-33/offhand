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


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all gen_data tests passed.")


if __name__ == "__main__":
    _run_all()
