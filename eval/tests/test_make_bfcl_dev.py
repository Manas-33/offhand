"""Validation-set builder tests, no model or downloads: `python eval/tests/test_make_bfcl_dev.py`.

The fixtures mimic BFCL live items: a question is a list of turns, each a list
of messages, and every item carries its own function list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_bfcl_dev as dev


def _q(qid, text, n_tools=1, before=()):
    tools = [{"name": f"tool_{qid}_{k}", "description": "d",
              "parameters": {"type": "dict", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}
             for k in range(n_tools)]
    return {"id": qid, "question": [[*before, {"role": "user", "content": text}]], "function": tools}


def _a(qid, n_calls=1):
    return {"id": qid, "ground_truth": [{f"tool_{qid}_0": {"x": [1]}}] * n_calls}


def test_call_items_keep_single_message_single_call_questions():
    qs = [_q("s0", "book a table for two"),
          _q("s1", "what is up", before=[{"role": "system", "content": "be brief"}]),
          _q("s2", "two calls please"),
          _q("s3", "an answer key nothing can match"),
          _q("m0", "pick one of these", n_tools=3),
          _q("m1", "too many to choose from", n_tools=5)]
    ans = [_a("s0"), _a("s1"), _a("s2", n_calls=2), {"id": "s3", "ground_truth": [{"tool_s3_0": {"x": []}}]},
           _a("m0"), _a("m1")]
    assert [it["id"] for it in dev.call_items(qs, ans, 1, 1)] == ["s0"]
    assert [it["id"] for it in dev.call_items(qs, ans, 2, 4)] == ["m0"]
    item = dev.call_items(qs, ans, 1, 1)[0]
    assert item["category"] == "single" and item["gold"] == {"name": "tool_s0_0", "arguments": {"x": [1]}}
    assert item["tools"][0]["type"] == "function" and item["query"] == "book a table for two"


def test_no_call_items_need_one_tool_and_one_message():
    two_turns = {**_q("i4", "b"), "question": [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]}
    qs = [_q("i0", "tell me a story"), _q("i1", "hmm", n_tools=2), _q("i2", "nothing", n_tools=0),
          _q("i3", "again", before=[{"role": "assistant", "content": "earlier reply"}]), two_turns]
    items = dev.no_call_items(qs)
    assert [it["id"] for it in items] == ["i0"]
    assert items[0]["gold"] == {"no_call": True} and items[0]["category"] == "irrelevance"


def test_build_dev_drops_overlaps_and_training_tools():
    simple = dev.call_items([_q("s0", "book a table for two at the thai place tonight"),
                             _q("s1", "convert ten dollars to euros right now please")], [_a("s0"), _a("s1")], 1, 1)
    no_call = dev.no_call_items([_q("i0", "tell me a story about dragons"), _q("i1", "what is love")])
    no_call[1]["tools"][0]["function"]["name"] = "set_alarm"  # a tool the training data uses
    items, log = dev.build_dev(simple, [], no_call, ["Convert ten dollars to euros right now please"],
                               {"set_alarm"}, n_multiple=10, n_no_call=10, seed=0)
    assert [it["id"] for it in items] == ["s0", "i0"]
    assert len(log) == 3 and "1 overlapping" in log[0] and "1 offering a training tool" in log[2]


def test_sampling_is_seeded_and_keeps_file_order():
    pool = dev.no_call_items([_q(f"i{k}", f"question number {k} about something else") for k in range(20)])
    first, _ = dev.build_dev([], [], pool, [], set(), n_multiple=0, n_no_call=5, seed=1)
    again, _ = dev.build_dev([], [], pool, [], set(), n_multiple=0, n_no_call=5, seed=1)
    assert first == again and len(first) == 5
    ids = [int(it["id"][1:]) for it in first]
    assert ids == sorted(ids)
    everything, _ = dev.build_dev([], [], pool, [], set(), n_multiple=0, n_no_call=50, seed=1)
    assert everything == pool  # asking for more than exist keeps them all


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all make_bfcl_dev tests passed.")


if __name__ == "__main__":
    _run_all()
