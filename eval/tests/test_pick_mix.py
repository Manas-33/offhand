"""Mix-picking rule tests, no model needed: `python eval/tests/test_pick_mix.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pick_mix

BASE = (0.80, 0.84)  # (call, no-call) on the validation set


def _c(name, call, no_call, phone=0.925):
    return {"name": name, "call": call, "no_call": no_call, "phone": phone}


def test_prefers_the_candidate_whose_worse_half_is_closest_to_base():
    # shaped like v2 (strong call, weak no-call), phone-only (the reverse) and a mix in between
    cands = [_c("callish", 0.82, 0.75), _c("balanced", 0.79, 0.83), _c("declinish", 0.74, 0.86)]
    best, fallback = pick_mix.pick(BASE, cands)
    assert best["name"] == "balanced" and not fallback
    assert round(best["worst_gap"], 3) == -0.01


def test_phone_floor_excludes_a_candidate():
    cands = [_c("weak_phone", 0.80, 0.84, phone=0.85), _c("ok", 0.78, 0.82)]
    best, fallback = pick_mix.pick(BASE, cands)
    assert best["name"] == "ok" and not fallback


def test_falls_back_to_all_when_none_holds_the_phone_set():
    cands = [_c("a", 0.78, 0.82, phone=0.80), _c("b", 0.80, 0.84, phone=0.85)]
    best, fallback = pick_mix.pick(BASE, cands)
    assert best["name"] == "b" and fallback


def test_ties_go_to_the_higher_total_then_the_earlier_candidate():
    best, _ = pick_mix.pick(BASE, [_c("a", 0.79, 0.84), _c("b", 0.79, 0.85)])
    assert best["name"] == "b"
    best, _ = pick_mix.pick(BASE, [_c("a", 0.79, 0.84), _c("c", 0.80, 0.83)])
    assert best["name"] == "a"


def test_halves_reads_the_strict_summary():
    result = {"summary": {"strict": {"call_accuracy": 0.8, "irrelevance_accuracy": 0.9},
                          "lenient": {"call_accuracy": 0.85, "irrelevance_accuracy": 0.9}}}
    assert pick_mix.halves(result) == (0.8, 0.9)


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all pick_mix tests passed.")


if __name__ == "__main__":
    _run_all()
