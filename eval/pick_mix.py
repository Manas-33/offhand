#!/usr/bin/env python3
"""Pick the training mix to send to the BFCL test, by a rule fixed before the run.

Each candidate has a result on the validation set (make_bfcl_dev.py) and on the
phone set; the base model has a validation result too. The rule, written down
before any candidate was trained:

 1. A candidate must hold the phone set at 0.90 or better (v1's level).
 2. Among those, take the one whose worse BFCL half is best against base: the
    largest min(call - base call, no-call - base no-call) on the validation set.
    Zero or more means neither half is below base.
 3. Ties go to the higher call + no-call sum, then to the earlier candidate.

When no candidate holds the phone set, the rule runs on all of them and says so.
The chosen model directory is written to --out for the runner's test step.

Usage:
  python eval/pick_mix.py --base <dev_base.json> \
      --candidate ic25 <dev json> <phone json> <model dir> \
      --candidate ic50 <dev json> <phone json> <model dir> --out <pick.txt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_PHONE = 0.90


def halves(result: dict) -> tuple[float, float]:
    """(call accuracy, no-call accuracy) from a result file's strict summary."""
    strict = result["summary"]["strict"]
    return strict["call_accuracy"], strict["irrelevance_accuracy"]


def pick(base: tuple[float, float], candidates: list[dict], min_phone: float = MIN_PHONE) -> tuple[dict, bool]:
    """The chosen candidate, and whether the phone-set floor had to be dropped.

    Each candidate is {"name", "call", "no_call", "phone", ...}; this adds "worst_gap".
    """
    for c in candidates:
        c["worst_gap"] = min(c["call"] - base[0], c["no_call"] - base[1])
    pool = [c for c in candidates if c["phone"] >= min_phone]
    order = {id(c): k for k, c in enumerate(candidates)}
    best = max(pool or candidates,
               key=lambda c: (round(c["worst_gap"], 6), round(c["call"] + c["no_call"], 6), -order[id(c)]))
    return best, not pool


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick the mix to test, by the rule in this file's docstring")
    ap.add_argument("--base", required=True, help="the base model's validation-set result")
    ap.add_argument("--candidate", nargs=4, action="append", required=True,
                    metavar=("NAME", "DEV_JSON", "PHONE_JSON", "MODEL_DIR"))
    ap.add_argument("--min-phone", type=float, default=MIN_PHONE)
    ap.add_argument("--out", required=True, help="file that receives the chosen model directory")
    args = ap.parse_args()

    base = halves(_load(args.base))
    candidates = []
    for name, dev_json, phone_json, model_dir in args.candidate:
        call, no_call = halves(_load(dev_json))
        phone = _load(phone_json)["summary"]["strict"]["call_accuracy"]
        candidates.append({"name": name, "call": call, "no_call": no_call, "phone": phone, "model": model_dir})

    best, fallback = pick(base, candidates, args.min_phone)
    print(f"{'':<8} {'call':>7} {'no-call':>8} {'phone':>7} {'worse half vs base':>19}")
    print(f"{'base':<8} {base[0]:>7.3f} {base[1]:>8.3f}")
    for c in candidates:
        mark = "  <- picked" if c is best else ""
        print(f"{c['name']:<8} {c['call']:>7.3f} {c['no_call']:>8.3f} {c['phone']:>7.3f} {c['worst_gap']:>+19.3f}{mark}")
    if fallback:
        print(f"note: no candidate held the phone set at {args.min_phone}; picked among all of them")
    Path(args.out).write_text(best["model"] + "\n", encoding="utf-8")
    print(f"picked {best['name']}: {best['model']} -> {args.out}")


if __name__ == "__main__":
    main()
