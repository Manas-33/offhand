#!/usr/bin/env python3
"""Audit failing eval items and print a failure taxonomy per result file.

Recomputes strict + lenient parses from each result's stored ``raw`` output and
re-scores against the CURRENT golds in mini_eval.jsonl (joined by id). So it
works on any results JSON that has ``query`` + ``raw``, regardless of the
harness version that produced it, and it reflects the latest scoring rules.

Categories for a call-expected item:
  ok            strict parse already correct
  format_break  right intent, but only the lenient parser could read it
                (dropped <tool_call> wrapper / name outside JSON). This is what
                constrained decoding recovers.
  wrong_tool    a call was made, wrong function
  wrong_arg     right function, wrong or missing argument (inspect these for
                gold-too-strict harness bugs)
  no_call       no parseable call at all (answered in prose / refused)
For an irrelevance item:
  spurious      wrongly called a tool

Usage:
  python eval/audit.py                    # audits eval/results/*.json
  python eval/audit.py a.json b.json ...  # specific files
"""

from __future__ import annotations

import glob
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offhand_eval.dataset import load_items, load_tools, tools_by_name  # noqa: E402
from offhand_eval.parse import parse_tool_calls, parse_tool_calls_lenient  # noqa: E402
from offhand_eval.scoring import score_item  # noqa: E402

HERE = Path(__file__).resolve().parent
CATEGORIES = ["format_break", "wrong_tool", "wrong_arg", "no_call", "spurious"]


def categorize(item, raw, by_name, valid_names):
    strict = parse_tool_calls(raw)
    lenient = parse_tool_calls_lenient(raw, valid_names)
    s = score_item(item, strict, by_name)

    if not s.expected_call:
        return ("ok" if s.no_call_correct else "spurious"), strict, lenient
    if s.right_args:
        return "ok", strict, lenient
    if score_item(item, lenient, by_name).right_args:
        return "format_break", strict, lenient
    if lenient:
        return ("wrong_tool" if lenient[0].get("name") != item["gold"]["name"] else "wrong_arg"), strict, lenient
    return "no_call", strict, lenient


def audit_file(path, gold_by_id, by_name, valid_names):
    data = json.load(open(path))
    label = data.get("label", Path(path).stem)
    model = data.get("model", "?")
    rows = data.get("results", [])

    counts = Counter()
    misses = []
    n_call = n_call_ok_strict = n_call_ok_lenient = 0

    for r in rows:
        item = gold_by_id.get(r["id"])
        if item is None:
            continue
        raw = r.get("raw", "")
        cat, strict, lenient = categorize(item, raw, by_name, valid_names)
        counts[cat] += 1
        if item["gold"].get("no_call"):
            continue
        n_call += 1
        if cat == "ok":
            n_call_ok_strict += 1
            n_call_ok_lenient += 1
        elif cat == "format_break":
            n_call_ok_lenient += 1
        if cat not in ("ok",):
            misses.append((r["id"], cat, item["gold"], strict, lenient, raw))

    print(f"\n================ {label}  ({model}) ================")
    if n_call:
        print(f"  strict  call_accuracy: {n_call_ok_strict}/{n_call} = {n_call_ok_strict / n_call:.3f}")
        print(f"  lenient call_accuracy: {n_call_ok_lenient}/{n_call} = {n_call_ok_lenient / n_call:.3f}")
        recover = n_call_ok_lenient - n_call_ok_strict
        print(f"  format-recoverable   : {recover} items ({recover / n_call:+.3f})")
    print("  taxonomy: " + ", ".join(f"{c}={counts.get(c, 0)}" for c in CATEGORIES))

    for id_, cat, gold, strict, lenient, raw in misses:
        pred = (lenient or strict)
        pred_str = json.dumps(pred[0]) if pred else "NO CALL"
        gold_str = json.dumps(gold)
        print(f"\n  [{id_}] {cat}")
        print(f"    gold: {gold_str}")
        print(f"    pred: {pred_str}")
        print(f"    raw : {textwrap.shorten(raw.strip().replace(chr(10), ' '), width=150)}")
    return counts


def main():
    paths = sys.argv[1:] or sorted(glob.glob(str(HERE / "results" / "*.json")))
    if not paths:
        print("no result files found (pass paths, or put JSON in eval/results/)")
        return

    tools = load_tools(HERE / "tools.json")
    by_name = tools_by_name(tools)
    valid_names = list(by_name.keys())
    gold_by_id = {it["id"]: it for it in load_items(HERE / "mini_eval.jsonl")}

    total = Counter()
    for path in paths:
        total.update(audit_file(path, gold_by_id, by_name, valid_names))

    print("\n================ TOTAL across files ================")
    print("  " + ", ".join(f"{c}={total.get(c, 0)}" for c in CATEGORIES))


if __name__ == "__main__":
    main()
