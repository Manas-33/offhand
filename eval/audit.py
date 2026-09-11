#!/usr/bin/env python3
"""Audit failing eval items and print a failure taxonomy per result file.

Recomputes strict + lenient parses from each result's stored ``raw`` output and
re-scores against the golds in a dataset file (joined by id). Works on any
results JSON that has ``query`` + ``raw``, regardless of the harness version
that produced it, and reflects the latest scoring rules and per-item tools.

Categories for a call-expected item:
  ok            strict parse already correct
  format_break  right intent, but only the lenient parser could read it
                (dropped <tool_call> wrapper / name outside JSON). This is what
                constrained decoding recovers.
  wrong_tool    a call was made, wrong function
  wrong_arg     right function, wrong or missing argument (inspect for
                gold-too-strict harness bugs)
  no_call       no parseable call at all (answered in prose / refused)
For an irrelevance item:
  spurious      wrongly called a tool

Usage:
  python eval/audit.py                                   # audit eval/results/*.json vs mini_eval
  python eval/audit.py eval/results/bfcl_*.json --data eval/data/bfcl.jsonl
"""

from __future__ import annotations

import argparse
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


def categorize(item, raw, global_tools):
    tools = item.get("tools") or global_tools
    by_name = tools_by_name(tools)
    strict = parse_tool_calls(raw)
    lenient = parse_tool_calls_lenient(raw, list(by_name))
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


def audit_file(path, gold_by_id, global_tools, show):
    data = json.load(open(path))
    label = data.get("label", Path(path).stem)
    model = data.get("model", "?")

    counts = Counter()
    misses = []
    n_call = ok_strict = ok_lenient = 0

    for r in data.get("results", []):
        item = gold_by_id.get(r["id"])
        if item is None:
            continue
        cat, strict, lenient = categorize(item, r.get("raw", ""), global_tools)
        counts[cat] += 1
        if item["gold"].get("no_call"):
            continue
        n_call += 1
        if cat == "ok":
            ok_strict += 1
            ok_lenient += 1
        elif cat == "format_break":
            ok_lenient += 1
        if cat != "ok":
            misses.append((r["id"], cat, item["gold"], strict, lenient, r.get("raw", "")))

    print(f"\n================ {label}  ({model}) ================")
    if n_call:
        print(f"  strict  call_accuracy: {ok_strict}/{n_call} = {ok_strict / n_call:.3f}")
        print(f"  lenient call_accuracy: {ok_lenient}/{n_call} = {ok_lenient / n_call:.3f}")
        rec = ok_lenient - ok_strict
        print(f"  format-recoverable   : {rec} items ({rec / n_call:+.3f})")
    print("  taxonomy: " + ", ".join(f"{c}={counts.get(c, 0)}" for c in CATEGORIES))

    for id_, cat, gold, strict, lenient, raw in misses[:show]:
        pred = lenient or strict
        print(f"\n  [{id_}] {cat}")
        print(f"    gold: {json.dumps(gold)[:200]}")
        print(f"    pred: {json.dumps(pred[0]) if pred else 'NO CALL'}")
        print(f"    raw : {textwrap.shorten(raw.strip().replace(chr(10), ' '), width=150)}")
    if len(misses) > show:
        print(f"\n  ... and {len(misses) - show} more misses (raise --show to see)")
    return counts


def main():
    ap = argparse.ArgumentParser(description="Audit eval result files into a failure taxonomy")
    ap.add_argument("results", nargs="*", help="result JSON files (default: eval/results/*.json)")
    ap.add_argument("--data", default=str(HERE / "mini_eval.jsonl"),
                    help="dataset providing golds + per-item tools, joined by id")
    ap.add_argument("--tools", default=str(HERE / "tools.json"),
                    help="fallback tools for items without their own")
    ap.add_argument("--show", type=int, default=15, help="max miss transcripts to print per file")
    args = ap.parse_args()

    paths = args.results or sorted(glob.glob(str(HERE / "results" / "*.json")))
    if not paths:
        print("no result files found (pass paths, or put JSON in eval/results/)")
        return

    global_tools = load_tools(args.tools)
    gold_by_id = {it["id"]: it for it in load_items(args.data)}

    total = Counter()
    for path in paths:
        total.update(audit_file(path, gold_by_id, global_tools, args.show))

    print("\n================ TOTAL across files ================")
    print("  " + ", ".join(f"{c}={total.get(c, 0)}" for c in CATEGORIES))


if __name__ == "__main__":
    main()
