#!/usr/bin/env python3
"""Check a phone-task eval set before any model runs on it.

A new test set can quietly give or cost points. This catches the usual causes:

 - answer-key structure: unique ids, a category that matches the answer key, a
   known tool, only arguments that tool has, allowed enum values, numbers for
   number arguments, an HH:MM entry for alarm times (the format the tool asks
   for), and lowercase answers, since the scorer lowercases the model's value;
 - winnable keys: a model answering exactly the answer key scores 100% on every
   item, each scored on its own;
 - leakage: no question repeats inside the set, and none is the same as or
   close to a question from another eval set or from the training data, using
   the overlap rule of train/check_contamination.py.

Usage:
  python eval/check_eval_set.py eval/phone_eval.jsonl --other eval/mini_eval.jsonl \
      --train <v1>/train.jsonl <hard>/train_clean.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))

from check_contamination import normalize, query_overlaps  # noqa: E402
from offhand_eval.dataset import load_items, load_tools, tools_by_name  # noqa: E402
from offhand_eval.harness import run_eval  # noqa: E402
from offhand_eval.loto import HELD_OUT  # noqa: E402
from offhand_eval.mock import reference_output  # noqa: E402
from offhand_eval.runners import MockRunner  # noqa: E402

HHMM = re.compile(r"^\d{2}:\d{2}$")


def _answer_problem(arg: str, accepted, spec: dict) -> str | None:
    """Why an answer-key entry can't match the way it is meant to, or None."""
    if isinstance(accepted, dict):
        key = accepted.get("contains")
        if set(accepted) != {"contains"} or not isinstance(key, str) or not key.strip():
            return f'{arg}: a dict answer must be {{"contains": "..."}}'
        if key != key.strip().lower():
            return f"{arg}: contains key {key!r} must be lowercase and trimmed"
        return None
    if not isinstance(accepted, list) or not accepted:
        return f"{arg}: the answer must be a non-empty list or a contains check"
    if any(str(v) != str(v).strip().lower() for v in accepted):
        return f"{arg}: accepted values {accepted} must be lowercase and trimmed"
    if "enum" in spec and any(str(v) not in {str(e).lower() for e in spec["enum"]} for v in accepted):
        return f"{arg}: accepted values {accepted} are not all in the enum {spec['enum']}"
    if spec.get("type") in ("number", "integer"):
        try:
            [float(v) for v in accepted]
        except (TypeError, ValueError):
            return f"{arg}: accepted values {accepted} must all be numbers"
    return None


def structure_problems(items: list[dict], tools: dict[str, dict]) -> list[str]:
    """One line per answer-key mistake, as "<id>: <what is wrong>"."""
    problems, seen = [], set()
    for it in items:
        iid = it.get("id") or "<no id>"
        if iid in seen or iid == "<no id>":
            problems.append(f"{iid}: missing or repeated id")
        seen.add(iid)
        if not str(it.get("query") or "").strip():
            problems.append(f"{iid}: empty query")
        gold = it.get("gold") or {}
        if gold.get("no_call"):
            if it.get("category") != "irrelevance" or gold != {"no_call": True}:
                problems.append(f'{iid}: a no-call item needs category irrelevance and gold {{"no_call": true}} only')
            continue
        if it.get("category") != "single":
            problems.append(f"{iid}: a call item needs category single")
        tool = tools.get(gold.get("name"))
        if tool is None:
            problems.append(f"{iid}: unknown tool {gold.get('name')!r}")
            continue
        props = tool["function"]["parameters"].get("properties", {})
        args = gold.get("arguments")
        if not isinstance(args, dict) or not args:
            problems.append(f"{iid}: no checked argument")
            continue
        for arg, accepted in args.items():
            if arg not in props:
                problems.append(f"{iid}: {gold['name']} has no argument {arg!r}")
                continue
            why = _answer_problem(arg, accepted, props[arg])
            if why:
                problems.append(f"{iid}: {why}")
        times = args.get("time")
        if gold["name"] == "set_alarm" and isinstance(times, list) and not any(HHMM.match(str(v)) for v in times):
            problems.append(f"{iid}: set_alarm needs an HH:MM entry, the format the tool asks for")
    return problems


def unwinnable(items: list[dict], tools: list[dict]) -> list[str]:
    """Items a model answering exactly the answer key would still fail."""
    by_name = tools_by_name(tools)
    bad = []
    for it in items:
        try:
            runner = MockRunner(mapping={it["query"]: reference_output(it, by_name)})
            strict = run_eval([it], tools, runner)["summary"]["strict"]
        except Exception as exc:  # noqa: BLE001 - any failure here means the key can't be answered
            bad.append(f"{it.get('id')}: {type(exc).__name__}: {exc}")
            continue
        rate = strict["irrelevance_accuracy"] if it["gold"].get("no_call") else strict["call_accuracy"]
        if rate != 1.0:
            bad.append(f"{it.get('id')}: the answer key itself scores {rate}")
    return bad


def repeats(items: list[dict], others: list[dict]) -> list[str]:
    """Questions that repeat inside the set or match one from another set, after normalizing."""
    found, seen = [], {}
    other = {normalize(o["query"]): o.get("id") for o in others}
    for it in items:
        key = normalize(it["query"])
        if key in seen:
            found.append(f"{it['id']}: same question as {seen[key]}")
        elif key in other:
            found.append(f"{it['id']}: same question as {other[key]} in another set")
        seen.setdefault(key, it["id"])
    return found


def overlaps(items: list[dict], queries: list[str]) -> list[str]:
    """Items close to any of the given questions, by the contamination check's rule."""
    return [f"{it['id']}: {reason}" for it, reason in query_overlaps(items, queries)]


def summary(items: list[dict]) -> str:
    kinds = Counter("no call" if it["gold"].get("no_call") else it["gold"].get("name") for it in items)
    subtypes = Counter(("no call" if it["gold"].get("no_call") else it["gold"].get("name"), it.get("subtype", "-"))
                       for it in items)
    held = sum(1 for it in items if not it["gold"].get("no_call") and it["gold"].get("name") in HELD_OUT)
    lines = [f"{len(items)} items: {len(items) - kinds['no call']} call ({held} on held-out tools), {kinds['no call']} no call"]
    for kind, n in sorted(kinds.items(), key=lambda kv: (kv[0] == "no call", kv[0])):
        parts = ", ".join(f"{sub} {c}" for (k, sub), c in sorted(subtypes.items()) if k == kind)
        lines.append(f"  {kind:<22} {n:>4}   {parts}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check an eval set's answer keys, repeats and overlap with training")
    ap.add_argument("data", help="the eval set to check (jsonl)")
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--other", nargs="*", default=[], help="other eval sets its questions must not repeat")
    ap.add_argument("--train", nargs="*", default=[], help="training jsonl files its questions must not overlap")
    args = ap.parse_args()

    items = load_items(args.data)
    tools = load_tools(args.tools)
    others = [it for path in args.other for it in load_items(path)]
    train_queries = [it["query"] for path in args.train for it in load_items(path)]

    print(summary(items))
    checks = [
        ("answer-key problems", structure_problems(items, tools_by_name(tools))),
        ("answer keys a perfect model would still fail", unwinnable(items, tools)),
        ("repeated questions", repeats(items, others)),
        ("close to a question in another eval set", overlaps(items, [o["query"] for o in others])),
        (f"close to one of {len(train_queries)} training questions", overlaps(items, train_queries)),
    ]
    failed = False
    for title, found in checks:
        print(f"\n{title}: {len(found)}")
        for line in found[:60]:
            print(f"  {line}")
        failed = failed or bool(found)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
