#!/usr/bin/env python3
"""Drop teacher-labeled hard examples whose answers would teach the wrong thing.

gen_data.py's agreement rule only checks the call decision: a no-call item
survives whenever the teacher's reply contains no parsable tool call. The first
real run showed that is not enough. A small share of the surviving replies
teach behavior an agent must never have:

 - claiming an action it did not take ("I will cancel your 7 a.m. alarm",
   "Sure, I can help you snooze the alarm"), or emitting a made-up tag
   ("<delete_alarm>") in place of a decline,
 - stating live facts it cannot know without the tool it lacks
   ("It is currently 2:30 AM in Berlin"),
 - treating the paraphraser's own commentary as a request
   ("First, I'll think of different ways to phrase ...").

The paraphrase pass also produces exact duplicates. This drops all four and
prints every dropped item, so the rules stay auditable.

Usage:
  python train/filter_hard.py --train <hard>/train_clean.jsonl --out <hard>/train_final.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

META_QUERY = re.compile(
    r"^(first|next|then|finally|alternatively)\b"
    r"|ways to (phrase|say|ask)|rephras|phrasing|more (polite|casual|formal|natural)"
    r"|\bI could (make|say|use|phrase|add)\b|\bI'll (think|rewrite|rephrase|try)\b",
    re.I,
)
FAKE_TAG = re.compile(r"<[A-Za-z_]+>")
# "I can help you ..." only counts as a claim when it is not the offer after a
# decline: "I can't check the weather, but I can help you find the sunset time"
# points at what a listed tool really does, which is the behavior we want.
ACTION_CLAIM = re.compile(
    r"\bI('ll| will| have|'ve) (cancel|delete|snooze|set|create|add|stop|pause|turn|remove|mark|read"
    r"|check|find|play|send|book|order|call|lock|unlock|start|reserve|pay|post)"
    r"|(?<!but )\bI can help you (cancel|delete|snooze|stop|pause|remove|mark|read|check|find)\b"
    r"|\b(has|have) been (set|cancel|delet|snooz|remov|added|mark|stopp|paus|order|book|sent|paid)",
    re.I,
)
# A computed answer ("70°F is 21.1°C") is fine; a reading only a live source
# could give ("It is currently 2:30 AM in Berlin") is not.
LIVE_FACT = re.compile(
    r"\b(it is|it's) currently\b"
    r"|\b(currently|right now|at the moment|as of now)\b[^.]*\d"
    r"|\bthe current (price|rate|exchange rate|temperature|time|score|value)\b[^.]*\d"
    r"|\btrading at\b[^.]*\d",
    re.I,
)


def reasons(item: dict) -> list[str]:
    """Why an item should be dropped; empty when it is fine to train on."""
    out = []
    if META_QUERY.search(item["query"]):
        out.append("paraphraser commentary as the query")
    if item["intended_tool"] is None:  # the rules below read a text answer, never a tool call
        target = item["target"]
        if FAKE_TAG.search(target):
            out.append("made-up tag instead of a decline")
        if ACTION_CLAIM.search(target):
            out.append("claims an action it did not take")
        if LIVE_FACT.search(target):
            out.append("states a live fact without the tool")
    return out


def filter_items(items: list[dict]) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    """Split items into kept and dropped (with reasons); duplicates within a kind drop too."""
    kept, dropped, seen = [], [], set()
    for item in items:
        why = reasons(item)
        key = (item.get("kind"), " ".join(item["query"].lower().split()))
        if not why and key in seen:
            why = ["duplicate question"]
        if why:
            dropped.append((item, why))
            continue
        seen.add(key)
        kept.append(item)
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop hard examples that would teach the wrong thing")
    ap.add_argument("--train", required=True, help="generated hard-set jsonl (after the contamination check)")
    ap.add_argument("--out", required=True, help="where to write the items that pass")
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.train, encoding="utf-8") if line.strip()]
    kept, dropped = filter_items(items)
    by_reason = Counter(r for _, why in dropped for r in why)
    print(f"kept {len(kept)} of {len(items)}; dropped {len(dropped)}: {dict(by_reason)}")
    for item, why in dropped:
        if why != ["duplicate question"]:
            print(f"  [{item.get('kind')}] {item['query'][:60]!r} -> {item['target'][:90]!r}  ({'; '.join(why)})")
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in kept:
            fh.write(json.dumps(item) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
