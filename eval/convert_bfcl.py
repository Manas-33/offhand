#!/usr/bin/env python3
"""Convert BFCL (Berkeley Function Calling Leaderboard) items into the Offhand
harness format: one ``{id, category, query, tools, gold}`` object per line.

Each BFCL item carries its own ``function`` list, which becomes the item's
``tools`` (so the model sees exactly the functions BFCL intended). Ground truth
for a call category comes from the matching possible_answer file; irrelevance
items have no ground truth (the correct behavior is to not call a tool).

Scope: single-call categories (``simple``) and ``irrelevance``. Multi-call
categories (multiple / parallel) need multi-call scoring and are skipped for now
(the CLI reports how many it skipped).

Get the BFCL files from Hugging Face, e.g.:
  base=https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main
  wget $base/BFCL_v3_simple.json
  wget $base/possible_answer/BFCL_v3_simple.json -O BFCL_v3_simple_answers.json
  wget $base/BFCL_v3_irrelevance.json

Usage:
  python eval/convert_bfcl.py --category simple \
      --questions BFCL_v3_simple.json --answers BFCL_v3_simple_answers.json \
      --out eval/data/bfcl_simple.jsonl
  python eval/convert_bfcl.py --category irrelevance \
      --questions BFCL_v3_irrelevance.json --out eval/data/bfcl_irrelevance.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wrap_tools(functions: list[dict]) -> list[dict]:
    """BFCL functions are bare {name, description, parameters}; wrap for the chat template."""
    return [{"type": "function", "function": fn} for fn in functions]


def first_user_query(question) -> str:
    """BFCL 'question' is a list of turns (list of message dicts). Take the first user message."""
    for turn in question:
        for msg in turn:
            if msg.get("role") == "user":
                return msg["content"]
    return question[0][0]["content"]


def convert_call_item(q_item: dict, gt_item: dict) -> dict:
    """Convert a single-call BFCL item + its ground truth into a harness item."""
    ground_truth = gt_item["ground_truth"]
    call = ground_truth[0]                 # single-call category: exactly one
    name = next(iter(call))
    return {
        "id": q_item["id"],
        "category": "single",
        "query": first_user_query(q_item["question"]),
        "tools": wrap_tools(q_item["function"]),
        "gold": {"name": name, "arguments": call[name]},
    }


def convert_irrelevance_item(q_item: dict) -> dict:
    return {
        "id": q_item["id"],
        "category": "irrelevance",
        "query": first_user_query(q_item["question"]),
        "tools": wrap_tools(q_item["function"]),
        "gold": {"no_call": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BFCL items to Offhand harness format")
    parser.add_argument("--category", required=True, choices=["simple", "irrelevance"])
    parser.add_argument("--questions", required=True, help="BFCL question file")
    parser.add_argument("--answers", help="BFCL possible_answer file (required for call categories)")
    parser.add_argument("--out", required=True, help="output .jsonl path")
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    out_items: list[dict] = []
    skipped = 0

    if args.category == "irrelevance":
        out_items = [convert_irrelevance_item(q) for q in questions]
    else:
        if not args.answers:
            parser.error("--answers is required for the simple category")
        gt_by_id = {g["id"]: g for g in load_jsonl(args.answers)}
        for q in questions:
            gt = gt_by_id.get(q["id"])
            if gt is None or len(gt.get("ground_truth", [])) != 1:
                skipped += 1  # missing answer or multi-call (not yet supported)
                continue
            out_items.append(convert_call_item(q, gt))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in out_items:
            fh.write(json.dumps(item) + "\n")

    print(f"wrote {len(out_items)} items to {out_path}" + (f" (skipped {skipped} multi-call/unmatched)" if skipped else ""))


if __name__ == "__main__":
    main()
