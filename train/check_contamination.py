#!/usr/bin/env python3
"""Check a generated training set against BFCL before training on it.

The hard families bring informational tools and question-style requests into
training, which moves the training data closer to the kind of task BFCL asks.
That is fine. Training on BFCL's own questions or function schemas is not,
because BFCL is the out-of-domain check. Two checks:

 - Tool names: any training tool (phone + informational) whose name matches a
   BFCL function name. A shared schema is leakage that dropping queries cannot
   undo, so this fails the check outright (exit code 1).
 - Queries: any training query that duplicates a BFCL question, by normalized
   exact match, containment, or high word 3-gram overlap. These are listed and
   left out of the cleaned file.

Usage (after gen_data.py, with BFCL converted as for the eval runs):
  python train/check_contamination.py --train <hard>/train.jsonl \
      --bfcl eval/data/bfcl.jsonl --out <hard>/train_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _trigrams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def query_overlaps(train_items: list[dict], queries: list[str],
                   min_shared: int = 3, min_frac: float = 0.6) -> list[tuple[dict, str]]:
    """Items whose query duplicates one of ``queries`` (BFCL's, here), with the reason.

    Short queries are only checked for exact matches and, from six words up,
    containment: a pair of shared trigrams in a short query is ordinary phrasing
    ("what is the"), not a copied question.
    """
    norm_b = [normalize(q) for q in queries]
    exact = set(norm_b)
    index: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for j, text in enumerate(norm_b):
        for gram in _trigrams(text.split()):
            index[gram].add(j)

    flagged = []
    for item in train_items:
        query = normalize(item["query"])
        words = query.split()
        reason = None
        if query in exact:
            reason = "exact match"
        elif len(words) >= 6 and (j := next((k for k, b in enumerate(norm_b) if query in b), None)) is not None:
            reason = f"contained in: {queries[j][:80]}"
        else:
            grams = _trigrams(words)
            if len(grams) >= min_shared:
                shared = Counter(j for g in grams for j in index.get(g, ()))
                for j, n in shared.most_common(1):
                    if n >= min_shared and n / len(grams) >= min_frac:
                        reason = f"{n}/{len(grams)} trigrams shared with: {queries[j][:80]}"
        if reason:
            flagged.append((item, reason))
    return flagged


def name_collisions(train_tools: list[dict], bfcl_items: list[dict]) -> list[str]:
    bfcl_names = {t["function"]["name"] for it in bfcl_items for t in it.get("tools") or []}
    return sorted({t["function"]["name"] for t in train_tools} & bfcl_names)


def _load_jsonl(paths: list[str]) -> list[dict]:
    return [json.loads(line) for path in paths for line in open(path, encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check training data against BFCL for leakage")
    ap.add_argument("--train", nargs="+", required=True, help="generated training jsonl file(s)")
    ap.add_argument("--bfcl", required=True, help="converted BFCL jsonl (eval/data/bfcl.jsonl)")
    ap.add_argument("--tools", default=str(HERE.parent / "eval" / "tools.json"))
    ap.add_argument("--extra-tools", default=str(HERE / "general_tools.json"))
    ap.add_argument("--out", default=None, help="write the items that pass the query check here")
    args = ap.parse_args()

    train = _load_jsonl(args.train)
    bfcl = _load_jsonl([args.bfcl])
    tools = [t for path in (args.tools, args.extra_tools) for t in json.load(open(path, encoding="utf-8"))]

    collisions = name_collisions(tools, bfcl)
    flagged = query_overlaps(train, [it["query"] for it in bfcl])
    print(f"checked {len(train)} training items and {len(tools)} tools against {len(bfcl)} BFCL items")
    print(f"tool-name collisions: {collisions or 'none'}")
    print(f"queries flagged: {len(flagged)}")
    for item, reason in flagged[:20]:
        print(f"  {item['query'][:70]!r}: {reason}")

    if args.out:
        bad = {id(item) for item, _ in flagged}
        clean = [it for it in train if id(it) not in bad]
        with open(args.out, "w", encoding="utf-8") as fh:
            for item in clean:
                fh.write(json.dumps(item) + "\n")
        print(f"wrote {len(clean)} items to {args.out}")

    if collisions:
        sys.exit("tool names shared with BFCL: rename them in general_tools.json before training")


if __name__ == "__main__":
    main()
