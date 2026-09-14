#!/usr/bin/env python3
"""Build a validation set from BFCL's live categories, for choosing a training mix.

The BFCL-840 test (simple, multiple, irrelevance) should score the model a mix
search settles on, once. Choosing among mixes by their BFCL-840 scores would tune
the model to the test. This builds a separate set from BFCL's "live" categories,
real user questions from a different source than the 840, shaped like the test:

 - call items: every live_simple question (one tool offered) plus a sample of
   live_multiple questions offering 2 to 4 tools,
 - no-call items: a sample of live_irrelevance questions offering exactly one tool,
 - only questions that are a single user message (no system prompt, no earlier turns),
 - no call question whose answer key accepts no value for some argument (a few
   live answers do), since no model could score on it,
 - nothing that overlaps a BFCL-840 or training question, and no item offering a
   tool the training data uses, so every tool is unseen, as in the test.

Get the files from the same Hugging Face dataset as the test:
  base=https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main
  wget $base/BFCL_v3_live_simple.json
  wget $base/possible_answer/BFCL_v3_live_simple.json -O BFCL_v3_live_simple_answers.json
  wget $base/BFCL_v3_live_multiple.json
  wget $base/possible_answer/BFCL_v3_live_multiple.json -O BFCL_v3_live_multiple_answers.json
  wget $base/BFCL_v3_live_irrelevance.json

Usage:
  python eval/make_bfcl_dev.py \
      --simple BFCL_v3_live_simple.json BFCL_v3_live_simple_answers.json \
      --multiple BFCL_v3_live_multiple.json BFCL_v3_live_multiple_answers.json \
      --irrelevance BFCL_v3_live_irrelevance.json \
      --test eval/data/bfcl.jsonl --train <v1>/train.jsonl <hard>/train_final.jsonl \
      --out eval/data/bfcl_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))

from check_contamination import query_overlaps  # noqa: E402
from convert_bfcl import convert_call_item, convert_irrelevance_item, load_jsonl  # noqa: E402


def single_user_message(question: list) -> bool:
    """True when a BFCL question is one turn holding one user message."""
    return len(question) == 1 and len(question[0]) == 1 and question[0][0].get("role") == "user"


def accepts_nothing(call: dict) -> bool:
    """True when some argument's list of accepted values is empty, which no answer can match."""
    args = next(iter(call.values()))
    return any(isinstance(v, list) and not v for v in args.values())


def call_items(questions: list[dict], answers: list[dict], min_tools: int, max_tools: int) -> list[dict]:
    """Single-call questions offering min_tools to max_tools tools, in harness format."""
    gt_by_id = {a["id"]: a for a in answers}
    out = []
    for q in questions:
        gt = gt_by_id.get(q["id"])
        if gt is None or len(gt.get("ground_truth", [])) != 1 or accepts_nothing(gt["ground_truth"][0]):
            continue
        if not single_user_message(q["question"]) or not min_tools <= len(q["function"]) <= max_tools:
            continue
        out.append(convert_call_item(q, gt))
    return out


def no_call_items(questions: list[dict]) -> list[dict]:
    """Irrelevance questions offering exactly one tool, like the test's, in harness format."""
    return [convert_irrelevance_item(q) for q in questions
            if single_user_message(q["question"]) and len(q["function"]) == 1]


def offers_tool(item: dict, names: set[str]) -> bool:
    return any(t["function"]["name"] in names for t in item["tools"])


def build_dev(simple: list[dict], multiple: list[dict], no_call: list[dict], avoid_queries: list[str],
              avoid_tools: set[str], n_multiple: int, n_no_call: int, seed: int) -> tuple[list[dict], list[str]]:
    """Drop overlapping questions and training tools, then sample; returns the items and a log."""
    log, pools = [], {}
    for name, items in (("simple", simple), ("multiple", multiple), ("no-call", no_call)):
        overlapping = {id(it) for it, _ in query_overlaps(items, avoid_queries)}
        seen_tool = {id(it) for it in items if offers_tool(it, avoid_tools)}
        pools[name] = [it for it in items if id(it) not in overlapping | seen_tool]
        log.append(f"{name}: {len(items)} eligible, dropped {len(overlapping)} overlapping a test or training "
                   f"question and {len(seen_tool - overlapping)} offering a training tool, {len(pools[name])} left")

    rng = random.Random(seed)

    def sample(items: list[dict], n: int) -> list[dict]:
        if n >= len(items):
            return items
        picked = {id(it) for it in rng.sample(items, n)}
        return [it for it in items if id(it) in picked]  # keep file order

    return pools["simple"] + sample(pools["multiple"], n_multiple) + sample(pools["no-call"], n_no_call), log


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a BFCL live validation set shaped like the BFCL-840 test")
    ap.add_argument("--simple", nargs=2, required=True, metavar=("QUESTIONS", "ANSWERS"))
    ap.add_argument("--multiple", nargs=2, required=True, metavar=("QUESTIONS", "ANSWERS"))
    ap.add_argument("--irrelevance", required=True, metavar="QUESTIONS")
    ap.add_argument("--test", required=True, help="the converted BFCL-840 test (eval/data/bfcl.jsonl)")
    ap.add_argument("--train", nargs="*", default=[], help="training jsonl files whose questions to avoid")
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--extra-tools", default=str(HERE.parent / "train" / "general_tools.json"))
    ap.add_argument("--n-multiple", type=int, default=125, help="live_multiple questions to sample")
    ap.add_argument("--n-no-call", type=int, default=300, help="live_irrelevance questions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    simple = call_items(load_jsonl(args.simple[0]), load_jsonl(args.simple[1]), 1, 1)
    multiple = call_items(load_jsonl(args.multiple[0]), load_jsonl(args.multiple[1]), 2, 4)
    no_call = no_call_items(load_jsonl(args.irrelevance))
    avoid_queries = [it["query"] for it in load_jsonl(args.test)]
    avoid_queries += [it["query"] for path in args.train for it in load_jsonl(path)]
    avoid_tools = {t["function"]["name"] for path in (args.tools, args.extra_tools)
                   for t in json.load(open(path, encoding="utf-8"))}

    items, log = build_dev(simple, multiple, no_call, avoid_queries, avoid_tools,
                           args.n_multiple, args.n_no_call, args.seed)
    for line in log:
        print(line)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    n_call = sum(1 for it in items if not it["gold"].get("no_call"))
    print(f"wrote {len(items)} items ({n_call} call, {len(items) - n_call} no-call) to {out}")


if __name__ == "__main__":
    main()
