#!/usr/bin/env python3
"""Compare eval runs item by item against one subject run.

For each comparison run: its accuracy, and on how many items it matches the
subject's verdict (right or wrong), predicted tool, and byte-identical raw
output. Then every item the subject missed or any run disagrees on, with each
run's verdict, so shared misses and new ones are easy to tell apart.

Runs are result JSONs from results/. Prefix a path with NAME= to set its column
name (default: the label without its "specialist_" and "w4a16_" parts).

Usage:
    python eval/compare_runs.py eval/results/specialist_w4a16_device.json \
        fp16=eval/results/specialist_loto.json eval/results/specialist_nf4.json \
        eval/results/specialist_w4a16_rtn.json eval/results/specialist_w4a16_spinquant.json \
        eval/results/specialist_w4a16_seqmse.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def short_name(label: str) -> str:
    return label.replace("specialist_", "").replace("w4a16_", "") or label


def load_run(spec: str) -> dict:
    """Load a result JSON (optionally NAME=path) into per-item verdicts."""
    name, sep, path = spec.partition("=")
    if not sep:
        name, path = "", spec
    data = json.loads(Path(path).read_text())
    label = data.get("label", Path(path).stem)
    items = {}
    for r in data["results"]:
        score = r.get("strict_score") or r.get("score")
        correct = bool(score["right_args"]) if score["expected_call"] else bool(score["no_call_correct"])
        calls = r.get("strict_calls") or []
        items[r["id"]] = {
            "query": r.get("query", ""),
            "correct": correct,
            "tool": calls[0].get("name") if calls else None,
            "raw": r.get("raw", ""),
        }
    summary = data.get("summary", {}).get("strict", {})
    return {"name": name or short_name(label), "label": label, "summary": summary, "items": items}


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"


def _mark(run: dict, item_id: str) -> str:
    verdict = run["items"].get(item_id)
    if verdict is None:
        return "?"
    return "ok" if verdict["correct"] else "x"


def main() -> None:
    ap = argparse.ArgumentParser(description="Item-by-item comparison of eval runs")
    ap.add_argument("subject", help="result JSON everything is compared against (NAME=path allowed)")
    ap.add_argument("runs", nargs="+", help="result JSONs to compare with (NAME=path allowed)")
    args = ap.parse_args()

    subject = load_run(args.subject)
    runs = [load_run(spec) for spec in args.runs]
    subj = subject["items"]
    ids = sorted(subj)

    s = subject["summary"]
    print(f"subject: {subject['label']}  call_acc {_fmt(s.get('call_accuracy'))}  "
          f"irrelevance {_fmt(s.get('irrelevance_accuracy'))}  n={len(ids)}\n")

    print(f"  {'run':<12}{'call_acc':>9}{'irr':>7}{'same verdict':>14}{'same tool':>11}{'same output':>13}")
    agreement = {}
    for run in runs:
        other = run["items"]
        shared = [i for i in ids if i in other]

        def same(key: str) -> int:
            return sum(subj[i][key] == other[i][key] for i in shared)

        agreement[run["name"]] = (same("correct"), same("raw"))
        cells = [f"{same(key)}/{len(shared)}" for key in ("correct", "tool", "raw")]
        rs = run["summary"]
        print(f"  {run['name']:<12}{_fmt(rs.get('call_accuracy')):>9}{_fmt(rs.get('irrelevance_accuracy')):>7}"
              f"{cells[0]:>14}{cells[1]:>11}{cells[2]:>13}")

    by_verdict = max(agreement, key=lambda n: agreement[n][0])
    by_output = max(agreement, key=lambda n: agreement[n][1])
    print(f"\n  closest by verdicts: {by_verdict} ({agreement[by_verdict][0]}/{len(ids)}), "
          f"by identical outputs: {by_output} ({agreement[by_output][1]}/{len(ids)})")

    shown = [i for i in ids
             if not subj[i]["correct"] or any(i in r["items"] and r["items"][i]["correct"] != subj[i]["correct"]
                                               for r in runs)]
    print(f"\nitems the subject missed or any run disagrees on ({len(shown)}):")
    print(f"  {'id':<16}{subject['name']:>9}" + "".join(f"{r['name']:>11}" for r in runs))
    for i in shown:
        print(f"  {i:<16}{_mark(subject, i):>9}" + "".join(f"{_mark(r, i):>11}" for r in runs))
        print(f"      query:  {subj[i]['query']}")
        print(f"      {subject['name']}: {' '.join(subj[i]['raw'].split())[:110]}")


if __name__ == "__main__":
    main()
