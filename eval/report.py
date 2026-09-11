#!/usr/bin/env python3
"""Summarize eval result files: bootstrap CIs, a size x precision table, and the
accuracy-vs-memory Pareto chart.

Re-scores each result from its stored ``raw`` output against a dataset's golds
(joined by id), so the numbers match audit.py and reflect the current scoring.
Memory footprint is weights-only (params x bytes: fp16=2, int8=1, nf4=0.5),
inferred from the run label, so it ignores KV cache and activations.

Usage:
  python eval/report.py                                    # eval/results/*.json vs mini_eval
  python eval/report.py eval/results/bfcl_*.json --data eval/data/bfcl.jsonl \
      --plot eval/results/pareto.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offhand_eval.dataset import load_items, load_tools, tools_by_name  # noqa: E402
from offhand_eval.parse import parse_tool_calls, parse_tool_calls_lenient  # noqa: E402
from offhand_eval.scoring import score_item  # noqa: E402
from offhand_eval.stats import bootstrap_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
BYTES_PER_PARAM = {"fp16": 2.0, "bf16": 2.0, "int8": 1.0, "nf4": 0.5}


def parse_label(label: str, model: str = ""):
    """Pull (params_in_billions, precision) from a label like 'bfcl_Qwen3-0.6B_fp16'."""
    precision = next((p for p in BYTES_PER_PARAM if p in label), None)
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z0-9])", f"{label} {model}")
    params = float(match.group(1)) if match else None
    return params, precision


def memory_gb(params, precision):
    if params is None or precision not in BYTES_PER_PARAM:
        return None
    return round(params * BYTES_PER_PARAM[precision], 2)


def _stored_outcomes(row):
    """Per-item outcomes from scores the harness already stored, or None.

    Handles the current schema (strict_score + lenient_score) and the older
    single-score schema (score). Avoids needing the dataset to re-score.
    """
    strict = row.get("strict_score")
    lenient = row.get("lenient_score", strict)
    if strict is None:
        strict = lenient = row.get("score")
    if strict is None:
        return None
    return (strict.get("expected_call"), bool(strict.get("right_args")),
            bool((lenient or strict).get("right_args")), strict.get("no_call_correct"))


def summarize_file(path, gold_by_id, global_tools, n_boot: int = 2000) -> dict:
    data = json.load(open(path))
    label = data.get("label", Path(path).stem)
    model = data.get("model", "")

    call_strict, call_lenient, irrelevance = [], [], []
    for r in data.get("results", []):
        outcomes = _stored_outcomes(r)
        if outcomes is None:  # fall back to re-scoring from raw using the dataset
            item = gold_by_id.get(r["id"])
            if item is None:
                continue
            by_name = tools_by_name(item.get("tools") or global_tools)
            raw = r.get("raw", "")
            strict = score_item(item, parse_tool_calls(raw), by_name)
            lenient = score_item(item, parse_tool_calls_lenient(raw, list(by_name)), by_name)
            outcomes = (strict.expected_call, strict.right_args, lenient.right_args, strict.no_call_correct)
        expected, strict_right, lenient_right, no_call_correct = outcomes
        if expected:
            call_strict.append(strict_right)
            call_lenient.append(lenient_right)
        else:
            irrelevance.append(bool(no_call_correct))

    params, precision = parse_label(label, model)
    s_mean, s_lo, s_hi = bootstrap_ci(call_strict, n_boot)
    l_mean, l_lo, l_hi = bootstrap_ci(call_lenient, n_boot)
    i_mean, i_lo, i_hi = bootstrap_ci(irrelevance, n_boot)
    gap = round(l_mean - s_mean, 4) if (l_mean is not None and s_mean is not None) else None

    return {
        "label": label, "model": model,
        "params_b": params, "precision": precision, "memory_gb": memory_gb(params, precision),
        "n_call": len(call_strict), "n_irr": len(irrelevance),
        "strict_acc": s_mean, "strict_lo": s_lo, "strict_hi": s_hi,
        "lenient_acc": l_mean, "lenient_lo": l_lo, "lenient_hi": l_hi,
        "format_gap": gap,
        "irr_acc": i_mean, "irr_lo": i_lo, "irr_hi": i_hi,
    }


def _fmt(mean, lo, hi):
    return "n/a" if mean is None else f"{mean:.3f} [{lo:.3f},{hi:.3f}]"


def print_table(rows):
    rows = sorted(rows, key=lambda r: (r["memory_gb"] is None, r["memory_gb"] or 0))
    header = (f"{'label':30} {'mem_GB':>7} {'nCall':>5} "
              f"{'strict call_acc':>24} {'lenient':>24} {'gap':>7} {'irrelevance':>24}")
    print(header)
    print("-" * len(header))
    for r in rows:
        gap = f"+{r['format_gap']:.3f}" if r["format_gap"] is not None else "n/a"
        print(f"{r['label']:30} {str(r['memory_gb'] if r['memory_gb'] is not None else '?'):>7} "
              f"{r['n_call']:>5} {_fmt(r['strict_acc'], r['strict_lo'], r['strict_hi']):>24} "
              f"{_fmt(r['lenient_acc'], r['lenient_lo'], r['lenient_hi']):>24} {gap:>7} "
              f"{_fmt(r['irr_acc'], r['irr_lo'], r['irr_hi']):>24}")


def write_csv(rows, path):
    keys = ["label", "model", "params_b", "precision", "memory_gb", "n_call", "n_irr",
            "strict_acc", "strict_lo", "strict_hi", "lenient_acc", "lenient_lo", "lenient_hi",
            "format_gap", "irr_acc", "irr_lo", "irr_hi"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def plot_pareto(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot (numbers are in the CSV)")
        return False

    pts = [r for r in rows if r["memory_gb"] is not None and r["strict_acc"] is not None]
    if not pts:
        print("no points with known memory + accuracy; skipping plot")
        return False

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for precision in sorted({r["precision"] for r in pts}):
        sub = sorted([r for r in pts if r["precision"] == precision], key=lambda r: r["memory_gb"])
        xs = [r["memory_gb"] for r in sub]
        ys = [r["strict_acc"] for r in sub]
        yerr = [[y - r["strict_lo"] for y, r in zip(ys, sub)],
                [r["strict_hi"] - y for y, r in zip(ys, sub)]]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=precision)
        for r in sub:
            ax.annotate(f"{r['params_b']}B", (r["memory_gb"], r["strict_acc"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("memory footprint (GB, weights only)")
    ax.set_ylabel("strict call accuracy")
    ax.set_title("Tool-calling accuracy vs memory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Summarize eval results with CIs and a Pareto chart")
    ap.add_argument("results", nargs="*", help="result JSON files (default: eval/results/*.json)")
    ap.add_argument("--data", default=str(HERE / "mini_eval.jsonl"),
                    help="dataset providing golds + per-item tools, joined by id")
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "report.csv"))
    ap.add_argument("--plot", default=None, help="path to save the accuracy-vs-memory PNG")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    paths = args.results or sorted(glob.glob(str(HERE / "results" / "*.json")))
    if not paths:
        print("no result files found (pass paths, or put JSON in eval/results/)")
        return

    global_tools = load_tools(args.tools) if Path(args.tools).exists() else []
    gold_by_id = {it["id"]: it for it in load_items(args.data)} if Path(args.data).exists() else {}
    rows = [summarize_file(p, gold_by_id, global_tools, args.n_boot) for p in paths]

    print_table(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out)
    print(f"\nwrote {args.out}")
    if args.plot:
        plot_pareto(rows, args.plot)


if __name__ == "__main__":
    main()
