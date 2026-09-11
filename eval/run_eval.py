#!/usr/bin/env python3
"""M0c CLI: evaluate a model's phone-tool-calling reliability.

Examples
--------
FP16 baseline (Colab, GPU):
    python run_eval.py --model Qwen/Qwen3-4B --label fp16

4-bit proxy for the "does the curve exist?" spike:
    python run_eval.py --model Qwen/Qwen3-4B --config nf4 --label w4_nf4

Local (Apple Silicon) fp16 smoke test with a small model:
    python run_eval.py --model Qwen/Qwen3-0.6B --config fp16 --device mps --label local_smoke

Smoke-test the harness with no model download:
    python run_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offhand_eval.dataset import load_items, load_tools, tools_by_name  # noqa: E402
from offhand_eval.harness import run_eval  # noqa: E402
from offhand_eval.scoring import ItemScore  # noqa: E402

HERE = Path(__file__).resolve().parent


def build_runner(args, tools):
    if args.dry_run:
        # Echo the reference-correct call back so the harness plumbing can be
        # exercised end-to-end without downloading a model.
        from offhand_eval.mock import build_mock_runner

        items = load_items(args.data)
        return build_mock_runner(items, tools_by_name(tools))

    from offhand_eval.runners import HFRunner

    return HFRunner(
        model_id=args.model,
        tools=tools,
        quant=args.config,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


def progress(i: int, total: int, score: ItemScore) -> None:
    if score.expected_call:
        mark = "ok " if score.right_args else ("tool" if score.right_tool else "MISS")
    else:
        mark = "ok " if score.no_call_correct else "MISS"
    print(f"  [{i:>3}/{total}] {mark}  {score.id}", file=sys.stderr)


def _print_agg(name: str, agg: dict) -> None:
    print(f"  [{name}]")
    for key, value in agg.items():
        print(f"    {key:<22} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offhand M0c tool-calling eval")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id or local path")
    parser.add_argument("--config", default="fp16", choices=["fp16", "int8", "nf4"],
                        help="precision for the HF reference runner")
    parser.add_argument("--label", default=None, help="name for this run (defaults to --config)")
    parser.add_argument("--tools", default=str(HERE / "tools.json"))
    parser.add_argument("--data", default=str(HERE / "mini_eval.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                        help="explicit device for local fp16 runs (e.g. mps on Apple Silicon); "
                             "omit on Colab to let device_map=auto place on the GPU")
    parser.add_argument("--out-dir", default=str(HERE / "results"))
    parser.add_argument("--dry-run", action="store_true",
                        help="use a mock runner (no model) to verify the harness")
    args = parser.parse_args()

    label = args.label or ("dry-run" if args.dry_run else args.config)
    tools = load_tools(args.tools)
    items = load_items(args.data)
    if args.limit:
        items = items[: args.limit]

    runner = build_runner(args, tools)
    outcome = run_eval(items, tools_by_name(tools), runner, on_item=progress)
    summary = outcome["summary"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(
        json.dumps({"label": label, "model": args.model, **outcome}, indent=2)
    )

    print()
    print(f"=== {label}  ({args.model}) ===")
    _print_agg("strict", summary["strict"])
    _print_agg("lenient", summary["lenient"])
    print(f"  format_recoverable (lenient - strict call_acc): {summary['format_recoverable']}")
    print(f"\nwrote {out_dir / f'{label}.json'}")


if __name__ == "__main__":
    main()
