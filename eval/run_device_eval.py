#!/usr/bin/env python3
"""Replay the eval on the phone's NPU through the Offhand app, then score it.

The device side is the app's EvalActivity (debug builds only). It runs every
item through the same GenieEngine the app uses: the prompt the eval renders
(see export_app_prompt.py), greedy decoding, one fresh request per item. It
records each raw output with the runtime's own timing. This script pushes the
items, starts that activity over adb, waits for it, pulls the outputs, and
scores them with the harness by replaying the recorded outputs as a runner, so
the scoring is the same code that scored the GPU runs.

The result JSON has the same shape as the GPU runs in results/, plus per-item
device timing and a check that every device prompt was token-identical to the
eval's, so report.py and compare_runs.py read it like any other run.

Needs a debug build of the app installed, the model bundle seeded under the
app's files/models/, and the phone on adb.

Usage:
    python eval/run_device_eval.py
    python eval/run_device_eval.py --label <name> --model-dir models/<bundle> --recipe "<how it was quantized>"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from offhand_eval.dataset import load_items, load_tools  # noqa: E402
from offhand_eval.harness import run_eval  # noqa: E402
from offhand_eval.loto import HELD_OUT, format_loto, loto_breakdown  # noqa: E402

PACKAGE = "com.offhand.offhand"
ACTIVITY = f"{PACKAGE}/.EvalActivity"
EVAL_DIR = "files/eval"
PROMPT_ASSET = HERE.parent / "app/android/app/src/main/assets/prompt_template.json"
DEVICE_FIELDS = ("ttft_ms", "prompt_tokens", "generated_tokens", "prefill_tps", "decode_tps", "stop_reason", "error")

# What the app ships (app/android/app/build.gradle.kts) and the bundle was compiled for.
RUNTIME = "GenieX 0.3.5 qairt plugin, QAIRT 2.45, Hexagon v79 NPU"
DEFAULT_RECIPE = (
    "qai-hub-models w4a16 default: SpinQuant R1+R2+R3, AdaScale, calibration on Wikitext + "
    "generated text; int4 per-channel weights, int8 lm_head and KV, int16 embedding and "
    "activations; layers.2.mlp.down_proj encoding offset patched for QAIRT 2.45"
)


class ReplayRunner:
    """Hand the harness the outputs the device already produced, keyed by query."""

    def __init__(self, raw_by_query: dict[str, str]):
        self.raw_by_query = raw_by_query

    def generate(self, query: str, tools: list[dict]) -> str:
        return self.raw_by_query[query]


def parse_device_rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def score_device_rows(items: list[dict], tools: list[dict], rows: list[dict], on_item=None) -> dict:
    """Score recorded device outputs with the harness, exactly like a live run."""
    if len({it["query"] for it in items}) != len(items):
        raise ValueError("replay needs unique queries")
    by_id = {r["id"]: r for r in rows}
    missing = [it["id"] for it in items if it["id"] not in by_id]
    if missing:
        raise ValueError(f"device results are missing {len(missing)} items: {missing[:5]}")

    runner = ReplayRunner({it["query"]: by_id[it["id"]].get("raw", "") for it in items})
    outcome = run_eval(items, tools, runner, on_item=on_item)
    outcome["loto"] = {"strict": loto_breakdown(items, outcome["results"], HELD_OUT, "strict")}
    for res in outcome["results"]:
        row = by_id[res["id"]]
        res["device"] = {k: row[k] for k in DEVICE_FIELDS if k in row}
    return outcome


def timing_summary(rows: list[dict]) -> dict:
    def median(key: str):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.median(vals), 2) if vals else None

    return {
        "n": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "median_ttft_ms": median("ttft_ms"),
        "median_decode_tps": median("decode_tps"),
        "median_prefill_tps": median("prefill_tps"),
        "median_prompt_tokens": median("prompt_tokens"),
        "median_generated_tokens": median("generated_tokens"),
    }


def token_check(items: list[dict], rows: list[dict], tokenizer_id: str) -> dict:
    """Compare each device prompt's token count with the eval's tokenization."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    template = json.loads(PROMPT_ASSET.read_text())
    by_id = {r["id"]: r for r in rows}
    mismatched = []
    for it in items:
        expected = len(tokenizer(template["prefix"] + it["query"] + template["suffix"]).input_ids)
        got = by_id[it["id"]].get("prompt_tokens")
        if got != expected:
            mismatched.append({"id": it["id"], "device": got, "eval": expected})
    return {"checked": len(items), "mismatched": mismatched}


def find_adb() -> str:
    found = shutil.which("adb")
    if found:
        return found
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or str(Path.home() / "Library/Android/sdk")
    candidate = Path(sdk) / "platform-tools" / "adb"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("adb not found: put it on PATH or set ANDROID_HOME")


class Device:
    """Thin adb wrapper; ``app`` runs a command as the app user via run-as."""

    def __init__(self, adb: str, serial: str | None = None):
        self.base = [adb, *(["-s", serial] if serial else [])]

    def adb(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run([*self.base, *args], capture_output=True, encoding="utf-8", errors="replace")
        if check and proc.returncode != 0:
            raise SystemExit(f"adb {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}")
        return proc.stdout

    def app(self, *args: str, check: bool = True) -> str:
        return self.adb("shell", "run-as", PACKAGE, *args, check=check)

    def read(self, path: str) -> str:
        return self.adb("exec-out", "run-as", PACKAGE, "cat", path, check=False)


def bundle_config(dev: Device, model_dir: str) -> dict | None:
    """The bundle's sampler and context settings, as the device has them."""
    try:
        dialog = json.loads(dev.read(f"files/{model_dir}/genie_config.json"))["dialog"]
        return {"context": dialog["context"], "sampler": dialog["sampler"]}
    except (ValueError, KeyError, TypeError):
        return None


def run_on_device(dev: Device, items: list[dict], model_dir: str, timeout_s: int) -> str:
    """Push the items, run EvalActivity, wait for it, and return results.jsonl text."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it["id"], "query": it["query"]}, ensure_ascii=False) + "\n")
    remote = "/data/local/tmp/offhand_eval_items.jsonl"
    try:
        dev.adb("push", fh.name, remote)
    finally:
        os.unlink(fh.name)
    dev.app("mkdir", "-p", EVAL_DIR)
    dev.app("cp", remote, f"{EVAL_DIR}/items.jsonl")
    dev.adb("shell", "rm", remote)
    dev.app("rm", "-f", f"{EVAL_DIR}/results.jsonl", f"{EVAL_DIR}/results.jsonl.part", f"{EVAL_DIR}/error.txt")

    # Start from a cold process: a model the app already loaded would double
    # the memory and NPU footprint.
    dev.adb("shell", "am", "force-stop", PACKAGE)
    started = dev.adb("shell", "am", "start", "-n", ACTIVITY, "--es", "model", model_dir)
    if "Error" in started:
        raise SystemExit(f"could not start {ACTIVITY}: {started.strip()}")

    deadline, shown = time.time() + timeout_s, -1
    while time.time() < deadline:
        time.sleep(3)
        names = dev.app("ls", EVAL_DIR, check=False).split()
        if "error.txt" in names:
            raise SystemExit("device eval failed: " + dev.read(f"{EVAL_DIR}/error.txt").strip())
        if "results.jsonl" in names:
            text = dev.read(f"{EVAL_DIR}/results.jsonl")
            dev.adb("shell", "am", "force-stop", PACKAGE, check=False)  # release the model
            return text
        if "results.jsonl.part" in names:
            done = len(dev.read(f"{EVAL_DIR}/results.jsonl.part").splitlines())
            if done != shown:
                print(f"  device: {done}/{len(items)}", file=sys.stderr)
                shown = done
    raise SystemExit(f"timed out after {timeout_s}s; see `adb logcat -s OffhandEval`")


def progress(i: int, total: int, score) -> None:
    if score.expected_call:
        mark = "ok " if score.right_args else ("tool" if score.right_tool else "MISS")
    else:
        mark = "ok " if score.no_call_correct else "MISS"
    print(f"  [{i:>3}/{total}] {mark}  {score.id}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the eval on the phone NPU through the Offhand app and score it")
    ap.add_argument("--label", default="specialist_w4a16_device")
    ap.add_argument("--data", default=str(HERE / "mini_eval.jsonl"))
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--model-dir", default="models/Qwen3-0.6B-Specialist",
                    help="bundle directory, relative to the app's files dir")
    ap.add_argument("--remedy", default="device_default")
    ap.add_argument("--recipe", default=DEFAULT_RECIPE, help="how the bundle was quantized, for the record")
    ap.add_argument("--serial", default=None, help="adb serial, when more than one device is attached")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the whole run")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                    help="for the prompt-token check; the merged checkpoint ships the base tokenizer")
    ap.add_argument("--no-token-check", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    items = load_items(args.data)
    tools = load_tools(args.tools)
    dev = Device(find_adb(), args.serial)
    device_name = dev.adb("shell", "getprop", "ro.product.model").strip()
    config = bundle_config(dev, args.model_dir)

    print(f"running {len(items)} items on {device_name} ({args.model_dir})", file=sys.stderr)
    t0 = time.time()
    rows = parse_device_rows(run_on_device(dev, items, args.model_dir, args.timeout))
    wall_s = round(time.time() - t0, 1)

    outcome = score_device_rows(items, tools, rows, on_item=progress)
    timing = {**timing_summary(rows), "wall_s_push_to_pull": wall_s}
    template = json.loads(PROMPT_ASSET.read_text())
    meta = {
        "label": args.label,
        "model": args.model_dir,
        "quant": "W4A16",
        "remedy": args.remedy,
        "recipe": args.recipe,
        "runtime": RUNTIME,
        "device": device_name,
        "bundle_config": config,
        "prompt_template": {
            "asset": str(PROMPT_ASSET.relative_to(HERE.parent)),
            "tools_sha256": template.get("tools_sha256"),
        },
        "timing": timing,
    }
    if not args.no_token_check:
        meta["prompt_token_check"] = token_check(items, rows, args.tokenizer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps({**meta, **outcome}, indent=2))

    s = outcome["summary"]["strict"]
    print(f"\n=== {args.label}  ({device_name}, {args.model_dir}) ===")
    print(f"  call_accuracy {s['call_accuracy']}  schema {s['schema_valid_rate']}  "
          f"right_tool {s['right_tool_rate']}  irrelevance {s['irrelevance_accuracy']}")
    print(f"  median ttft {timing['median_ttft_ms']} ms  decode {timing['median_decode_tps']} tok/s  "
          f"prompt {timing['median_prompt_tokens']} tokens  wall {wall_s}s")
    errors = [r["id"] for r in rows if r.get("error")]
    if errors:
        print(f"  device errors on {len(errors)} items: {errors}")
    tc = meta.get("prompt_token_check")
    if tc:
        bad = tc["mismatched"]
        print(f"  prompt tokens match the eval on {tc['checked'] - len(bad)}/{tc['checked']} items"
              + (f"; mismatched: {bad[:5]}" if bad else ""))
    print()
    print(format_loto(outcome["loto"]["strict"]))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
