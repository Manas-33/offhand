#!/usr/bin/env python3
"""Quantize the merged specialist to W4A16 with AIMET and run the LOTO eval on it.

This produces the REAL W4A16 accuracy number: AIMET QuantSim simulates the
on-device quantized numerics on GPU, which replaces the nf4 bitsandbytes proxy in
Claims 4 and 8. Runs on the Colab A100 (needs aimet-torch + CUDA). The ONNX-QDQ
export for the actual device (Genie/QNN) is a later step; this is the accuracy.

Status: v1 does round-to-nearest (RTN) W4A16 using only the confirmed QuantSim
API, and prints what aimet_torch exposes for seq_mse / spinquant so the next
version can add those remedy rungs against the installed API instead of guessing.

Two things to confirm from v1's output before we trust the headline number:
  1. whether "A16" should be int16 activations (--output-bw 16, the current default)
     or weight-only int4 with fp16 activations (a different QuantSim config);
  2. the exact seq_mse / spinquant entry points (printed at startup).

Usage (Colab, after `pip install aimet-torch`):
  python eval/quantize_aimet.py \
      --model /content/drive/MyDrive/offhand_out/merged \
      --calib-file /content/drive/MyDrive/offhand_data/train.jsonl \
      --label specialist_w4a16_rtn
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from offhand_eval.dataset import load_items, load_tools, tools_by_name  # noqa: E402
from offhand_eval.harness import run_eval  # noqa: E402
from offhand_eval.loto import HELD_OUT, format_loto, loto_breakdown  # noqa: E402
from offhand_eval.runners import DEFAULT_SYSTEM_PROMPT  # noqa: E402  (same prompt as training/eval)


def build_prompt(tokenizer, query: str, tools: list[dict]) -> str:
    """The exact deployment prompt (matches HFRunner and train_lora)."""
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


class AimetRunner:
    """Greedy-decode through the QuantSim-wrapped model, same interface as HFRunner.

    A manual argmax loop rather than model.generate(), so it does not depend on the
    HF generation mixin surviving AIMET's module wrapping.
    """

    def __init__(self, model, tokenizer, device, max_new_tokens: int = 96):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, query: str, tools: list[dict]) -> str:
        import torch

        prompt = build_prompt(self.tokenizer, query, tools)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        start = ids.shape[1]
        eos = self.tokenizer.eos_token_id
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = self.model(ids)
                logits = out.logits if hasattr(out, "logits") else out[0]
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                if nxt.item() == eos:
                    break
        return self.tokenizer.decode(ids[0, start:], skip_special_tokens=True)


def _probe_aimet_api() -> None:
    """Print what this aimet_torch build exposes, so we wire the remedies correctly."""
    import pkgutil

    import aimet_torch
    from aimet_torch import QuantizationSimModel

    print("=== AIMET API probe ===")
    print("submodules:", sorted(m.name for m in pkgutil.iter_modules(aimet_torch.__path__)))
    print("QuantizationSimModel.__init__:", inspect.signature(QuantizationSimModel.__init__))
    try:
        from aimet_torch.common.defs import QuantScheme
        print("QuantScheme:", [x for x in dir(QuantScheme) if not x.startswith("_")])
    except Exception as exc:  # noqa: BLE001
        print("QuantScheme import failed:", exc)
    for mod in ("seq_mse", "spinquant", "experimental.spinquant"):
        try:
            m = __import__(f"aimet_torch.{mod}", fromlist=["_"])
            print(f"aimet_torch.{mod}:", [x for x in dir(m) if not x.startswith("_")])
        except Exception as exc:  # noqa: BLE001
            print(f"aimet_torch.{mod}: {exc}")
    # exact signatures for the remedy entry points (so v2 wires them precisely)
    try:
        from aimet_torch.seq_mse import SeqMseParams, apply_seq_mse
        print("apply_seq_mse:", inspect.signature(apply_seq_mse))
        print("SeqMseParams:", inspect.signature(SeqMseParams))
    except Exception as exc:  # noqa: BLE001
        print("seq_mse sigs:", exc)
    try:
        from aimet_torch.experimental.spinquant import apply_spinquant
        print("apply_spinquant:", inspect.signature(apply_spinquant))
    except Exception as exc:  # noqa: BLE001
        print("apply_spinquant sig:", exc)
    print("=== end probe ===\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="AIMET W4A16 quantize + LOTO eval (v1: RTN)")
    ap.add_argument("--model", required=True, help="merged specialist path (on Drive)")
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--data", default=str(HERE / "mini_eval.jsonl"))
    ap.add_argument("--calib-file", required=True, help="jsonl of items to calibrate on (train.jsonl)")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--param-bw", type=int, default=4, help="weight bitwidth (W4)")
    ap.add_argument("--output-bw", type=int, default=16, help="activation bitwidth (A16)")
    ap.add_argument("--config-file", default="default", help="AIMET quant config (HTP/blockwise TBD)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--label", default="specialist_w4a16_rtn")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aimet_torch import QuantizationSimModel
    from aimet_torch.common.defs import QuantScheme

    _probe_aimet_api()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval().to(device)
    model.config.return_dict = False  # AIMET's tracer needs tuple/tensor outputs, not ModelOutput
    model.config.use_cache = False    # cleaner trace; the manual decode loop doesn't use the cache

    by_name = tools_by_name(load_tools(args.tools))
    all_tools = load_tools(args.tools)

    # Calibration inputs: training prompts in the exact deployment format.
    calib_items = load_items(args.calib_file)[: args.n_calib]
    calib_inputs = []
    for it in calib_items:
        tools = [by_name[n] for n in it["tools_listed"]]
        text = build_prompt(tokenizer, it["query"], tools)
        calib_inputs.append(tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device))
    print(f"calibration set: {len(calib_inputs)} prompts")

    # PTQ scheme (no training); fall back if this build renamed the enum.
    quant_scheme = getattr(QuantScheme, "post_training_tf_enhanced", None) or getattr(QuantScheme, "min_max")
    sim = QuantizationSimModel(
        model,
        dummy_input=calib_inputs[0],
        default_param_bw=args.param_bw,
        default_output_bw=args.output_bw,
        quant_scheme=quant_scheme,
        config_file=args.config_file,
    )

    def forward_pass(m) -> None:
        with torch.no_grad():
            for inp in calib_inputs:
                m(inp)

    print("computing encodings (RTN calibration)...")
    sim.compute_encodings(forward_pass)

    runner = AimetRunner(sim.model, tokenizer, device, args.max_new_tokens)
    items = load_items(args.data)
    outcome = run_eval(items, all_tools, runner)
    outcome["loto"] = {"strict": loto_breakdown(items, outcome["results"], HELD_OUT, "strict")}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"label": args.label, "model": args.model,
            "quant": f"W{args.param_bw}A{args.output_bw}", "remedy": "rtn", "config_file": args.config_file}
    (out / f"{args.label}.json").write_text(json.dumps({**meta, **outcome}, indent=2))

    s = outcome["summary"]["strict"]
    print(f"\n=== {args.label}  (W{args.param_bw}A{args.output_bw} RTN) ===")
    print(f"  call_accuracy {s['call_accuracy']}  schema {s['schema_valid_rate']}  "
          f"right_tool {s['right_tool_rate']}  irrelevance {s['irrelevance_accuracy']}")
    print()
    print(format_loto(outcome["loto"]["strict"]))
    print(f"\nwrote {out / f'{args.label}.json'}")


if __name__ == "__main__":
    main()
