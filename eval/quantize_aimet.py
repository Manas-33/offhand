#!/usr/bin/env python3
"""Quantize the merged specialist to W4A16 with AIMET and run the LOTO eval on it.

Produces the REAL W4A16 accuracy: AIMET QuantSim simulates the on-device numerics
on GPU, replacing the nf4 bitsandbytes proxy in Claims 4 and 8. Runs on the Colab
A100 (needs aimet-torch + CUDA). ONNX-QDQ export for the device is a later step.

Remedy rungs (--remedy):
  rtn       : round-to-nearest W4A16 (QuantSim + calibration)
  spinquant : + SpinQuant rotations (R1+R2; R3 is not exposed, which sidesteps the
              grouped-query-attention shape issue)

Speed: the model is quantized IN PLACE, so the eval uses the model's own cached
``.generate()`` (the same path the fp16/nf4 runs used) instead of a hand-rolled
decode. Quantization is traced through a thin logits-only wrapper (AIMET wants a
tensor out; transformers 5.x Qwen3 needs return_dict internally).

Usage (Colab):
  python eval/quantize_aimet.py --model /content/drive/MyDrive/offhand_out/merged \
      --calib-file /content/drive/MyDrive/offhand_data/train.jsonl \
      --remedy rtn --label specialist_w4a16_rtn
"""

from __future__ import annotations

import argparse
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
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


class GenRunner:
    """Cached greedy generation through the (in-place quantized) model.

    Mirrors HFRunner.generate so the W4A16 numbers are directly comparable to the
    fp16/nf4 runs: same prompt, same greedy decode, just a KV cache instead of the
    O(n^2) manual loop.
    """

    def __init__(self, model, tokenizer, device, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, query: str, tools: list[dict]) -> str:
        import torch

        text = build_prompt(self.tokenizer, query, tools)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


def progress(i: int, total: int, score) -> None:
    if score.expected_call:
        mark = "ok " if score.right_args else ("tool" if score.right_tool else "MISS")
    else:
        mark = "ok " if score.no_call_correct else "MISS"
    print(f"  [{i:>3}/{total}] {mark} {score.id}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="AIMET W4A16 quantize + LOTO eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--data", default=str(HERE / "mini_eval.jsonl"))
    ap.add_argument("--calib-file", required=True, help="jsonl to calibrate on (train.jsonl)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--remedy", choices=["rtn", "spinquant"], default="rtn")
    ap.add_argument("--param-bw", type=int, default=4)
    ap.add_argument("--output-bw", type=int, default=16)
    ap.add_argument("--config-file", default="default", help="AIMET quant config (HTP/blockwise TBD)")
    ap.add_argument("--quant-scheme", default="min_max", help="min_max is fast; post_training_tf_enhanced is slower")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aimet_torch import QuantizationSimModel
    from aimet_torch.common.defs import QuantScheme

    label = args.label or f"specialist_w4a16_{args.remedy}"
    device = "cuda"

    class _LogitsWrapper(torch.nn.Module):
        """Logits-only output for AIMET's tracer; inner model keeps return_dict."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            return self.inner(input_ids=input_ids).logits

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval().to(device)

    by_name = tools_by_name(load_tools(args.tools))
    all_tools = load_tools(args.tools)
    eval_items = load_items(args.data)

    # Baseline smoke on the UN-quantized model to isolate a decode-path bug from a
    # quantization collapse: if this prints a clean tool call but the post-quant
    # smoke is garbage, the quantization is the problem (e.g. per-tensor 4-bit),
    # not the generate path.
    base.config.use_cache = True
    _smoke = GenRunner(base, tokenizer, device, args.max_new_tokens)
    print("pre-quant  smoke:", repr(_smoke.generate(eval_items[0]["query"], all_tools)[:100]))

    base.config.use_cache = False  # off for the AIMET trace; flipped back on for eval

    if args.remedy == "spinquant":
        from aimet_torch.experimental.spinquant import apply_spinquant
        print("applying SpinQuant (R1+R2)...")
        apply_spinquant(base, enable_r1=True, enable_r2=True)

    calib_items = load_items(args.calib_file)[: args.n_calib]
    calib_inputs = [
        tokenizer(build_prompt(tokenizer, it["query"], [by_name[n] for n in it["tools_listed"]]),
                  return_tensors="pt").input_ids.to(device)
        for it in calib_items
    ]
    print(f"calibration set: {len(calib_inputs)} prompts")

    wrapped = _LogitsWrapper(base).to(device).eval()
    quant_scheme = getattr(QuantScheme, args.quant_scheme)
    sim = QuantizationSimModel(
        wrapped,
        dummy_input=calib_inputs[0],
        default_param_bw=args.param_bw,
        default_output_bw=args.output_bw,
        quant_scheme=quant_scheme,
        config_file=args.config_file,
        in_place=True,  # quantize `base` in place so base.generate() is the quantized model
    )

    def forward_pass(m) -> None:
        with torch.no_grad():
            for inp in calib_inputs:
                m(inp)

    print(f"computing encodings ({args.remedy}, {args.quant_scheme})...")
    sim.compute_encodings(forward_pass)

    # Eval through the in-place-quantized model's own cached generate.
    base.config.use_cache = True
    runner = GenRunner(base, tokenizer, device, args.max_new_tokens)
    items = eval_items
    print("post-quant smoke:", repr(runner.generate(items[0]["query"], all_tools)[:100]))

    outcome = run_eval(items, all_tools, runner, on_item=progress)
    outcome["loto"] = {"strict": loto_breakdown(items, outcome["results"], HELD_OUT, "strict")}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"label": label, "model": args.model, "quant": f"W{args.param_bw}A{args.output_bw}",
            "remedy": args.remedy, "config_file": args.config_file, "quant_scheme": args.quant_scheme}
    (out / f"{label}.json").write_text(json.dumps({**meta, **outcome}, indent=2))

    s = outcome["summary"]["strict"]
    print(f"\n=== {label}  (W{args.param_bw}A{args.output_bw} {args.remedy}) ===")
    print(f"  call_accuracy {s['call_accuracy']}  schema {s['schema_valid_rate']}  "
          f"right_tool {s['right_tool_rate']}  irrelevance {s['irrelevance_accuracy']}")
    print()
    print(format_loto(outcome["loto"]["strict"]))
    print(f"\nwrote {out / f'{label}.json'}")


if __name__ == "__main__":
    main()
