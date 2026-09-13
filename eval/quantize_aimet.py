#!/usr/bin/env python3
"""Quantize the merged specialist to W4A16 with AIMET and run the LOTO eval on it.

Produces the REAL W4A16 accuracy: AIMET QuantSim simulates the on-device numerics
on GPU, replacing the nf4 bitsandbytes proxy in Claims 4 and 8. Runs on the Colab
A100 (needs aimet-torch + CUDA). ONNX-QDQ export for the device is a later step.

Why the earlier runs collapsed (both fixes are in this script now):
  1. RMSNorm gammas were silently quantized to PER-TENSOR int4: the config defaults
     ("params": is_quantized) override the None quantizer that AIMET's own
     QuantizedQwen3RMSNorm registers. Norm weights now stay fp (the standard W4A16
     contract quantizes matmul weights only).
  2. Per-channel was never the missing piece - AIMET 2.39's "default" config alias
     already IS per-channel, so the custom per-channel JSON changed nothing. The
     Linear granularity that actually matches the device and the nf4 proxy is
     blockwise/LPBQ, set post-hoc via aimet_torch.quantsim.config_utils.

Weight granularity (--granularity) is the lever on the Linears:
  per_channel : one scale per output row. NOTE: AIMET 2.39's "default" config alias
                already IS per-channel (default_config_per_channel.json), so this is
                what every earlier run used - and 4-bit per-channel collapses this
                0.6B (one outlier weight per row zeros out the rest of the row).
  blockwise   : one scale per 64-column block of each row (matches the nf4 blockwise
                proxy, which held 0.925 at blocksize 64).
  lpbq        : grouped blockwise (LPBQ: int4 blocks, block scales quantized to int8
                against a per-channel scale, decompressed_bw=8). This is what QNN
                actually runs on the HTP, so it is the device-representative setting.

Remedy rungs (--remedy):
  rtn               : round-to-nearest W4A16 (QuantSim + calibration)
  seqmse            : + SeqMSE weight-encoding search before calibration
  spinquant         : + SpinQuant rotations (R1+R2) on the fp model before QuantSim
                      (R3 is not exposed, which sidesteps the GQA shape issue)
  spinquant_seqmse  : both

Speed: the model is quantized IN PLACE, so the eval uses the model's own cached
``.generate()`` (the same path the fp16/nf4 runs used) instead of a hand-rolled
decode. Quantization is traced through a thin logits-only wrapper (AIMET wants a
tensor out; transformers 5.x Qwen3 needs return_dict internally).

Version note: aimet-torch 2.39 needs transformers in roughly the 5.9-5.16 window.
Below that, AIMET's jit trace trips a masking_utils back-compat branch
(IndexError in sdpa_mask); at 5.17+ its qwen2_5_vl integration fails to import
(renamed class). Verified locally: 5.10.4 and 5.12.1 work, 5.8.1 and 5.17.0 don't.

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


def untie_lm_head(model) -> None:
    """SpinQuant rotates embed_tokens and lm_head differently, so tied weights
    (Qwen3-0.6B ties them) must be split first; apply_spinquant raises otherwise."""
    import torch

    if model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
        model.config.tie_word_embeddings = False
        print("untied lm_head from embed_tokens for SpinQuant")


def main() -> None:
    ap = argparse.ArgumentParser(description="AIMET W4A16 quantize + LOTO eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tools", default=str(HERE / "tools.json"))
    ap.add_argument("--data", default=str(HERE / "mini_eval.jsonl"))
    ap.add_argument("--calib-file", required=True, help="jsonl to calibrate on (train.jsonl)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--remedy", choices=["rtn", "seqmse", "spinquant", "spinquant_seqmse"],
                    default="rtn")
    ap.add_argument("--granularity", choices=["per_channel", "blockwise", "lpbq"],
                    default="lpbq",
                    help="weight-quantizer granularity; lpbq is what QNN runs on the HTP")
    ap.add_argument("--block-size", type=int, default=64,
                    help="input-channel block size for blockwise/lpbq (nf4 proxy used 64)")
    ap.add_argument("--param-bw", type=int, default=4)
    ap.add_argument("--output-bw", type=int, default=16)
    ap.add_argument("--config-file", default="default",
                    help="AIMET config alias/path. 'default' already means per-channel "
                         "in aimet 2.39 (default_config_per_channel.json); 'htp_v79' is "
                         "the 8 Elite HTP config")
    ap.add_argument("--quant-scheme", default="min_max",
                    help="min_max matches nf4's absmax; post_training_tf_enhanced clips outliers")
    ap.add_argument("--n-seqmse", type=int, default=32,
                    help="calibration prompts used by SeqMSE (subset of --n-calib)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aimet_torch import QuantizationSimModel
    from aimet_torch.common.defs import QuantScheme

    label = args.label or f"specialist_w4a16_{args.remedy}"
    device = args.device

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
    # smoke is garbage, the quantization is the problem, not the generate path.
    base.config.use_cache = True
    _smoke = GenRunner(base, tokenizer, device, args.max_new_tokens)
    print("pre-quant  smoke:", repr(_smoke.generate(eval_items[0]["query"], all_tools)[:100]))

    base.config.use_cache = False  # off for the AIMET trace; flipped back on for eval

    if args.remedy.startswith("spinquant"):
        from aimet_torch.experimental.spinquant import apply_spinquant
        untie_lm_head(base)
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

    # Standard W4A16 carve-out: keep the LM head and token embeddings out of 4-bit.
    # Must happen before the granularity pass below, which skips None quantizers.
    excluded = []
    for name, module in sim.model.named_modules():
        if (name.endswith("lm_head") or name.endswith("embed_tokens")) and hasattr(module, "param_quantizers"):
            for key in list(module.param_quantizers.keys()):
                module.param_quantizers[key] = None
            excluded.append(name)
    if excluded:
        print("excluded from quant:", excluded)
    else:
        cand = [n for n, _ in sim.model.named_modules() if "head" in n.lower() or "embed" in n.lower()]
        print("excluded NONE; candidate module names:", cand)

    # The other half of the W4A16 contract: 4-bit applies to matmul weights ONLY.
    # AIMET's config defaults ("params": is_quantized) re-enable a PER-TENSOR 4-bit
    # quantizer on every RMSNorm gamma (q_norm/k_norm included), overriding the
    # None that aimet's QuantizedQwen3RMSNorm.__quant_init__ sets. Norm gammas have
    # a huge dynamic range, and per-tensor int4 on them collapses generation
    # regardless of what the Linears do. llm-compressor and the Qualcomm ai-hub
    # llama recipes both keep norm weights at high precision.
    n_norm = 0
    for name, module in sim.model.named_modules():
        if hasattr(module, "param_quantizers") and not isinstance(module, torch.nn.Linear):
            pq = dict(module.param_quantizers)
            if "weight" in pq and pq["weight"] is not None:
                module.param_quantizers["weight"] = None
                n_norm += 1
    print(f"norm weights kept fp: {n_norm} modules")

    if args.granularity in ("blockwise", "lpbq"):
        from aimet_torch.quantsim.config_utils import (
            set_blockwise_quantization_for_weights,
            set_grouped_blockwise_quantization_for_weights,
        )
        if args.granularity == "blockwise":
            set_blockwise_quantization_for_weights(
                sim, [torch.nn.Linear], args.param_bw, True, args.block_size)
        else:
            set_grouped_blockwise_quantization_for_weights(
                sim, [torch.nn.Linear], args.param_bw, True, 8, args.block_size, -1)

    # Trust nothing: print the weight quantizer actually attached to a transformer
    # Linear. per_channel must show shape (out, 1); blockwise/lpbq must show a real
    # block_size. The earlier collapse hunt burned days on a config that "looked"
    # applied, so this is a hard check now.
    probe = None
    for name, module in sim.model.named_modules():
        if name.endswith("layers.0.self_attn.q_proj"):
            probe = module.param_quantizers["weight"]
            print(f"q_proj weight quantizer: {type(probe).__name__} shape={tuple(probe.shape)} "
                  f"block_size={getattr(probe, 'block_size', None)} bw={probe.bitwidth}")
    if probe is None:
        raise RuntimeError("could not find layers.0.self_attn.q_proj to verify granularity")
    if args.granularity in ("blockwise", "lpbq") and getattr(probe, "block_size", None) is None:
        raise RuntimeError(f"{args.granularity} requested but weight quantizer has no block_size")
    if args.granularity == "per_channel" and len([d for d in probe.shape if d > 1]) == 0:
        raise RuntimeError("per_channel requested but weight quantizer is per-tensor")

    if args.remedy.endswith("seqmse"):
        from torch.utils.data import DataLoader
        from aimet_torch.seq_mse import apply_seq_mse
        # SeqMSE stacks cached activations across batches, so every batch must have
        # the same sequence length: truncate the subset to its shortest prompt.
        subset = calib_inputs[: args.n_seqmse]
        seq_len = min(t.shape[1] for t in subset)
        subset = [t[:, :seq_len] for t in subset]
        loader = DataLoader(subset, batch_size=None)
        print(f"applying SeqMSE over {len(subset)} prompts @ {seq_len} tokens...")
        apply_seq_mse(sim, loader, num_candidates=20)

    def forward_pass(m) -> None:
        with torch.no_grad():
            for inp in calib_inputs:
                m(inp)

    print(f"computing encodings ({args.remedy}, {args.granularity}, {args.quant_scheme})...")
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
            "remedy": args.remedy, "granularity": args.granularity, "block_size": args.block_size,
            "config_file": args.config_file, "quant_scheme": args.quant_scheme}
    (out / f"{label}.json").write_text(json.dumps({**meta, **outcome}, indent=2))

    s = outcome["summary"]["strict"]
    print(f"\n=== {label}  (W{args.param_bw}A{args.output_bw} {args.granularity} {args.remedy}) ===")
    print(f"  call_accuracy {s['call_accuracy']}  schema {s['schema_valid_rate']}  "
          f"right_tool {s['right_tool_rate']}  irrelevance {s['irrelevance_accuracy']}")
    print()
    print(format_loto(outcome["loto"]["strict"]))
    print(f"\nwrote {out / f'{label}.json'}")


if __name__ == "__main__":
    main()
