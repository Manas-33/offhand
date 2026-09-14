#!/usr/bin/env python3
"""Train a LoRA tool-calling specialist on the teacher-labeled set (Qwen3-0.6B).

This is the distillation step: the 0.6B student learns the 4B teacher's tool-call
decisions on the phone tool set. Two properties matter most:

 - Same prompt as eval. The prompt is built with the SAME system prompt and the
   SAME ``apply_chat_template(..., enable_thinking=False)`` call the eval harness
   (HFRunner) uses, and the completion is ``target + eos``. So the model is
   trained on exactly the format it is scored and deployed on; nothing (not even
   the think-block handling) can drift between train and eval.
 - Completion-only loss. The prompt, including the tool schemas, is masked to
   -100; loss is computed only on the assistant response. The job is to PRODUCE
   the call (or the abstention), not to language-model the schema text.

After training it merges the adapter into the base and saves a standalone model
that ``HFRunner(model_id=<out>/merged)`` can load for the LOTO eval.

Run on Colab / a GPU box:
  python train/train_lora.py --train-file train/data/train.jsonl --out-dir train/out

With the hard call/no-call families (gen_data.py --hard), pass the v1 set and
the hard set together; --drop-kinds leaves families out for an ablation:
  python train/train_lora.py --train-file <v1>/train.jsonl <hard>/train_clean.jsonl --out-dir <out>/v2

Smoke the data/masking path locally (tokenizer only, no GPU, no peft):
  python train/tests/test_train_lora.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))

from offhand_eval.dataset import load_tools, tools_by_name  # noqa: E402
from offhand_eval.runners import DEFAULT_SYSTEM_PROMPT  # noqa: E402  (share the eval prompt)

# Qwen3 is Llama-style; LoRA on the attention + MLP projections is the usual set.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_example(tokenizer, item: dict, by_name: dict, max_length: int) -> dict:
    """Tokenize one item into (input_ids, labels, attention_mask) for causal LM.

    The prompt is reconstructed exactly as HFRunner builds it at eval time, so the
    student trains on the deployment format. The completion is the teacher target
    plus the EOS/turn-end token. Labels mask the whole prompt (-100) so loss falls
    only on the response.
    """
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": item["query"]},
    ]
    tools = [by_name[n] for n in item["tools_listed"]]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(item["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class PadCollator:
    """Right-pad a batch; pad positions are masked in labels (-100) and attention (0)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            gap = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * gap)
            labels.append(b["labels"] + [-100] * gap)
            attn.append(b["attention_mask"] + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def item_kind(item: dict) -> str:
    """The item's family. v1 items predate the field, so it is derived for them."""
    return item.get("kind") or ("no_call" if item["intended_tool"] is None else "phone_call")


def load_train_files(paths: list[str]) -> list[dict]:
    """Concatenate one or more jsonl training files, skipping blank lines."""
    return [json.loads(line) for path in paths for line in open(path, encoding="utf-8") if line.strip()]


def filter_kinds(items: list[dict], drop: set[str]) -> list[dict]:
    return [it for it in items if item_kind(it) not in drop]


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA-train the 0.6B tool-call specialist")
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train-file", nargs="+", default=[str(HERE / "data" / "train.jsonl")],
                    help="one or more jsonl files, concatenated")
    ap.add_argument("--tools", default=str(HERE.parent / "eval" / "tools.json"))
    ap.add_argument("--extra-tools", default=str(HERE / "general_tools.json"),
                    help="training-only tool schemas the hard families list")
    ap.add_argument("--drop-kinds", default="",
                    help="comma-separated item kinds to leave out (ablations)")
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="examples that reach this length are dropped, since their target would be cut")
    ap.add_argument("--max-samples", type=int, default=None, help="cap items (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true", help="save the adapter only, skip merge")
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    class _RowDataset(Dataset):  # map-style dataset so Trainer never second-guesses a bare list
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            return self.rows[idx]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tools = load_tools(args.tools)
    if args.extra_tools and Path(args.extra_tools).exists():
        tools += load_tools(args.extra_tools)
    by_name = tools_by_name(tools)

    drop = {k.strip() for k in args.drop_kinds.split(",") if k.strip()}
    items = filter_kinds(load_train_files(args.train_file), drop)
    if args.max_samples:
        items = items[: args.max_samples]
    print(f"training kinds: {dict(Counter(item_kind(it) for it in items))}"
          + (f" (dropped kinds: {sorted(drop)})" if drop else ""))

    examples = [build_example(tokenizer, it, by_name, args.max_length) for it in items]
    kept = [ex for ex in examples if len(ex["input_ids"]) < args.max_length]
    if len(kept) < len(examples):
        print(f"dropped {len(examples) - len(kept)} examples at --max-length {args.max_length} "
              "(their targets would be cut off)")
    examples = kept
    n_no_call = sum(1 for it in items if it["intended_tool"] is None)
    print(f"built {len(examples)} examples ({n_no_call} no-call) from {args.train_file}")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model.config.use_cache = False  # incompatible with training; re-enabled implicitly at inference
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # TrainingArguments kwargs drift across transformers versions (e.g. v5.2 drops
    # warmup_ratio in favor of warmup_steps); keep only what this version accepts.
    import inspect

    wanted = dict(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to="none",
        seed=args.seed,
    )
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(set(wanted) - valid)
    if dropped:
        print(f"note: this transformers ignores unsupported TrainingArguments {dropped}")
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in valid})
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=_RowDataset(examples),
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter -> {adapter_dir}")

    if not args.no_merge:
        merged = model.merge_and_unload()
        merged_dir = out / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"saved merged model -> {merged_dir}  (eval with HFRunner(model_id='{merged_dir}'))")


if __name__ == "__main__":
    main()
