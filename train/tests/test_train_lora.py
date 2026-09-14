"""Smoke-test the LoRA data/masking path with the real tokenizer (no GPU, no peft).

Verifies the thing most likely to silently break training: that the prompt is
masked and the completion (target + eos) is the only thing with loss, and that
the tools/system prompt actually land in the prompt. Skips cleanly if the
tokenizer can't be loaded (offline). Run: `python train/tests/test_train_lora.py`.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # train/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))    # eval/

import train_lora
from offhand_eval.dataset import load_tools, tools_by_name

BASE = "Qwen/Qwen3-0.6B"
PHONE_TOOLS = load_tools(str(Path(train_lora.HERE).parent / "eval" / "tools.json"))
TOOLS = tools_by_name(PHONE_TOOLS)
ALL_TOOLS = tools_by_name(PHONE_TOOLS + load_tools(str(Path(train_lora.HERE) / "general_tools.json")))

TOOL_ITEM = {
    "intended_tool": "set_alarm",
    "query": "wake me up at 7am",
    "tools_listed": ["set_timer", "set_alarm", "create_note"],
    "target": '<tool_call>{"name": "set_alarm", "arguments": {"time": "07:00"}}</tool_call>',
}
NO_CALL_ITEM = {
    "intended_tool": None,
    "query": "what's the capital of France?",
    "tools_listed": ["set_alarm", "play_music", "draft_email"],
    "target": "The capital of France is Paris.",
}
INFO_ITEM = {
    "kind": "info_call",
    "intended_tool": "fx_rate",
    "query": "what's the exchange rate from dollars to euros?",
    "tools_listed": ["set_alarm", "fx_rate", "local_time"],
    "target": '<tool_call>{"name": "fx_rate", "arguments": {"base": "USD", "quote": "EUR"}}</tool_call>',
}


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as exc:  # noqa: BLE001 - offline / missing dep -> skip, don't fail
        print(f"SKIP (tokenizer unavailable): {type(exc).__name__}: {exc}")
        return None


def test_completion_only_masking(tok):
    ex = train_lora.build_example(tok, TOOL_ITEM, TOOLS, max_length=1024)
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    assert set(ex["attention_mask"]) == {1}

    masked = [lab == -100 for lab in ex["labels"]]
    assert any(masked) and not all(masked), "need a masked prompt and an unmasked completion"
    # the mask must be a clean prefix (prompt), then the completion
    first_real = masked.index(False)
    assert all(masked[:first_real]) and not any(masked[first_real:]), "mask must be a prefix"

    completion_ids = [lab for lab in ex["labels"] if lab != -100]
    assert ex["input_ids"][first_real:] == completion_ids
    assert TOOL_ITEM["target"] in tok.decode(completion_ids)  # target is what carries loss


def test_prompt_carries_tools_and_system(tok):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": train_lora.DEFAULT_SYSTEM_PROMPT},
         {"role": "user", "content": TOOL_ITEM["query"]}],
        tools=[TOOLS[n] for n in TOOL_ITEM["tools_listed"]],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    assert "Offhand" in prompt          # the shared eval system prompt
    assert "set_alarm" in prompt        # a listed tool is shown
    assert "draft_sms" not in prompt    # an unlisted tool is not


def test_no_call_example(tok):
    ex = train_lora.build_example(tok, NO_CALL_ITEM, TOOLS, max_length=1024)
    completion_ids = [lab for lab in ex["labels"] if lab != -100]
    assert "Paris" in tok.decode(completion_ids)


def test_informational_tool_example(tok):
    ex = train_lora.build_example(tok, INFO_ITEM, ALL_TOOLS, max_length=2048)
    first_real = [lab == -100 for lab in ex["labels"]].index(False)
    prompt = tok.decode(ex["input_ids"][:first_real])
    assert "fx_rate" in prompt and "local_time" in prompt  # listed informational tools are shown
    assert INFO_ITEM["target"] in tok.decode([lab for lab in ex["labels"] if lab != -100])


def test_pad_collator():
    out = train_lora.PadCollator(pad_token_id=0)([
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 1]},
    ])
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"][1].tolist() == [-100, 5, -100]       # pad position masked from loss
    assert out["attention_mask"][1].tolist() == [1, 1, 0]     # and from attention


def test_item_kind_and_filter():
    items = [TOOL_ITEM, NO_CALL_ITEM, INFO_ITEM, {**NO_CALL_ITEM, "kind": "near_miss_phone"}]
    # v1 items have no kind field, so it is derived from intended_tool
    assert [train_lora.item_kind(it) for it in items] == ["phone_call", "no_call", "info_call", "near_miss_phone"]
    assert train_lora.filter_kinds(items, {"info_call", "near_miss_phone"}) == [TOOL_ITEM, NO_CALL_ITEM]


def test_load_train_files_concatenates():
    paths = []
    for rows in ([TOOL_ITEM, NO_CALL_ITEM], [INFO_ITEM]):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n\n")  # trailing blank line
            paths.append(fh.name)
    try:
        items = train_lora.load_train_files(paths)
        assert [it["query"] for it in items] == [TOOL_ITEM["query"], NO_CALL_ITEM["query"], INFO_ITEM["query"]]
    finally:
        for path in paths:
            os.unlink(path)


def test_parse_keep_fracs():
    assert train_lora.parse_keep_fracs(["info_call=0.25", " near_miss_general =1"]) == {
        "info_call": 0.25, "near_miss_general": 1.0}
    for bad in ("info_call", "=0.5", "info_call=", "info_call=1.5"):
        try:
            train_lora.parse_keep_fracs([bad])
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


def test_subsample_kinds_spreads_over_tools():
    info = [{**INFO_ITEM, "intended_tool": tool, "query": f"{tool} question {n}"}
            for tool in ("fx_rate", "local_time", "stock_quote") for n in range(6)]
    items = [TOOL_ITEM, *info, NO_CALL_ITEM]
    kept = train_lora.subsample_kinds(items, {"info_call": 0.5}, seed=0)
    kept_info = [it for it in kept if train_lora.item_kind(it) == "info_call"]
    # the same share of every tool, other families untouched, order kept
    assert sorted(it["intended_tool"] for it in kept_info) == ["fx_rate"] * 3 + ["local_time"] * 3 + ["stock_quote"] * 3
    assert kept[0] == TOOL_ITEM and kept[-1] == NO_CALL_ITEM
    assert kept == [it for it in items if it in kept]
    assert kept == train_lora.subsample_kinds(items, {"info_call": 0.5}, seed=0)  # same seed, same draw
    assert train_lora.subsample_kinds(items, {"info_call": 1.0}, seed=0) == items
    assert train_lora.subsample_kinds(items, {"info_call": 0.0}, seed=0) == [TOOL_ITEM, NO_CALL_ITEM]


def _run_all():
    for name in ("test_pad_collator", "test_item_kind_and_filter", "test_load_train_files_concatenates",
                 "test_parse_keep_fracs", "test_subsample_kinds_spreads_over_tools"):
        globals()[name]()
        print(f"  PASS  {name}")
    tok = _load_tokenizer()
    if tok is None:
        print("tokenizer-dependent tests skipped.")
        return
    for name in ("test_completion_only_masking", "test_prompt_carries_tools_and_system",
                 "test_no_call_example", "test_informational_tool_example"):
        globals()[name](tok)
        print(f"  PASS  {name}")
    print("all train_lora tests passed.")


if __name__ == "__main__":
    _run_all()
