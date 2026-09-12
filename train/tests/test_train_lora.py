"""Smoke-test the LoRA data/masking path with the real tokenizer (no GPU, no peft).

Verifies the thing most likely to silently break training: that the prompt is
masked and the completion (target + eos) is the only thing with loss, and that
the tools/system prompt actually land in the prompt. Skips cleanly if the
tokenizer can't be loaded (offline). Run: `python train/tests/test_train_lora.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # train/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))    # eval/

import train_lora
from offhand_eval.dataset import load_tools, tools_by_name

BASE = "Qwen/Qwen3-0.6B"
TOOLS = tools_by_name(load_tools(str(Path(train_lora.HERE).parent / "eval" / "tools.json")))

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


def test_pad_collator():
    out = train_lora.PadCollator(pad_token_id=0)([
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 1]},
    ])
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"][1].tolist() == [-100, 5, -100]       # pad position masked from loss
    assert out["attention_mask"][1].tolist() == [1, 1, 0]     # and from attention


def _run_all():
    test_pad_collator()
    print("  PASS  test_pad_collator")
    tok = _load_tokenizer()
    if tok is None:
        print("tokenizer-dependent tests skipped.")
        return
    for name in ("test_completion_only_masking", "test_prompt_carries_tools_and_system", "test_no_call_example"):
        globals()[name](tok)
        print(f"  PASS  {name}")
    print("all train_lora tests passed.")


if __name__ == "__main__":
    _run_all()
