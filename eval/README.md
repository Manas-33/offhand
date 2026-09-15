# Tool-calling reliability eval

Measures whether quantization breaks the model's ability to call phone tools
correctly.

The harness is quantization agnostic. It scores the tool-calling behaviour of
whatever model a runner exposes. Swap the runner (FP16, a 4-bit proxy, a custom
W4A16 checkpoint, or the on-device bridge) and the tools, dataset, and scoring
all stay identical.

## What it measures

For every item in [`mini_eval.jsonl`](mini_eval.jsonl) the model sees the phone
tools in [`tools.json`](tools.json) and one user query. We parse its tool call
and score four separate signals:

| signal | question |
|---|---|
| `schema_valid_rate` | valid tool name, required args present, no junk params |
| `right_tool_rate`   | picked the correct function |
| `call_accuracy`     | correct function and correct arguments |
| `irrelevance_accuracy` | stayed quiet on questions with no matching tool |

Keeping them separate is the point. Quantization tends to hurt schema validity
and argument accuracy first, and grammar-constrained decoding is meant to bring
schema validity back toward 1.0.

## The 400-question test set

[`phone_eval.jsonl`](phone_eval.jsonl) is the held-out test set, with the same
tools and answer-key format as `mini_eval.jsonl` at ten times the size:

- 300 single-call items, 30 per tool. The three tools the specialist never
  trained on get 90 of them.
- 100 no-call items: 40 general questions, 30 near misses that ask for
  something no tool can do (reading, editing or deleting existing alarms,
  events, texts or notes), and 30 actions no tool covers (calling someone,
  ordering food, smart-home control).

Each item carries a `subtype`: `direct`, `indirect` or `tricky` for calls, and
`general`, `near_miss` or `unserviceable` for no-calls.

It was written independently of the training data: the writers saw only
`tools.json` and `mini_eval.jsonl`. Every item then passed
[`check_eval_set.py`](check_eval_set.py), which checks that answer keys are
valid and winnable, that nothing repeats, and that nothing is close to a
training question. Separate reviewers (model agents that saw only the question
and the tools) then labeled every item blind. Their calls matched the answer
keys on all 400 items.

With 40 call items, `mini_eval.jsonl` only separates large differences, since
each score carries a 95% interval of about ±10 points. Fine-grained
comparisons, such as two models on the phone, should use `phone_eval.jsonl`.

```bash
python eval/check_eval_set.py eval/phone_eval.jsonl --other eval/mini_eval.jsonl --train <training jsonl files>
python eval/run_eval.py --model <model> --config fp16 --data eval/phone_eval.jsonl --loto --label <label>
```

## Run it

Verify the harness with no model download (uses a mock runner):

```bash
python eval/run_eval.py --dry-run
python eval/tests/test_harness.py
```

Real runs (GPU box, `pip install -r eval/requirements.txt`):

```bash
python eval/run_eval.py --model Qwen/Qwen3-4B --config fp16 --label fp16
python eval/run_eval.py --model Qwen/Qwen3-4B --config int8 --label w8_int8
python eval/run_eval.py --model Qwen/Qwen3-4B --config nf4  --label w4_nf4
```

Local (Apple Silicon) fp16 smoke test with a small model:

```bash
python eval/run_eval.py --model Qwen/Qwen3-0.6B --config fp16 --device mps --label local_smoke
```

Each writes `eval/results/<label>.json` (summary plus per-item raw output and
scores). Compare the `call_accuracy` line across labels to see how precision
affects reliability.

> Note: `int8` and `nf4` use bitsandbytes, which is CUDA only. They do not run
> on a Mac. The fp16 vs quantized comparison has to run on a GPU.

## Proxies vs the real numbers

`int8` and `nf4` are bitsandbytes proxies for a fast signal. The on-device
target is W4A16: AIMET QuantSim calibration, then SeqMSE/AdaScale, then SpinQuant
rotations. Those produce checkpoints that are evaluated through a new runner
satisfying the same `ModelRunner.generate` interface in
[`offhand_eval/runners.py`](offhand_eval/runners.py).

## Layout

```
eval/
  tools.json              10 phone-tool schemas (reused in the app system prompt)
  mini_eval.jsonl         48 items: 40 single-call (4 x 10 tools) + 8 irrelevance
  phone_eval.jsonl        400 items, the test set: 300 single-call (30 x 10 tools) + 100 no-call
  check_eval_set.py       checks an eval set's answer keys, repeats and overlap with training
  run_eval.py             CLI
  offhand_eval/
    parse.py              extract tool calls from raw model text
    scoring.py            the four signals plus aggregation
    runners.py            ModelRunner protocol, MockRunner, HFRunner
    mock.py               reference-correct mock runner (for tests and --dry-run)
    harness.py            run loop
    dataset.py            load tools and items
  tests/test_harness.py   runs with no ML deps
```

## Known limitations

- Argument matching is exact or substring against accepted-value lists.
  Free-text slots (email and note bodies) are scored on the discriminating slot
  only.
- Single-call items only. Multi-step chains are not covered yet.
- 18 of the 48 `mini_eval.jsonl` questions are close to training questions (9
  word for word), all on tools the specialist trained on. Model rankings hold
  on its 24 clean call questions, but headline numbers should come from
  `phone_eval.jsonl`, which has no overlap.
- `nf4` and `int8` are not the on-device W4A16 numerics. They are proxies for
  signal, not the reported result.
