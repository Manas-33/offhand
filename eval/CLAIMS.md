# Claims ledger

A living record of what this study can assert and what evidence each claim still
owes. Updated as runs land.

Two evidence sets so far:
- Pilot: 48 phone-tool items, one greedy pass per config. Deltas are noisy (the
  95 percent interval is about 9 points at 40 call items).
- BFCL: 840 items (400 simple + 200 multiple + 240 irrelevance), one greedy pass
  per config, with bootstrap CIs over items.

## Claims

### 1. Capability floor around 1.7B
- Status: established (BFCL + pilot), consistent with TinyLLM (2025).
- Evidence: BFCL fp16 call accuracy 0.807 [.775,.838] (0.6B) to 0.910 [.887,.932] (1.7B) to 0.908 [.885,.932] (4B). The 0.6B and 1.7B intervals do not overlap (a real ~10-point jump), and 1.7B to 4B is flat.
- Kill criterion: constrained decoding lifting 0.6B close to 1.7B would recast this as a format floor, not a capability floor.
- Still owed: the phone-eval constrained-decoding run settles format-vs-capability at the low end.

### 2. Memory Pareto: 1.7B owns the frontier, 4B is not worth it
- Status: REVISED by BFCL. The pilot's "quantized-4B dominates fp16-1.7B" did not survive at scale.
- Evidence (accuracy vs weights memory): frontier is 1.7B-nf4 (0.85 GB, 0.872) and 1.7B-fp16 (3.4 GB, 0.910). 4B-nf4 (2.0 GB, 0.880) is beaten by 1.7B-nf4, and 4B-fp16 (8 GB, 0.908) by 1.7B-fp16. 4B buys no accuracy over 1.7B on this task.
- Kill criterion: a task where 4B separates from 1.7B (multi-step chains) would restore a reason to pay for 4B.
- Still owed: check whether chains re-open a 4B advantage.

### 3. Small-model failures are format-dominated on phone tools, not on BFCL
- Status: split result.
- Evidence: on the phone-tool pilot, small/quantized failures were mostly malformed calls (name outside JSON). On BFCL, format_break = 0 across all six runs; failures are wrong-arg or wrong-tool, not format.
- Reading: format brittleness is distribution-specific (the phone setup: 10 tools, free-text bodies), not a property of the models on standard function calling.
- Still owed: quantify the phone-tool format gap at larger n.

### 4. 4-bit weight quantization is nearly free for tool calling
- Status: supported (nf4 proxy).
- Evidence: BFCL nf4 vs fp16 costs 2 to 4 points per size, CIs mostly overlapping (borderline only at 1.7B: 0.910 vs 0.872).
- Kill criterion: a real cliff under the on-device W4A16 contract.
- Still owed: AIMET W4A16 (round-to-nearest, then SeqMSE, then SpinQuant), and W4A8 / int8 KV, to replace the nf4 proxy with the real numerics.

### 5. Constrained decoding recovers the format gap
- Status: open, rescoped to the phone-tool eval.
- Evidence: BFCL is format-clean, so there is nothing to recover there. The gap exists only on the phone-tool distribution.
- Kill criterion: little strict-vs-lenient gap even on the phone eval, meaning failures are semantic, not format.
- Still owed: build constrained decoding, run it on a larger phone-tool eval, measure recovery.

### 6. The GPU proxy predicts on-device behavior
- Status: open (device runs, accuracy not yet compared).
- Evidence: device proof passed (Qwen3-4B W4A16 on the 8 Elite NPU: ~73ms TTFT, ~22 tps decode), but on-device accuracy is not yet compared to the GPU proxy.
- Still owed: run a subset through the on-device runtime and correlate.

### 7. Irrelevance / over-triggering
- Status: measured (BFCL).
- Evidence: irrelevance accuracy 0.80 to 0.87 across models (13 to 20 percent false tool calls on 240 hard negatives), slightly worse when smaller. A safety metric for an agent with side effects.
- Still owed: whether constrained decoding or a stricter prompt cuts false triggers without hurting call accuracy.

## Decisions forced by the data
- App model: 1.7B (nf4 ~0.85 GB, or fp16 ~3.4 GB), not 4B. 4B was memory-marginal on the S25's 12 GB in the device proof and gives no accuracy gain here.

## Prior art this updates

A 2025 Qualcomm-funded study found format (JSON parsability) is the make-or-break
for small-model function calling on a Snapdragon 8 Gen 2, using GGUF Q4_K_M and a
fine-tuning fix. This work revisits the question on the 8 Elite, under the W4A16
NPU contract, with a training-free fix (constrained decoding).
