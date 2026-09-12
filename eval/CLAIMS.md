# Claims ledger

A living record of what this study can assert and what evidence each claim still
owes. Updated as runs land.

Three evidence sets so far:
- Pilot: 48 phone-tool items, one greedy pass per config. Deltas are noisy (the
  95 percent interval is about 9 points at 40 call items).
- BFCL: 840 items (400 simple + 200 multiple + 240 irrelevance), one greedy pass
  per config, with bootstrap CIs over items.
- Specialist LOTO: the 48-item phone set again, split into 7 trained (seen) tools
  and 3 held-out (unseen) tools, to score the M1 distilled 0.6B specialist. Same
  small-n caveat as the pilot.

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
- Status: supported (nf4 proxy), including on the 0.6B specialist.
- Evidence: BFCL nf4 vs fp16 costs 2 to 4 points per size, CIs mostly overlapping (borderline only at 1.7B: 0.910 vs 0.872). Specialist (0.6B) on the phone LOTO set: nf4 0.925 vs fp16 0.90 call accuracy (a single item at n=48), schema validity and no-call both 1.00 at 4-bit, so the distilled specialist survives 4-bit intact.
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

### 8. A distilled 0.6B specialist clears the tool-calling floor and generalizes to unseen tools
- Status: established (M1 kill-test on the specialist LOTO set, n=48), pending 4-bit and on-device replays.
- Evidence: LoRA-distilling the 4B teacher's calls into Qwen3-0.6B lifts phone-tool call accuracy from 0.55 (base) to 0.90 and schema validity from 0.63 to 1.00. Held-out queries: seen tools (7 trained) 0.93, the unseen trio (never trained) 0.83 (gap +0.10), and schema validity is 1.00 even on the unseen tools. No-call abstention 1.00 for both.
- Reading: refines Claim 1. The ~1.7B floor is for a generalist caller; a 0.6B distilled onto a fixed tool set clears it, and the competence transfers to tools seen only in the prompt (unseen schema 1.00), so the low-end limit here is format/competence, not raw capacity or tool-specific memorization.
- Kill criterion: 4-bit (nf4, then W4A16) failing to preserve ~0.90, or the gap widening sharply at larger n, puts the floor back on capacity.
- Still owed: the nf4 replay holds (0.925, schema 1.00, no-call 1.00 at 4-bit); still owed the on-device W4A16 numerics (SpinQuant); widen n beyond 48 (per-tool cells are 4 items); residual misses are confusable siblings (calendar->reminder, sms->email) plus one AM/PM parse.

## Decisions forced by the data
- App model: 1.7B (nf4 ~0.85 GB, or fp16 ~3.4 GB), not 4B. 4B was memory-marginal on the S25's 12 GB in the device proof and gives no accuracy gain here.

## Prior art this updates

A 2025 Qualcomm-funded study found format (JSON parsability) is the make-or-break
for small-model function calling on a Snapdragon 8 Gen 2, using GGUF Q4_K_M and a
fine-tuning fix. This work revisits the question on the 8 Elite, under the W4A16
NPU contract, with a training-free fix (constrained decoding).
