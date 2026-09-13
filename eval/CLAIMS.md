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

### 4. 4-bit weight quantization is free for tool calling WITH THE RIGHT METHOD, not for free
- Status: established under the device-representative W4A16 contract (AIMET QuantSim, LPBQ int4 block-64, A16, lm_head/embeddings/norm gammas fp, 128 calibration prompts). This replaces the nf4 proxy and sharpens the old "nearly free" wording.
- Evidence, specialist (0.6B) on the phone LOTO set (n=48):

  | config | call_acc | schema | no-call |
  |---|---|---|---|
  | fp16 | 0.90 | 1.00 | 1.00 |
  | nf4 (bnb, NF4 datatype, block-64) | 0.925 | 1.00 | 1.00 |
  | W4A16 int4 RTN | 0.775 | 0.975 | 1.00 |
  | W4A16 int4 + SpinQuant (R1+R2) | 0.600 | 0.85 | 0.875 |
  | W4A16 int4 + SeqMSE | 0.925 | 1.00 | 1.00 |

- Reading: three findings. (a) Real int4 RTN costs 12.5 points where the nf4 proxy showed a free lunch; "4-bit" is not one thing, the NF4 datatype's nonlinear grid was doing real work. (b) SeqMSE recovers all of it (0.925, equal to nf4, above fp16), so the on-device claim survives, but only method-conditional. (c) Remedies are not monotonic: SpinQuant, applied under blockwise quant, made things worse (rotation spreads outliers that block-64 already handles locally, and its RMSNorm-gamma fusion widens the range inside each 64-column block; R2 here is an untrained Hadamard). Rotation is a coarse-granularity remedy, not a universal one.
- Failure anatomy: RTN's entire loss is tool selection (schema 0.975, no-call 1.00): six items collapse onto set_reminder, the most over-represented training tool (334 examples), with create_note (second-rarest, 112) falling 1.00 to 0.25. Quantization loses fine discrimination first and falls back to the highest-prior tool, echoing the note/reminder confusion from the data audit. SpinQuant's loss is diffuse instead: dropped <tool_call> tags, hallucinated tool names, argument corruption.
- Kill criterion: the on-device (QNN/HTP) run diverging from this QuantSim proxy.
- Still owed: ONNX-QDQ export and on-device replay of the SeqMSE config; W4A8 / int8 KV; optional cross-check that SpinQuant does rescue per-channel granularity (would complete the rotation-granularity interaction story).

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
- 4-bit replays, both landed: nf4 holds (0.925, schema 1.00, no-call 1.00), and the device-representative W4A16 holds at 0.925 with SeqMSE (schema 1.00, no-call 1.00, seen 0.93 vs unseen 0.92, gap +0.01, the tightest of any config). Method matters though: W4A16 RTN drops to 0.775 by over-routing confusables to set_reminder, and SpinQuant to 0.600. See Claim 4 for the ladder.
- Still owed: the on-device replay of the W4A16 SeqMSE config through the QNN/HTP runtime; widen n beyond 48 (per-tool cells are 4 items); residual misses under SeqMSE are two routing items (settings_03, sms_01) plus the recurring quarter-past-eight time parse (alarm_04).

## Decisions forced by the data
- App model: 1.7B (nf4 ~0.85 GB, or fp16 ~3.4 GB), not 4B. 4B was memory-marginal on the S25's 12 GB in the device proof and gives no accuracy gain here.

## Prior art this updates

A 2025 Qualcomm-funded study found format (JSON parsability) is the make-or-break
for small-model function calling on a Snapdragon 8 Gen 2, using GGUF Q4_K_M and a
fine-tuning fix. This work revisits the question on the 8 Elite, under the W4A16
NPU contract, with a training-free fix (constrained decoding).
