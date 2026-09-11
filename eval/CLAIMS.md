# Claims ledger

A living record of what this study can assert and what evidence each claim still
owes. Updated as runs land.

Numbers below are from the pilot (48 items, one greedy pass per config), so
single-config deltas are noisy: the 95 percent interval at 40 call items is
about 9 points. Treat size effects (tens of points) as real and precision
effects (a few points) as not yet established.

## Claims

### 1. Capability floor around 1.7B
- Status: established (pilot), consistent with external work (TinyLLM, 2025).
- Evidence: fp16 call accuracy 0.55 (0.6B) vs 0.875 (1.7B) vs 0.925 (4B). The 0.6B gap is far outside the noise band.
- Kill criterion: if constrained decoding lifts 0.6B close to 1.7B, this was a format floor, not a capability floor, and the claim is reworded.
- Still owed: larger n; decide whether the floor is capability or format (needs the audit and the constrained-decoding run).

### 2. Memory Pareto: a 4-bit larger model dominates an fp16 smaller one
- Status: established for nf4.
- Evidence: 4B-nf4 (about 2.3 GB) matches 1.7B-fp16 (about 3.4 GB) on accuracy (0.90 vs 0.875, inside noise) at roughly two-thirds the memory.
- Kill criterion: the real W4A16 or W4A8 contract dropping 4B below 1.7B-fp16.
- Still owed: re-check under the on-device contract, not nf4.

### 3. Small-model failures are format-dominated, not reasoning-dominated
- Status: directional (pilot), to confirm by audit.
- Evidence: 0.6B schema-valid 0.625 while right-tool 0.60, so most misses are malformed or absent calls, not wrong tools. Same pattern at 4B-nf4 (correct intent emitted as `.create_note {...}`).
- Kill criterion: audit showing most small-model misses are wrong-tool or wrong-arg rather than format.
- Still owed: the failure-taxonomy audit across all sizes.

### 4. 4-bit weight quantization is nearly free for tool calling
- Status: open.
- Evidence: flat fp16-vs-nf4 at every size, but nf4 is a gentle proxy (float4 weights, fp16 activations), not the on-device contract.
- Kill criterion: a real cliff appearing under W4A16 round-to-nearest or lower.
- Still owed: AIMET W4A16, then SeqMSE, then SpinQuant, and pushing to W4A8 / int8 KV.

### 5. Constrained decoding recovers the format gap
- Status: open.
- Evidence: none yet; the strict-vs-lenient split sizes the opportunity (lenient recovers the name-outside-JSON cases).
- Kill criterion: lenient parse recovering little over strict, meaning the failures are semantic, not format.
- Still owed: implement constrained decoding, measure recovery per size and precision.

### 6. The GPU proxy predicts on-device behavior
- Status: open.
- Evidence: none; the study is GPU-simulated so far.
- Kill criterion: on-device W4A16 numbers diverging from the GPU proxy on a shared subset.
- Still owed: run a subset through the real AI Hub bundle on the 8 Elite.

## Prior art this updates

A 2025 Qualcomm-funded study found format (JSON parsability) is the make-or-break
for small-model function calling on a Snapdragon 8 Gen 2, using GGUF Q4_K_M and a
fine-tuning fix. This work revisits the question on the 8 Elite, under the W4A16
NPU contract, with a training-free fix (constrained decoding).
