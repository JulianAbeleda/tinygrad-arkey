# NVIDIA prefill Q4-down intermediate capture - C0 (2026-08-29)

## Verdict: BLOCKED / STOP

C0 cannot identify the first mismatch with the currently authorized harnesses.
The existing capture path can retain the exact saved-Z input, the compact-Q8
record, and the final post-epilogue FP32 output.  It cannot retain decoded Q4
metadata, per-K32 corrected subtotals, or the pre-epilogue accumulator without
changing the compiler-owned CUDA main kernel ABI/source.  Such a change would
be instrumentation of the production asset and is outside this diagnostic-only
packet.

## Existing boundary

`nv_q4down_capture_io.py` captures `blk.4`'s FP16 down input and final FP16
model output.  `nv_q4down_matched_ab.py` replays all 18 real GGML type-12
roles, invokes the compact-Q8 producer and compiler-generated Q4 main, and
compares final FP32 projection output against the installed FP16 control.

The Q4 binding allocates a single compact record, realizes the producer, then
launches the main with ABI `(out, record, words)`.  The only main output is the
final FP32 matrix.  There is no debug-output pointer, stage selector, or
intermediate buffer in `nv_compiler_q4k_down_pp512_binding.py` or the compiled
`DownAsset.main_program`.

## Retained versus unavailable stages

| Stage | Status | Evidence or boundary |
|---|---|---|
| compact-Q8 record | available | saved-Z capture v4 producer gate |
| decoded Q4 group metadata | unavailable | performed inside main from packed `words` |
| per-K32 corrected subtotal | unavailable | register-local accumulator inside main |
| pre-epilogue FP32 output | unavailable | no pre-epilogue store/ABI output |
| post-epilogue output | available | matched A/B final matrix |

The compact record is already independently qualified for q/scale/sum.  The
final mismatch remains reproducible (`max_abs=2.6956463`, `mean_abs=0.0053902`)
on the exact saved-Z fixture, but no first divergent stage can be proven from
the accessible tensors.

## Required unblock

Create a separate diagnostic CUDA main variant, with a distinct cache identity,
that accepts explicit debug buffers for decoded group metadata, each K32
subtotal, pre-epilogue output, and final output.  Compare one type-12 role
against the independent static/CPU oracle before considering any arithmetic or
geometry change.  Do not modify or promote the production binding until that
diagnostic variant identifies one owning expression.

No kernel correction, geometry tuning, model integration, timing claim, or
production ABI change was made in C0.
