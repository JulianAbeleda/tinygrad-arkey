# Decode FFN Activation Producer Fusion Result (Phase B1)

Date: 2026-06-20

Scope: `docs/decode-fusion-build-scope-20260620.md` Phase B (Candidate B1).

Verdict: `BUILT_EXACT_BUT_NO_WIN` — the fusion is **byte-exact** and eliminates the standalone `E_49152` launch,
but yields **zero speedup**. This **refutes the Deliverable-3 "launch-overhead-bound, fusion-recoverable"
hypothesis**: the FFN activation cost is *conserved work*, not launch overhead. Default decode behavior NOT
changed (built/tested in a standalone A/B harness; `tinygrad/llm/model.py` untouched).

## What was built

A fused "up" GEMV that writes `silu(gate[row]) * (Σ w·x)` directly at its store, so the `silu(gate)*up`
activation never round-trips through the standalone `E_49152_32_3` elementwise kernel.

- `extra/q4_k_gemv_primitive.py`: two new kernel generators (research-only, never wired into the model):
  - `q4k_gemv_silu_gate_kernel` — REG accumulator over a flattened k-reduce + activated store.
  - `q4k_gemv_silu_gate_v2_kernel` — keeps `q4k_gemv_partial`'s fast nested blk+pos buffer-accumulator
    (scratch buffer) and applies the activation at the final store.
- `extra/qk_decode_ffn_activation_producer_fusion_ab.py`: standalone clock-pinned A/B.

## Results (real `blk.0.ffn_gate`/`ffn_up` weights, clock-pinned, isolated gate+up+act lifecycle)

| variant | opts | correctness (rel) | baseline µs | fused µs | delta |
|---|---|---:|---:|---:|---:|
| v1 (REG, flattened) | LOCAL:0:64 | **0.0e0 (byte-exact)** | 172 | 171 | +0.3% |
| v1 | LOCAL:0:128 | 0.0e0 | 172 | 171 | +0.1% |
| v1 | LOCAL:0:256 | 0.0e0 | 174 | 185 | −6.0% |
| v2 (buffer-accum, scratch) | LOCAL:0:64 | **0.0e0 (byte-exact)** | 168 | 167 | +0.3% |
| v2 | LOCAL:0:128 | 0.0e0 | 168 | 168 | −0.1% |

Correctness is byte-identical (the UOp `silu = g/(1+exp(-g))` matches `Tensor.silu()` bit-for-bit here).
**Net timing: ~0% at every clean config** (delta within measurement noise).

## Why it does not win (the decisive mechanism)

The baseline lifecycle is `gate_GEMV(~51µs) + up_GEMV(~51µs) + E_49152(~33µs)`. The fused lifecycle is
`gate_GEMV(~51µs) + fused_up_GEMV(~84µs)` — the fused up GEMV is ~33µs slower, exactly the E_49152 it replaced.
The `silu(gate)*up` **work is conserved**: it just moves from a standalone launch into the GEMV's store epilogue,
where it runs **serially after** the reduce (same ~33µs). The launch itself was negligible (decode is
GPU-execution-bound, D≈W, host-sync 0% — confirmed in Deliverable 0).

This refutes the Deliverable-3 inference that `E_49152`'s ~33µs/call was recoverable launch overhead. **It is real
work** (the gate load + sigmoid + write). llama gets the activation effectively "free" because its `mul_mat_vec_q`
is HBM-bandwidth-bound and the activation ALU hides under the memory-load latency. Recovering it in tinygrad would
require **interleaving the activation into the GEMV's inner memory-load loop so it overlaps the HBM stalls** — a
latency-hiding kernel-scheduling problem, not a fusion. Naive fusion (this build) cannot recover it.

## Gate status

| gate | result |
|---|---|
| `E_49152` disappears | YES (folded into the up GEMV) |
| correctness exact | YES (rel 0.0, byte-identical) |
| elementwise recovers ≥0.5 ms/token @1024 | **NO** (~0% — work conserved) |

Local gate FAILED. Per the scope's stop condition ("If producer fusion … materially worsens GEMV time, stop"):
the fused GEMV does not worsen, but it does not improve either — the activation cost reappears inside it. **Stop;
do not pursue producer/prologue fusion variants** (B2 `ffn_down` prologue would conserve the same work; B3 q8 is
the same activation). The only path to the ~1.24 ms is activation/HBM-latency overlap inside the GEMV inner loop.

## Commands

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_ffn_activation_producer_fusion_ab.py
```

## Artifacts

- `extra/q4_k_gemv_primitive.py` (`q4k_gemv_silu_gate_kernel`, `q4k_gemv_silu_gate_v2_kernel`)
- `extra/qk_decode_ffn_activation_producer_fusion_ab.py`
- `bench/qk-decode-fusion-build/ffn_activation_producer_fusion_ab.json`

## Boundary

No decode default changed. The fused kernels live in `extra/` and are not wired into `tinygrad/llm/model.py`.
Clock pinned for measurement; `auto` restored after (verified).
