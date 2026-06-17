# Decode "~6.5 ms copy/gather" diagnostic — RESOLVED: measurement artifact (2026-06-17)

The Phase-1 decode census attributed ~17% of decode GPU time to a single ~6.5 ms copy/gather kernel and flagged
it the highest-value diagnostic. This narrow diagnostic (`extra/qk_decode_copy_diagnostic.py`,
`bench/qk-decode-copy-diagnostic/result.json`) identifies it and proves it is **Bucket C: a measurement
artifact**, not real GPU work.

## 1. What the 6.5 ms kernel was
`copy        4 B,     AMD <- PYTHON` — a **4-byte** host→device copy, at **idx 0** (the first kernel of the
decode step), **0 | 0 GB/s** (zero bandwidth). tm reported ~6.45–6.48 ms.

## 2. What caused it
The per-step input upload: `Tensor([[token_id]])` / the `start_pos` scalar pushed to the GPU at the start of
each decode step. In the eager DEBUG=2 census it is the first kernel after `GlobalCounters.reset()`, so its
measured `tm` absorbs the **queue-drain / launch sync** at the step boundary.

## 3. Was it avoidable / necessary?
The 4-byte upload is **necessary and trivially cheap** (~µs physically). The **6.5 ms is NOT real cost** — a
4-byte transfer at 0 GB/s cannot consume 6.5 ms of GPU compute or bandwidth (physical impossibility). The figure
is a **sync/launch stall mismeasured as kernel `tm`** by the eager DEBUG=2 capture at the step boundary. Bucket
**C (measurement artifact)**.

Note on the causality micro-probe: timing `Tensor([[tok]]).contiguous().realize()` standalone gave ~18 ms even
sync-bracketed — but that is itself **confounded** (eager Python re-scheduling + `synchronize` overhead ≈ a whole
decode step), so it does NOT isolate the upload and is **not relied upon**. The physical 4-byte / 0-GB/s / idx-0
evidence is the definitive proof; the real warm decode is ~18.4 ms/token total (54 tok/s) — there is no
6.5 ms/token copy in it.

## 4. What changed
No model change (Bucket C → stop, per the gates). One risk-free **harness correctness fix**: the census
(`extra/qk_decode_primitive_census.py`) now (a) captures the full kernel name (regex up to `arg`, not just the
first token) and (b) classifies the 4-byte `<- PYTHON` upload as `input_upload_sync_EXCLUDED` and **excludes it
from the GPU-time denominator** — encoding the invariant "a 4-byte, 0-GB/s copy is a sync stall, not GPU work."

## 5. Measured impact (corrected per-class decode GPU, sync artifact excluded)
| class | % decode GPU (relative proxy) |
|---|---:|
| QK GEMV (ffn_down 24 + ffn_gate/up 18 + lm_head 18 + attn_q/o 13 + attn_k/v 2) | **~75%** |
| non-GEMV small ops (580 kernels) | **~25%** |

Warm decode unchanged (~54 tok/s, 18.4 ms/token, 780 programs) — the "copy" was never real cost, so there is
nothing to speed up here. The correction **sharpens** the gap picture: decode GPU is QK GEMVs (competitive,
~76% HBM) **+ the ~25% non-GEMV small-op tail**. No mysterious copy.

## 6. Next highest-value target
The "copy" is closed (artifact). The remaining real levers, unchanged from the gap plan:
1. **Flash-decode default for long context** (measured 1.73× @ ctx 4096, already built) — still #1.
2. **Decode small-op fusion** — the ~25% non-GEMV tail (RMSNorm+scale+add, gate·up·silu, RoPE+view), exactly
   what llama fuses (`norm.cu:147`, `mmvq.cu has_fusion`). This is now the clear short-context structural target
   — but it's the broad fusion arc, explicitly out of scope for this narrow diagnostic. Recommend: **move to a
   scoped small-op-fusion audit** (does tinygrad's scheduler already fuse some? which of the 580 are fusible?)
   before building anything.

The lesson banked: DEBUG=2 attributes step-boundary sync stalls to the first kernel's `tm`; a 4-byte/0-GB/s
"copy" at idx 0 is a sync artifact, not GPU work — always check bytes/bandwidth before treating a kernel's tm as
real cost.
