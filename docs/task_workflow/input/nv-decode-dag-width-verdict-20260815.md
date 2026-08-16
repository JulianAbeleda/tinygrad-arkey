# NV decode DAG width verdict: width-1 is falsified at HEAD (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `c74567a24`)
Status: **measured. Schedule-only: the decode linear is captured and analyzed,
no decode kernel executes. GPU lock held for model/weight allocation only.**

This closes the single open question in
`nv-240-audit-reconciled-20260815.md` section 6: is the decode DAG a serial
chain (width 1) or does it carry recoverable parallelism?

## 1. Method

`scratchpad/nv_decode_dag_width_probe.py` hooks `tinygrad.engine.jit.jit_lower`
and captures the compiled decode schedule at HEAD. It flattens every CUDA graph
batch back into an ordered call list, resolves each call to its memory-planned
`Buffer` objects, and reconstructs the runtime dependency DAG with the exact
`DepsTracker` model (`id(buf.base)` key plus byte ranges, RAW+WAR+WAW). A
second RAW-only pass separates true data flow from memory-plan aliasing. It
then feeds both into the runtime `plan_multi_stream` scheduler (4 streams).

## 2. Result

| metric | runtime RAW+WAR+WAW | RAW only |
| --- | ---: | ---: |
| decode calls | 668 | 668 |
| graph batches | 5 (32+64+128+256+188) | same |
| max ready width | 4 | 4 |
| critical path (kernel count) | 442 | 425 |
| chain ratio | 0.0045 | 0.0045 |
| roots | 3 | 3 |

`plan_multi_stream(4)` stream counts:

| deps | stream 0 | stream 1 | stream 2 | stream 3 |
| --- | ---: | ---: | ---: | ---: |
| runtime RAW+WAR+WAW | 410 | 139 | 48 | 71 |
| RAW only | 410 | 122 | 63 | 73 |

258 of 668 calls (39%) are placed off stream 0. Per-batch widths are 3-4 and
every batch distributes:

| batch calls | full width | raw width | stream counts (4 streams) |
| ---: | ---: | ---: | --- |
| 32 | 3 | 3 | 5 / 19 / 4 / 4 |
| 64 | 4 | 3 | 40 / 15 / 4 / 5 |
| 128 | 4 | 4 | 78 / 30 / 7 / 13 |
| 256 | 4 | 4 | 161 / 51 / 17 / 27 |
| 188 | 4 | 3 | 126 / 25 / 18 / 19 |

## 3. What the width actually is

The widest ready bundle is the attention head: the q GEMV
(`q4k_g3_lanemap_gemv_4096_4096`), the k GEMV
(`q4k_g3_lanemap_gemv_1024_4096`), the v GEMV
(`q6k_gen_partial_1024_4096_4`), and one support eltwise are all ready at the
same instant. So the parallelism is q/k/v concurrency, not a deep fan-out.

Gate/up are already fused into one kernel
(`q4k_g3_lanemap_gemv_w1w3fused16_12288_4096`), and the FFN down-projection
follows it serially. There is no gate/up concurrency to recover; that question
is moot at this graph.

RAW-only and RAW+WAR+WAW differ by only 17 kernels on the critical path, so
memory planning is not meaningfully serializing independent work. The width 4
is intrinsic data flow, not an over-fusion artifact.

## 4. Reconciliation with the Route B record

`nv-overlap-route-b-wall-outcome-20260815.md` reported "1020 of 1021 calls on
stream 0, width ~1". That record was measured at HEAD `319241408` on a
1021-kernel graph. At current HEAD the decode graph is 668 kernels and the same
scheduler distributes 39% of calls across 4 streams. The width-1 conclusion is
**stale**, not wrong-for-its-time: the post-`319241408` promotions (Q4 fp16
geometry, reduce-output/norm folding, JIT capture fix) changed the topology.

The audit's section 2 line "our DAG has width 1, nothing to distribute" and the
section 5 line "Route B: FLAT, because the ready-set scheduler finds no
independent branches in the width-1 DAG" are superseded by this record.

## 5. What this does and does not prove

Proved:

- The decode DAG is **not width 1**. There is real, schedulable q/k/v
  parallelism, and `plan_multi_stream` can exploit it.

Resolved by follow-up:

- The width-4 wall A/B at HEAD measures FLAT across 1..4 streams. q, k, and v
  are memory-bound GEMVs on the same HBM, so the scheduler distributes them but
  wall does not move. See `nv-overlap-route-b-head-wall-record-20260815.md`.
- NV native multi-queue (Route A) would distribute the same bandwidth-bound
  q/k/v GEMVs and is expected flat for the same reason; it is not a lever until
  an independent latency-bound support tail exists.

## 6. Next measured steps

1. Q6 GEMV core is the next measured kernel-work lever (~240 us): replicate the
   Q4 four-warp geometry fix on Q6 V and Q6 FFN-down.
2. The anchor-shadow path stays closed unless support kernels are first emitted
   as independent latency-bound work (already measured body-free FLAT), so it is
   not a near-term 240 path.

## Evidence

- `/tmp/nv_decode_dag_width.json` (full report)
- `/tmp/nv_decode_dag_width_core.pkl` (names, preds, costs, stream assignment)
- `/tmp/nv_dag_probe10.log` (raw probe output)
- `scratchpad/nv_decode_dag_width_probe.py`
