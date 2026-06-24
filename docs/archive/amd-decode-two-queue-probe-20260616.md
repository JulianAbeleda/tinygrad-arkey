# Two-queue overlap micro-prototype (Milestone 0) — 2026-06-16

Gate for the decode-overlap build: can tinygrad issue two compute kernels to two AMD compute
queues and get *real* concurrent execution intra-process (`wall_concurrent < t_A + t_B`)?
Harness: `extra/qk_two_queue_probe.py` (standalone, raw-HIP kernels, on-device GPU timestamps,
confound-controlled). RX 7900 XTX gfx1100.

## Result — uniform serialization, invariant to kernel shape

Bandwidth kernel A (grid 512 = 669 GB/s = **77.9% peak**) vs a tiny compute kernel B (grid 16,
`t_B ≈ 0.5 t_A`, easily fits idle CUs), median of 40 reps, warm:

| pairing | concurrent wall | serial sum | overlap factor |
|---|---:|---:|---:|
| A ‖ B | 1788 µs | 1802 µs | **1.01×** |
| A ‖ A | 2382 µs | 2406 µs | 1.01× |
| B ‖ B (tiny — *could* overlap) | 1194 µs | 1197 µs | 1.00× |

Even a 16-workgroup B that would trivially co-reside with A on the 96 CUs does **not** overlap.
The factor is ~1.0 regardless of kernel shape.

## Root cause — one hardware compute ring (not a hardware limit)

`AMDDevice` creates a **single** compute ring (`tinygrad/runtime/ops_amd.py:1001`,
`self.compute_queue = self.create_queue(KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, …)`), and
`AMDComputeQueue._submit` **hardcodes** `dev.compute_queue` (`ops_amd.py:409-422,456-465`). Every
`dev.hw_compute_queue_t()` is just a command **builder** that funnels into that one ring, so all
compute executes in submission order → serializes. (Copy/SDMA uses *multiple* rings —
`self.sdma_queues` is a dict — which is why copy overlaps compute; compute itself is singular.)

The uniform ~1.0× (not noise) **validates** the harness: it faithfully measures execution
ordering. This is a software architecture limit, **not** a hardware one — the cross-process test
(`amd-decode-overlap-derisk-20260616.md`) already showed the GPU runs two *processes'* compute
concurrently (+32%), because each process gets its own KFD compute queue → its own ring/ACE pipe.

## Verdict: lever **alive**, build scope **escalated**

Overlap is reclaimable on this GPU (~+30%, hardware-proven), but it is **not** achievable by
"issue to a second queue object" — tinygrad has one compute ring. The build now requires a
**`[runtime]` change to the AMD backend** *before* the scheduler:

1. **M1a `[runtime]` — second hardware compute ring.** Allocate a 2nd compute AQL queue in
   `AMDDevice` (reuse `create_queue`, already general) and let a `HWQueue`/`_submit` target a
   *selectable* ring instead of the hardcoded `dev.compute_queue`. KFD supports multiple compute
   queues/process (cross-process proof). **Re-run `qk_two_queue_probe.py` — A‖B should jump > 1.2×
   once a 2nd ring exists.** This is the next killable gate.
2. **M1b `[runtime]`/`[codegen]` — cross-layer scheduler** (`runtime/graph/hcq.py` `ji_schedule`/
   `_resolve_deps`): issue layer-(N+1) weight-GEMV to ring 2 concurrent with layer-N non-GEMV,
   with cross-ring signal deps. Gated, exact (greedy token parity), AMD-validated.

This is materially deeper than the original "second-queue scheduler" estimate — it is AMD-backend
queue surgery + the scheduler — so the decision to invest is a real one:

| | |
|---|---|
| upside | ~+25–32% decode → ~69–72 tok/s, ~65–68% of llama |
| cost | a `[runtime]` 2nd-compute-ring change in `ops_amd.py` + per-ring submit routing + the cross-layer scheduler; higher risk than any lever shipped this arc |
| alternative | pivot to **B3** (read fewer bytes — sub-4-bit, orthogonal, no runtime surgery) or bank the arc at 58% |

The harness `extra/qk_two_queue_probe.py` is kept as the M1a gate: it will prove (or refute) real
overlap the moment a 2nd ring is wired.

Anchors: `amd-decode-overlap-derisk-20260616.md` (+32% cross-process floor / +38% ceiling),
`amd-decode-overlap-feasibility-spike-20260616.md`, `amd-decode-beyond-llama-roadmap.md` (B2).
