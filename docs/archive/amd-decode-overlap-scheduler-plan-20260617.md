# Decode-overlap scheduler — design & honest feasibility (Phase 6, 2026-06-17)

The multi-ring primitive is proven (Phases 0–5): two AMD compute rings overlap independent work, cross-ring
deps order correctly, and a minimal DAG scheduler overlaps a dependent chain with independent work. This doc
scopes whether and how that helps **decode**, before any model code (Phase 7). **Bottom line up front: the
single-stream decode latency win is bounded and uncertain (host-bound + bandwidth-saturated); the primitive's
clearest payoff is concurrent-stream throughput. Phase 7 should only proceed against a controlled e2e gate.**

## Decode dependency structure (where is the independent work?)

Batch-1, T=1 decode is a tight serial chain:
`embed → for each layer { attn: qkv_proj → rope → flash_decode → out_proj ; ffn: gate&up → silu·mul → down } → norm → lm_head`.
The only genuinely **independent** work within the chain:
- **QKV projections** — `q=Wq·x`, `k=Wk·x`, `v=Wv·x` share input `x`, no mutual dep (3-way independent).
- **FFN gate & up** — `gate=Wg·x`, `up=Wu·x` share input (2-way independent).
- Across layers: layer N+1 depends fully on layer N (no overlap without speculative weight prefetch).

So a scheduler *could* put `{k,v}` on ring 1 while `q` runs on ring 0, and `up` on ring 1 while `gate` runs on
ring 0. The DAG-runner concept from Phase 5 expresses this directly (tasks + ring + deps).

## The catch: these independent ops are all HBM-bandwidth-bound

Decode projections are GEMVs (matrix·vector) — **memory-bound**, not compute-bound. Two bandwidth-bound kernels
on two rings share the ONE HBM bus, so they overlap only to the extent each leaves bandwidth headroom.
Characterization (`extra/amd_ring_overlap_characterize.py`, control-relative one-ring/two-ring metric):

| pairing | overlap (one_ring/two_ring) | reading |
|---|---|---|
| bandwidth ‖ bandwidth (A‖A) | ~2.0× | two *under-saturated* bandwidth kernels co-run well |
| bandwidth ‖ compute (A‖B) | ~1.08× | the small compute kernel runs **fully hidden** under the memory shadow (span≈max) |
| compute ‖ compute (B‖B) | ~1.34× | partial (and note: Phase 3 measured 2.0× for the same config — see fragility) |

**Crucial caveat:** the A‖A 2.0× is for a *heavily under-saturated* kernel (16 workgroups). A real decode GEMV
already runs at ~76% of HBM peak (banked: `amd-decode-kernel-beats-llamacpp`), so it has **little bandwidth
headroom** — two such GEMVs on two rings can't exceed 100% of the bus, i.e. a **~1.3× ceiling**, not 2×. The
2× only appears when a single op leaves the GPU mostly idle.

## Measurement fragility (a hard constraint on the gate)

The same compute‖compute pairing measured 2.0× in Phase 3 (with a clean one-ring control) and 1.34× here, with
no config difference that should matter. Micro-overlap factors are **not stable** across harnesses (clock ramp,
thermal, allocation state — the `amd-decode-measurement-confounds` lessons). **Therefore the decode gate must be
a controlled, repeated, end-to-end tok/s measurement (two-ring build vs one-ring build, same code, back-to-back,
warm), NOT micro-overlap factors.**

## The host-overhead ceiling (Amdahl)

Decode wall-time is **~55% host overhead** (CPU dispatch / Python), ~45% GPU (banked:
`amd-decode-real-bottleneck`). Multi-ring overlap only touches the **GPU** portion. Even if every overlappable
GPU op perfectly overlapped (optimistic), the wall ceiling is roughly:
`1 / (0.55 + 0.45·(1 - overlappable_fraction·(1 - 1/overlap)))`. With only the projections overlappable (a
fraction of the 45% GPU) at a ~1.3× bandwidth-limited factor, the realistic single-stream wall gain is **~1.05–
1.15×** — likely below a "material" gate, and at risk of being eaten by the extra scheduling/dispatch host cost
(which *adds* to the 55% that already dominates).

## Memory hazards

- QKV/gate-up write **disjoint** output buffers and read a shared **read-only** input — no hazard; safe to split.
- Cross-ring ordering for the *consumer* (e.g. `out_proj` needs q,k,v; `down` needs gate·up) must `wait` on
  both producers' signals (Phase-4 semantics, proven). The join is the natural barrier.
- The KV-cache store (flash_decode) writes the cache; must not be reordered against the next step's read —
  keep it on the critical-path ring with its existing ordering.

## Queue assignment & fallback

- Assignment: critical-path op on ring 0; the largest *independent* sibling on ring 1 (e.g. `k,v` ‖ `q`;
  `up` ‖ `gate`). The Phase-5 DAG runner generalizes this.
- **Gating:** `DECODE_OVERLAP=1` AND `AMD_COMPUTE_RINGS>=2`, default **off**. KFD-local + non-AQL only (the
  ring guard already enforces this). Any unsupported path → fall back to the single-ring chain.
- **The HCQ serialization trap (banked):** the default `HCQProgram.__call__` starts each kernel with
  `.wait(timeline_signal, timeline_value-1)` → serializes through the global timeline. A real overlap scheduler
  must bypass that per-op global wait and use explicit per-producer signals (as all the probes do). This is the
  single biggest implementation risk — it means decode overlap cannot be a thin wrapper over the normal dispatch
  path; it needs its own submission path for the overlapped region.

## The exact benchmark gate for Phase 7

Proceed to wire decode overlap **only if** a controlled experiment shows:
1. **Same output** — greedy token stream byte-identical to the single-ring path (overlap must not change values).
2. **Material e2e speedup** — decode tok/s (warm, many reps, median) with `DECODE_OVERLAP=1` beats the
   `=0` build by **≥1.15×** single-stream, measured both wall-clock AND `time_sum_s`@DEBUG=2, on Qwen3-8B.
3. **No regression** with the flag off (byte-identical, same tok/s).
4. **No fault/hang**; VRAM unchanged.
If single-stream can't clear ≥1.15×, **bank the primitive for throughput** and do not wire single-stream decode.

## Recommendation

The primitive is real and valuable, but the **single-stream decode latency** case is Amdahl-bounded (host 55% +
bandwidth-saturated GEMVs with ~1.3× headroom) and the win likely lands in the noise/scheduling-overhead band.
The **honest highest-value use is concurrent-stream throughput** — two independent decode streams (or a batched
server) genuinely overlap on two rings (the cross-process +32% premise, now reproducible in-process). 

So Phase 7 has two framings; pick deliberately:
- **(7a) Throughput** (recommended): run two decode streams on two rings, measure aggregate tok/s vs one ring.
  Highest expected payoff, cleanest measurement, no critical-path surgery.
- **(7b) Single-stream latency** (speculative): split QKV / gate-up across rings via a dedicated overlapped
  submission path, gate ≥1.15×. High implementation risk (the HCQ serialization trap), low expected payoff.
  Only worth it if (7a) isn't the goal and a controlled micro-benchmark of *real* projection shapes first shows
  headroom.

STOP after this doc (Phase 6 requested as the scope). Phase 7 is a separate, deliberate go/no-go.

Anchors: `amd-multiring-compute-primitive-20260617.md` (P0 audit), `bench/amd-two-compute-ring-probe/` (P3),
`bench/amd-two-ring-dependency-probe/` (P4), `bench/amd-two-ring-dag-probe/` (P5),
`bench/amd-ring-overlap-characterize/` (this phase's input). Memory: `amd-multiring-compute`,
`amd-decode-real-bottleneck`, `amd-decode-measurement-confounds`.
