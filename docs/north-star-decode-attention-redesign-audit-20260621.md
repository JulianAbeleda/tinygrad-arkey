# North-Star Decode Attention Candidate Redesign Audit

Date: 2026-06-21

Scope: use the measured failure of the first executable north-star `flash_attn_tile` candidate
(`docs/north-star-flash-attn-tile-execution-result-20260621.md`) to scope the next candidate — **audit first, build
only if a clearly different bounded candidate targets the *measured* ceiling.**

## Decision: **`REDESIGN_AUDIT_POINTS_TO_CODEGEN_DATAFLOW`** — NO BUILD

The bounded standalone-tile lever is **exhausted**: every hand-rolled tile (warp-cooperative, fused-LDS, vector)
replaces `gqa_coop_vec`'s **optimized matmul q·k** with a slower hand-rolled dot and loses. The combine is **not**
the ceiling (refuting the prior diagnosis). The remaining 10× gap to llama is **hand-tuned-vector-kernel codegen
quality**, not a dataflow/combine restructure tinygrad can express better. No bounded build is justified.

## The measurement gap (Phase 1 gate) — RESOLVED, and it corrects the prior diagnosis

The prior result classified `FAIL_LOCAL_AB` as *"HBM-bandwidth-bound combine on materialized split-partials."*
**Traffic accounting + a throughput probe refute that:**

**Traffic accounting (per the failed candidate, S=64):**
| quantity | bytes | time @ 960 GB/s peak |
|---|---:|---:|
| `pout` written by partial (Hq·S·Hd·4) | 1.0 MB | ~1.0 µs |
| `pout` reread by combine | 1.0 MB | ~1.0 µs |
| K+V read by partial @ctx1024 (fp16) | 4.2 MB | ~4.4 µs |
| total candidate HBM traffic @ctx1024 | ~6 MB | **~6.5 µs** |

But the **measured** candidate throughput @ctx1024 is **162 µs** — ~25× off peak bandwidth. So the candidate is
**not** bandwidth-bound anywhere, and the combine's reread (~1 µs) is negligible. The latency-measured "combine
cost" (~75 µs, S-invariant, *decreasing* with ctx) was **2nd-raw-dispatch / inter-kernel latency**, not traffic.

**Throughput probe** (`extra/qk_north_star_dispatch_probe.py`, back-to-back calls, one final sync — the fair
in-model comparison; vs the original per-iteration-sync latency):

| ctx | coop latency | cand latency | latency speedup | coop **throughput** | cand **throughput** | **throughput speedup** |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 95.3 | 178.8 | 0.53× | 75.5 | 163.7 | **0.46×** |
| 1024 | 104.8 | 176.6 | 0.59× | 85.0 | 162.4 | **0.52×** |
| 4096 | 173.7 | 245.8 | 0.71× | 143.6 | 164.8 | **0.87×** |

**Findings:** (1) the candidate is genuinely slower under the fair throughput metric (0.46–0.87×), so the
`FAIL_LOCAL_AB` verdict **stands**; (2) but the candidate throughput is **flat ~163 µs** while coop **scales**
(75→85→144 µs) — so the ceiling is the **cooperative-dot q·k partial** (latency/occupancy-bound: 512 small
workgroups × LDS-load + barrier + a 32-step ds_bpermute butterfly per key, work hidden under launch latency at
ctx≤1024), **not the combine**; (3) coop's matmul q·k scales efficiently from 75 µs — it is near-optimal for
tinygrad primitives. Byte-exact (err 0.0) throughout.

## Phase 0 — evidence table

| | q·k mapping | KV split | GQA pack | partial state materialized | partial bytes (S=64) | combine | kernels | workgroups by ctx | win mechanism | risk |
|---|---|---|---|---|---|---|---|---|---|---|
| **gqa_coop_vec** (winner) | **optimized matmul** → scores buf | S=ceil(ctx/128), 4–32 | V-reuse, d→LOCAL | scores[Hq,ctx] + pout[Hq,S,Hd+1] | scores 128KB@1024 | flash_gmax/den/combine (graph) | 6, **batched JIT graph** | 32–256 | efficient matmul q·k + graph dispatch; **75–144 µs throughput** | none (the bar) |
| **failed north_star tile** | **ds_bpermute cooperative dot** | S=16–96, 128–768 wg | 4 warps = 4 heads | pout[Hq,S,Hd] + pm/pl | 1.0 MB | flash_reduce (serial) / streamk (128 wg) | 2, **raw dispatches** | 128–768 (grows) | (intended: many splits) — **lost: partial floor ~163 µs** | partial latency/occupancy-bound |
| **llama flash_attn_tile** (oracle) | vector FMA, **LDS K/V staged once** | parallel_blocks 48–144 (grows) | ncols2 column-pack | small register `(m,lse,acc)` per block | (in-register) | parallel merge + streamk fixup | in-kernel | 8×48 … 8×144 | **hand-tuned vector kernel = 9.2 µs/layer (10× coop)** | CUDA/HIP, not tinygrad-native |
| **proposed next** | — | — | — | — | — | — | — | — | **none bounded** (see ranking) | — |

## Phase 2 — candidate family ranking (why no bounded build)

| family | expected value | complexity | first gate | differs from refuted? | bounded build? |
|---|---|---|---|---|---|
| **A. Compact-state combine** (reduce only `(m,lse,acc)`) | **~0** — the combine is already negligible (~1 µs traffic); the ceiling is the partial. `pout` *is* the `acc` state; there is no smaller state. | low | ≥1.05×@1024 | no (combine, refuted as non-ceiling) | **no win** |
| **B. Coop-qk-preserving** (keep coop's matmul q·k, change combine/V) | **insight, not a build** — coop's matmul q·k + graph combine *is* the fast path; preserving it AND its combine = coop (no delta). Confirms the right q·k is the matmul, but yields no candidate to build. | n/a | n/a | n/a | **no delta** |
| **C. In-kernel split combine** (persistent / two-level, avoid writing partials) | low — the combine isn't the ceiling; avoiding the ~1 µs `pout` write cannot beat coop when the partial floor (~163 µs) already exceeds coop@1024. Risks collapsing workgroups (closed lane). | high | ≥1.05×@1024 | partially (still a cooperative-dot partial) | **fails @1024** |
| **D. llama source port / bridge** | high *as a reference oracle* (gives 9.2 µs) but it is a CUDA/HIP port of `fattn-tile.cuh` + ggml infra — large, and it is *importing*, not a tinygrad primitive (violates the "tinygrad machine search" principle). | very high | n/a (port) | yes | **not bounded; reference only** |
| **E. Deeper codegen / dataflow** | **the real lever** — even coop (the best tinygrad attention) is 10× slower than llama's hand-tuned vector kernel; closing that is *llama-style local primitive quality* via codegen (instruction scheduling, register/LDS use), not a bounded kernel tweak. | very high / unbounded | n/a | yes | **not bounded** |

## Phase 3 — build / no-build

**No build.** A/B/C cannot pass the @ctx1024 gate (the partial is the ceiling and coop's matmul q·k is already
near-optimal). D is a large reference-oracle port, not a tinygrad primitive. E is the real but unbounded lever.
None satisfies "differs materially + clear first gate + bounded + goes through the binding system" for a quick
build. The honest next lever is **codegen / primitive quality**, scoped separately if at all.

## Phase 4 — build: SKIPPED (no-build decision).

## Audit answers (the 6 questions)

1. **Where is the partial-traffic created?** `warp_flash_tile` writes `pout[Hq·S·Hd]` f32 + `pm/pl[Hq·S]`; the
   combine (`flash_reduce`/`streamk`) rereads `pout`. At S=64 that is **1.0 MB** — **~1 µs at HBM peak, negligible.**
2. **Can partial traffic be reduced without collapsing workgroups?** It *can* (compact two-level / in-kernel
   combine), but it **does not matter** — the traffic is already ~1 µs; the ceiling is the partial, not the combine.
3. **Can we keep coop's q·k and change only later stages?** coop's q·k (matmul) + its graph combine *is* already
   the fast path; "preserving it" yields no candidate distinct from coop. The lesson: the **matmul q·k is the right
   primitive**; hand-rolled cooperative dots are slower.
4. **What does llama materialize between split workers and combine?** A small per-block **register `(m, lse, acc)`**
   state with LDS K/V staged once; it merges 48–144 blocks with a parallel merge + streamk fixup — **9.2 µs/layer**.
   tinygrad's coop materializes scores + `pout` and combines in a graph — **~85–144 µs**, a codegen-quality gap, not
   a state-size gap.
5. **Bounded kernel change, dataflow/codegen capability, or impossible?** **Codegen/primitive-quality capability** —
   the 10× gap is hand-tuned-vector-kernel quality; the bounded standalone-tile space is exhausted.
6. **Smallest next executable candidate meaningfully different from the failed one?** **None that clears the gate.**
   Combine-side changes (A/C) don't move @1024; a faster-than-coop q·k partial is not a bounded change (coop's
   matmul is already near-optimal). The next real step is codegen, not a kernel.

## Acceptance gates

| gate | result |
|---|---|
| G1 partial traffic accounted | PASS (~1 MB / ~1 µs — negligible) |
| G2 combine behavior explained | PASS (latency "cost" = 2nd-dispatch; throughput shows partial is the ceiling) |
| G3 ≥4 candidate families ranked | PASS (A–E) |
| G4 next candidate differs in a named dimension | n/a — no build (named: q·k mapping is the dimension, not bounded) |
| G5 no closed lane reopened | PASS |
| G6 no model/default route changes | PASS (no build) |
| G7 build goes through decode_eval/lifecycle | n/a (no build) |
| G8 policy guard passes | PASS |
| G9 tree clean after commit | PASS |

## Corrections to prior canonical claims

- **Refuted:** "the combine is HBM-bandwidth-bound on materialized split-partials" (prior execution result +
  handoff). Traffic is ~1 MB (~1 µs); the combine is negligible. The latency-measured combine "cost" was
  2nd-raw-dispatch overhead. Corrected here + in the refutation ledger + handoff.
- **Confirmed + refined:** `FAIL_LOCAL_AB` stands (throughput 0.46–0.87×); the ceiling is the cooperative-dot q·k
  partial (latency/occupancy-bound), and coop's matmul q·k is near-optimal for tinygrad primitives.

## Next action

Scope **codegen / primitive-quality** (or a llama-port reference oracle) as a *separate, large* project — or
**rest the north-star** here. Do **not** build another bounded standalone tile or combine variant: it will repeat
the failure. Comparator stays `gqa_coop_vec`.

## Changed files

`extra/qk_north_star_dispatch_probe.py` (new, the resolving measurement), this doc, refutation ledger update,
handoff/READMEs corrections.

## Boundary

Audit only — no kernel built, no model route/default, no closed lane reopened. Clock-pinned diagnostic probe,
perf-state restored to `auto`.
