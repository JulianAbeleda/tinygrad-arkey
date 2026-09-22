# Kernel golf: the operational method

Date: 2026-08-30
Status: operational companion to `docs/what-makes-inference-fast.md`. That document owns the
theory and remains the canonical answer to "what makes inference fast." This document organizes
the measured record into a decision structure: three axes, a tiered lever ledger, one triage
rule, and one per-GPU column format. New performance theory still belongs in the principles doc
first. New measurements still belong in dated evidence docs. This document only indexes both, so
that the next unit of work is derivable rather than argued.

### Scope

Dense autoregressive transformers only. This inherits the principles doc's token-ledger scope
restriction verbatim: every token traverses every layer, so every layer's weights and boundaries
enter the accounting. Do not transfer this method to mixture-of-experts, recurrent, speculative,
or other conditional-compute architectures without first rebuilding their route ledger.

---

## 0. The three axes

The optimization space factors into three axes. Each axis is a different kind of split. Keeping
the kinds distinct is what makes the structure usable.

1. **Regime: prefill vs decode.** Partitions the problem by binding resource. Decode (`M=1`)
   sits below the crossover `M* = (w/16)·(R/BW)`: bytes bind. Prefill (`M=512`) sits above:
   the multiply unit's achievable rate binds. The boundary itself is a per-target number because
   `R` and `BW` are per-target measurements (principles doc §2).
2. **Lifecycle: kernel vs token.** Partitions the work by success claim. The kernel lifecycle
   ends at "this kernel is faster in isolation and its output contract is declared." The token
   lifecycle ends at "the production token wall improved and the ledger explains why." The two
   can disagree: split-KV combine was a real kernel win and a rejected token change
   (`COMBINE_TAX_DOMINATES`, `docs/prefill-lessons-ledger.md`).
3. **GPU: the target.** Does not partition anything — it **parameterizes**. Levers are stated
   device-invariant; their applicability and magnitude are device-conditional measured facts.
   Each target contributes a column: measured facts, per-lever flags, and a live par sheet (§5).

Regime × lifecycle gives four quadrants (§1). Magnitude tiers rank the levers inside a quadrant
(§2). The triage rule picks the quadrant and tier from the gap-to-ceiling (§3). The GPU column
says which levers are legal on this die and what the current gap is (§5).

---

## 1. The 2×2 invariant core

The regime split cuts through both lifecycles, not just the token path. The same semantic
operation (a Q4_K projection) is a different kernel problem at `M=1` vs `M=512`, and the same
weights want opposite representations per phase: stay-packed fused GEMV for decode; materialize
or fuse-into-the-matrix-unit-feed for prefill (principles doc §3/§4).

| | decode (bytes bind) | prefill (rate binds) |
| --- | --- | --- |
| **kernel lifecycle** | Score: achieved GB/s vs the target's sustainable memory rate. Levers: stay packed, coalescing, lane mapping, occupancy / memory-level parallelism. Evidence: `q4k_g3_lanemap_gemv_*` routes; NV decode at 764.8 GB/s = 41% of its 1792 GB/s bound while llama reaches ~62% of the same bound (`docs/nv-prefill-decode-diagnosis-20260801.md` §2). | Score: achieved TFLOPS vs the target's **measured** matrix-unit peak `R`. Levers: **which-unit (tier 1)**, fused dequant into the matrix-unit operands, tile geometry, operand staging. Evidence: 5.5× WMMA switch, 3.4× tile-geometry search, 2.25× fused Q4_K dequant→WMMA (§2, tier 1). |
| **token lifecycle** | Score: token wall vs `B_route/BW` plus the measured non-overlapped boundary term. Levers: delete materializations and launches, split-KV economics, device-native selection/sampling, serial tail. Evidence: +13.3–18.7% buffer-identity KV read; 948 kernels/token at depth 512 with the GEMV class at 3.774 ms of 5.923 ms total (`docs/five-lever-test-20260803-l3-gemv-census.json`). | Score: chunk wall vs `F/R` with config-derived FLOPs (never `2·P·T`). Levers: strategy selection by memory arithmetic (`FULL_RESIDENT_OVERLAY` vs `BOUNDED_PACKED_TILES` decided by VRAM fit, not preference), graph capture, composition, the promotion bar. Evidence: the strategy ladder's preconditions (`docs/8b-vs-14b-prefill-regression-20260721.md`). |

---

## 2. The needle-mover ledger: levers tiered by measured magnitude

Every entry cites the evidence doc. Shipped means promoted to production under a clean bracket.
The tiers are not equal-weight menu options: a lower tier is noise while a higher tier is open.

### Tier 1 — order of magnitude. One discrete choice: which execution unit does the multiply.

| evidence | delta | status | source |
| --- | ---: | --- | --- |
| AMD 14B prefill, vector-ALU → WMMA (`BOUNDED_PACKED_TILES`) | ~5.5× (354 → 1948 tok/s) | shipped | `docs/8b-vs-14b-prefill-regression-20260721.md` |
| NV sm_120 prefill, scalarized GEMMs vs llama's q8_1+dp4a MMQ on the same box | ~130× open gap (101–115 vs 14,250 tok/s) | open | `docs/nv-prefill-decode-diagnosis-20260801.md` |
| Metal precontract prefill, tile geometry searched within the unit | 3.4× (1061 → 3610 GFLOPS) | correct, not promoted | `docs/qwen3-8b-prefill-metal-precontract-campaign-20260731.md` |
| Fused Q4_K dequant→WMMA, stay packed, unpack in the feed | 2.25× (359 → 808 tok/s, 14B pp512) | shipped | `docs/prefill-lessons-ledger.md` (Quant/int8) |

### Tier 2 — tens of percent. Bytes and structure.

| evidence | delta | status | source |
| --- | ---: | --- | --- |
| Buffer-identity KV read: delete the full-MAXC slice materialization | +13.3–18.7% by context, byte-identical | shipped, default-on | `docs/prefill-lessons-ledger.md` (Decode) |
| DBUF operand staging structure: overlap built into the loop body | ~1.5× kernel-level (7.7 vs 11.5 TFLOPS) | partial | `docs/prefill-lessons-ledger.md` (DBUF) |

### Tier 3 — single digits, and mostly refutations.

| evidence | outcome | source |
| --- | --- | --- |
| Split-KV combine B4 | +5.6–5.85% at ctx4096, below the +7% promotion bar; `COMBINE_TAX_DOMINATES`; attention is ~17% of decode | `docs/prefill-lessons-ledger.md` (Decode) |
| Waitcnt tuning before stage ownership | flat — cannot create overlap the structure did not build | `docs/prefill-lessons-ledger.md` (K-major) |
| Safe DS-offset immediate folding | measured **loss** — keep materialized VALU offsets | `docs/prefill-lessons-ledger.md` (DBUF) |

### The refutation ledger is load-bearing

A refuted lever on one target may be live on another. Check flags (§5) before starting work.

| claim | verdict | scope | source |
| --- | --- | --- | --- |
| "int8 gives 2× over fp16" | refuted: iu8 WMMA = fp16 rate | RDNA3 only; real on RDNA4/CDNA | `docs/prefill-lessons-ledger.md` (Quant/int8) |
| "which-unit is worth 10–20×" | refuted: FMA peak ≈ simdgroup peak (0.933×), one shared unit | M4 only; holds on gfx1100 (~5× measured) and sm_120 | `docs/what-makes-inference-fast.md` §10 |
| NACC scaling hides matrix-op latency | inverted on M4: throughput falls past nacc=2 | per-target sweep required | `docs/what-makes-inference-fast.md` §10 |
| "hand 4413 tok/s beats generated" | invalid baseline: leaked LDS geometry ran 1/16 of the output | corrected: generated 3561.32 > hand 2095.70 | `docs/prefill-lessons-ledger.md`, `docs/prefill-current-state.md` |

---

## 3. The triage rule

The gap-to-ceiling tells you which tier you are in. This is the rule the corpus has been
implicitly following; stated once:

1. **Compute the ceiling from measured target facts.** BoltBeam already does this per phase
   (NV: decode 383.6 tok/s bandwidth-bound, prefill 13,664 tok/s compute-bound,
   `docs/nv-prefill-decode-diagnosis-20260801.md` §1). Use measured achievable `R` and `BW`
   as denominators wherever they exist; a spec-sheet denominator is a flagged gap, not a fact.
2. **Measure % of ceiling per phase.**
   - **~1–20%:** a units/route problem. Only tier 1 matters. Everything else is noise.
   - **~20–60%:** tier 2. Bytes, fusion, tile geometry, achieved rate.
   - **above ~60–80%:** tier 3. Boundaries, serial tails, overlap. Every candidate must clear
     the promotion bar, because composition eats single-digit wins.
3. **Rank rows by measured share; apply the highest applicable tier to the biggest row.**
   Ranking comes from a fresh trace, never from a stale one (promotion moves the bottleneck).
   Example: NV prefill trace — ffn_down Q6_K 33.2% + ffn_gate_up Q4_K 31.5%, fused attention
   0.4%. Nothing below the top rows is worth touching.
4. **Check the refutation ledger and the target's lever flags first.** The lever must exist
   and must not already be refuted on this die.

The percent boundaries are working thresholds induced from the record, not laws. When a case
sits on a boundary, the tie-break is the promotion bar: work the tier whose expected delta
clears it.

---

## 4. The join: contract and promotion

The two lifecycles connect through exactly one gate, and the corpus's most expensive mistakes
all happened at this gate.

- A kernel candidate crosses from kernel lifecycle to token lifecycle only with a declared
  output contract: dtype, layout, ownership, destination. An undeclared contract recreates the
  deleted boundary as a hidden copy (principles doc §0.4.4).
- A local win is provisional until the complete production token wall improves under a clean
  repeated bracket, with the ledger explaining where the time went (principles doc §0.1
  promote stage). The decode promotion bar in force is +7%.
- After any promotion, re-capture the entire ledger. Topology and ranking change; yesterday's
  next target may no longer be today's next target (principles doc §0.5.7).

---

## 5. Per-GPU columns

The invariant core (§1–§4) is written once. Each GPU is a column with three parts: measured
facts, lever flags, live par sheet. Three columns exist today. Populating a new column IS
`docs/bringing-up-a-new-target-20260731.md`: its Phase 0 fills the facts, its Phase 2 sets the
first lever flag, its Phases 3–6 walk down the tiers. The bring-up method and this structure
are the same object viewed from two ends.

### gfx1100 (AMD RX 7900 XTX)

- Facts: `R` = 105.5 TF **measured** (`extra/llm_research/microbench/wmma_peak.cpp`; 86% of the
  122.8 spec figure — use 105 TF as every efficiency denominator). `BW` ~960 GB/s is HBM peak,
  not a measured sustainable rate — flagged gap.
- Flags: which-unit **shipped** (~5.5×). Fused dequant→WMMA **shipped** (2.25×). int8-2×
  **refuted**. 4x4 WMMA **parked** (VGPR exhaustion; emergent generated-stream fault).
- Par sheet (2026-07-24, `docs/prefill-current-state.md`): 8B pp512 +11.4% over same-session
  llama (soft — llama's noisiest point), pp4096 +3.3%; 14B pp512 +5.6%, pp4096 +8.8%.
  Tier-3 territory: single-digit margins, promotion-bar discipline applies to everything.

### M4 (Apple, 10-core, Metal)

- Facts: `R` = 3.78 TF **measured** (`extra/llm_research/microbench/wmma_peak_metal.py`).
  Plain FMA fp16→fp32 = 0.933× of `R` — one shared unit. `BW` **unmeasured**; `M*` is a
  function `M*(BW) = 1063 / BW_GBps`, low single/double digits across the plausible range, so
  the regime classification is robust even though the crossover point is not pinned.
- Flags: which-unit **does not exist** (tier 1 vacated — the headroom is routing/tiling, not
  units). NACC **inverted** past 2. TC-candidate compilation through the provider **blocked**
  (5/5 `provider_compile:provider_failure`) — the load-bearing blocker.
- Par sheet (2026-07-31): decode 17.24 tok/s = 84.8% of llama; prefill 54.2 tok/s = 24.5% of
  llama on `DIRECT_PACKED_FALLBACK`. Precontract kernel is correct at 3610 GFLOPS vs 1063
  control but **not promoted** (QUALIFY and POLICY blocked) — a tier-1-magnitude gain parked
  at the §4 join, which is the design's point: the kernel lifecycle finished, the token
  lifecycle has not accepted.

### sm_120 (NVIDIA RTX 5090)

- Facts: 1792 GB/s and 180 TFLOPS are the BoltBeam hardware inputs — **neither is a measured
  achievable rate yet**. No `wmma_peak`-equivalent microbench has run on this die. Per §3.1
  this is the column's first gap: measure achievable `R` and sustainable `BW` before quoting
  any efficiency percentage as authoritative.
- Flags: which-unit **exists and is open** — llama reaches the int-dot/tensor path (q8_1
  quantize + dp4a MMQ, zero `mma` kernels, traced) at 14,250 tok/s pp512 on this exact box;
  our GEMMs are scalarized. Everything below tier 1 is untested.
- Par sheet (2026-08-01, `docs/nv-prefill-decode-diagnosis-20260801.md`): prefill 101–115
  tok/s = ~1% of the 13,664 ceiling; decode 158.2 tok/s = 41% of the 383.6 ceiling (llama:
  237.1 = 62% of the same bound). Trace ranking: ffn_down Q6_K 33.2%, ffn_gate_up Q4_K 31.5%,
  ffn_down Q4_K 14.6%, attn_qo 13.4%; fused attention 0.4% (healthy, not a target).

---

## 6. Reading the current state through the structure

The triage rule reads sm_120 unambiguously:

- **Prefill at ~1% of ceiling → kernel × prefill quadrant, tier 1, top trace rows.** The entire
  game is one move: get the Q4_K/Q6_K GEMM routes onto the dp4a/tensor unit, starting with
  ffn_down Q6_K and ffn_gate_up Q4_K (64.7% of traced time between them). llama proves the
  ceiling is real on this box. No tier-2 or tier-3 work on NV prefill is justified until this
  lever lands.
- **Decode at 41% of ceiling → kernel × decode quadrant, tier 2.** A ~1.5× achieved-bandwidth
  problem against a comparator at 62% of the same hard floor. Not a units problem.
- **M4 → token lifecycle work, not kernel work.** The fast kernel exists; the join (§4) is
  blocked. Unblocking QUALIFY/POLICY is worth ~3.4× and requires no new kernel.
- **gfx1100 → tier 3 discipline.** Margins are single-digit; every candidate faces the
  promotion bar; ledger re-capture after each promotion is mandatory.

## 7. Maintenance rules

1. A new measured delta enters §2 in its tier, with its evidence doc. A new refutation enters
   the refutation ledger with its scope (which die).
2. A promotion invalidates the affected par sheet; re-capture before ranking the next target.
3. A new GPU adds a column via the bring-up method; it never edits the invariant core.
4. If a campaign finds a win that no §0.4 lever of the principles doc explains, that is new
   theory: it goes to `docs/what-makes-inference-fast.md` first, then this index cites it.
