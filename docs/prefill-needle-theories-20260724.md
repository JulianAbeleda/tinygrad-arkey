# 8B prefill: where the needle actually is (measured 2026-07-24)

Derived entirely from hardware counters (`extra/llm_research/prefill/prefill_boltbeam_trace.py --hw-trace`, restored in
`654c9b2ce`) on 8B/gfx1100 under `TINYGRAD_PREFILL_PACKED_WMMA=0`, plus the whole-model authority
(`extra/llm_research/prefill/prefill_whole_synced.py --mode authority`). Numbers here are measurements, not estimates.

## The frame

| pp | tok/s | % of fp16 WMMA peak (122 TF) | % of HBM peak (960 GB/s) |
|------|-------|------|------|
| 512  | 3607  | 47%  | 3.7% |
| 1024 | 3512  | 46%  | 3.6% |
| 2048 | 3327  | 44%  | 3.4% |
| 4096 | 3003  | 39%  | 3.1% |

**HBM is irrelevant at ~3% of peak.** Every bandwidth/roofline framing is a dead end for prefill,
including BoltBeam's own `roofline --peak 960.0`. The workload is *issue-bound* at roughly half of
compute peak, so every remaining lever is instruction issue and LDS — never memory.

Attention specifically sits at **AI = 818 FLOP/byte, 6x past the ridge point (127)**, using 0.98% of
HBM, with L2 hit *rising* 62.8% -> 85.5% as context grows. A memory roofline cannot diagnose it.

## The split

**STATUS 2026-07-24 late.** All three theories tested; see `docs/prefill-current-state.md` for the
AUTHORITATIVE numbers. llama has since been re-measured same-session: its 8B **pp4096 = 3158**, so the
`llama 3160` figures used below are confirmed accurate; only llama's pp512 was drifted (3347, not 3571).
14B is also live again (1948/1787 tok/s) after `7463a6774`.

**READ THIS BEFORE ANY NUMBER BELOW.** Two arms exist and confusing them was an error in the first
version of this doc. The **SHIPPED** kernel is `PREFILL_V_TRANSPOSED=0`: **952 instructions per KV tile,
16 WMMA (1.68% useful), VMEM = 65.0% of SQ busy**. The 843-instruction / VMEM-36.7% figures belong to the
`PREFILL_V_TRANSPOSED=1` arm, which is **default OFF and net -3.4%** (`fd654024e`). Never quote the vT
numbers as the baseline.

Per-kernel, ctx512 (PMC-instrumented run; shares are what matter, not its absolute wall time):

| kernel class | share of kernel time | valu | mem | occ | l2 hit | lds conflict |
|---|---|---|---|---|---|---|
| `E_4_*` WMMA GEMMs (top 4) | **79.6%** | 1.6–1.9 | 9.9–11.6 | 37–42 | 33–54 | **12.5** |
| `amd_gfx1100_q16_grid_hd128_loop_attention` | 6.1% (pp512) -> ~23% (pp4096) | 5.3 | 23.7 | 36.4 | 63.1 | 5.1 |

(tinygrad names a WMMA matmul `E_*` because the tensor core consumes the reduce, so these ARE the GEMMs.
Normalized counter fields are IPC-like rates x100, NOT duty cycles — never read `valu_busy_pct` as a
utilization; see the raw passthrough.)

GEMMs are the **large** part and carry the 47%. Attention is the **inefficient** part at ~6% of peak —
8x worse than the model average — and it is the only term that grows with context, i.e. exactly llama's
advantage (llama's attention plateaus at 15.1% of budget at pp4096 vs our 23%).

## Prize sizing at pp4096 (attention = 23% of budget, ~6% of peak)

| if attention reaches | whole-model | vs llama 3160 |
|---|---|---|
| 2x (12% of peak) | +11.5% -> **3348** | ahead |
| 4x (25% of peak) | +17.5% -> **3528** | clearly ahead |
| GEMM parity (47%) | +20.1% -> 3606 | ceiling |

## THEORY 1 — work per wave — **DEAD (measured 2026-07-24, do not implement)**

The claim was: the loop body is mostly overhead, so put more WMMA work behind it by giving a wave 32-64
q-rows instead of 16, funding the extra accumulators by slicing the output dim (`acc_blocks`). **Refuted
on four independent grounds. Scope + full model: `scratchpad/theory1-work-per-wave-scope.md`,
`scratchpad/t1_cost_model.py`, resource table `scratchpad/t1_resource_table.py`.**

**1. Slicing frees far less allocation than it frees state.** Compiled, gfx1100, zero spills/scratch
throughout (metadata key is **`vgpr`**, not `vgprs`):

| acc_blocks | VGPR | KV body instrs | WMMA | instr/WMMA | global_ld/tile |
|---:|---:|---:|---:|---:|---:|
| 8 (shipped) | **254** | 952 | 16 | 59.5 | 144 |
| 4 | **240** | 837 | 12 | 69.8 | 80 |
| 2 | **212** | 785 | 10 | 78.5 | 48 |
| 1 | **171** | 753 | 9 | 83.7 | 32 |

8->4 removes 64 registers of accumulator state but only **14** of allocation: at acc<=4 the whole-kernel
high-water mark becomes the QK K-fragment `s_clause` preload (`global_load_b128 v[236:239]`), which is
invariant in acc_blocks. Every slice makes instr/WMMA *worse*, not better.

**2. The register cost of extra q-rows is +144, not +64 — and this corrects the repo's own ledger.**
`docs/SHARED_ATTENTION_LIVE_STATE_RESIDENCY_LEDGER_20260723.md` attributes 80 VGPR of long-lived state
(64 acc + 8 m + 8 l). In the actual binary PV acc is `v0:v63` and **`v64:v127` is a 64-VGPR
loop-invariant Q-fragment preload (16 `global_load_b128` at `0x17BC`, outside the loop) that the ledger
does not list at all.** True long-lived per-16-q-row state is **144 VGPR**. A second 16-row group costs
>=48 marginal VGPR even after reclaiming 56 by de-preloading Q: acc=8 -> **302 VGPR, 46 over the cap**.
Extra q-rows exist only at >=2 KV passes.

**3. The arithmetic never closes.** With `passes = 8/acc_blocks`, K VMEM scales as passes/q_groups, V VMEM
as 1/q_groups, and **R (QK + softmax + P repack) as passes**. Calibrated from the pp4096 V-transpose A/B
(`dSQ_BUSY/dVMEM_cycles = 0.363`): **VMEM-removable is 23.6% of busy; R is 76.4%.** One extra pass costs
+76.4%; perfect 2x load amortization buys -11.8%. Best feasible arm (q_groups=2, acc=8) is 1.134x and is
**VGPR-infeasible**; every feasible arm is a loss (q=2/acc=4 -> 0.598x).

Two independent ceilings on ANY load-side attention work: wall/VMEM slope extrapolated to zero VMEM =
**1.449x**; VALU-issue floor (22.400M busy / 13.158M VALU) = **1.702x**. The prize table needs 2x. At the
adversarially favourable transfer coefficient 1.0 the acc=4 arm is still 0.79x, so the verdict does not
depend on the fitted slope.

**4. This lever was ALREADY MEASURED in zero-VGPR form and lost.** `b53ebfebd` (G2 shared-K/V LDS
staging) is exactly Theory 1's amortization without the register cost — 128 scalar-half loads -> 4 `b128`
load sites — and it measured **1.65% SLOWER at KV4096** (two barriers per tile; a VMEM issue slot simply
becomes an LDS issue slot). The first version of this doc listed it under "what to drop" as an unrelated
dead end. It is not unrelated: it is this theory's refutation.

**Falsifier run, and it held.** The R-dominates decomposition predicts the shipped causal tile-skip (which
removes 12.1% of *both* VMEM and R) improves attention ~1.121x. Measured with PMC at ctx4096: aggregate
attention busy **800.1M -> 706.3M = 1.133x**, wall 717.0ms -> 648.2ms (1.106x), and VMEM share stays flat
(63-65% -> 62-67%) — the signature of removing VMEM and R proportionally. Per chunk the causal-waste model
also holds: kv512 **1.830x** (predicted 1.94x), kv2048 1.146x (1.14x), kv4096 1.088x (1.06x).

**Where the work actually is: R = 76.4% of SQ busy.** Per 16 WMMAs the shipped body spends 429 VALU,
**96 cross-lane `ds_bpermute`** (the online-softmax row max/sum reductions, invariant at every
acc_blocks), and 16 transcendentals. Nothing in this doc previously addressed the row reductions. That is
the only remaining path to a 2x attention kernel.

## THEORY 2 — GEMM LDS bank conflicts (untouched, on 79.6% of runtime)

**Claim:** all three top GEMM kernels report `lds_conflict` at **exactly 12.5% (= 1/8)**, identical across
three different shapes, against attention's 5.1%. That exactness reads as a systematic stride-8 conflict
in the graph-GEMM's LDS swizzle. Nobody has looked at this; these counters are the first time it has been
visible.

**Verify FIRST:** confirm 12.5% is not a quantization artifact of `lds_conflict = SQC_LDS_BANK_CONFLICT /
SQC_LDS_IDX_ACTIVE`. Attention's 5.1% shows the metric does vary, but three identical values want one
confirmation before anyone spends on a fix.

**Then test:** pad or swizzle the LDS tile in the graph-GEMM candidate (`PREFILL_GRAPH_GEMM` route,
`prefill_wmma_lds_dbuf_generated`, 2 LDS slots / 256 threads) and re-measure the conflict percentage AND
whole-model throughput. A conflict-rate drop with no throughput change means LDS stalls were not on the
critical path — record that as the verdict rather than chasing it further.

**TESTED 2026-07-24 (real, not artifact; one bounded padding change; net negative, reverted):**

*Metric verification.* Dumped the raw per-WGP PMC samples (not just the total/max/count the tracer
normally reports) for the top three `E_4_*` GEMM kernels and for attention, all under the same PMC
pipeline (`SQC_LDS_BANK_CONFLICT`/`SQC_LDS_IDX_ACTIVE`, both block `SQ`, same 48-WGP sample granularity —
no numerator/denominator scaling mismatch). Every nonzero-active WGP in every GEMM kernel reports
`conflict/active` at bit-exact `0.125`, independent of that WGP's absolute load (e.g. one dispatch showed
active in {524288, 655360, 786432} per WGP, conflict in {65536, 81920, 98304} respectively — each pair an
exact 1/8). Attention's ratio, sampled through the identical pipeline, varies continuously WGP-to-WGP
(0.0479–0.0489, no two values equal). The GEMM determinism is explained mechanistically, not by a broken
counter: `bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/candidate-set.json` shows all
four roles (attn_qo/attn_kv/ffn_down/ffn_gate_up) share one identical `schedule.lds` block (`strides.a=b=80`,
`padding=16`, `tile k=32/m=128/n=128`) — it's the same compile-time LDS swizzle geometry reused verbatim
across shapes, so an identical, data-independent conflict fraction is exactly what a real deterministic
swizzle-caused conflict looks like, not a measurement artifact. **Verdict: the 12.5% is real.**

*The fix attempt.* Tried the one bounded change specified: padded the LDS row stride from 80 to 96 bytes
(padding 16→32; unpadded row is 64 bytes = tile-k=32 × fp16). Edited `candidate-set.json`
(`strides.a/b`, `windows.a/b`), recomputed `canonical_identity` via
`extra.llm_research.runtime_specs._canonical_full_kernel_identity` so admission still validates (confirmed via
`admit_full_kernel_candidate_set` before spending GPU time). Result was the opposite of the hypothesis:

| | lds_conflict_pct (ctx512, top GEMM) | pp512 | pp1024 | pp2048 | pp4096 |
|---|---|---|---|---|---|
| baseline (stride 80) | 12.5% | 3607 | 3512 | 3327 | 3003 |
| padded (stride 96) | **50.0%** | 3399 (−5.8%) | 3314 (−5.6%) | 3144 (−5.5%) | 2849 (−5.1%) |

Conflict rate went 4x worse and whole-model throughput dropped ~5.5% uniformly across all four context
lengths — consistent, monotonic, and in the direction the counter predicted. This is strong independent
confirmation the counter is real AND that LDS conflicts on this route ARE on the critical path (contrary
to the doc's fallback framing that a conflict-rate change with no throughput change would be the likely
outcome — here throughput moved with conflict, in lockstep). **Change reverted; candidate-set.json is
back to the original stride-80/padding-16 state (verified byte-identical), nothing shipped.**

**Correction to the theory as written:** the theory's implicit direction (more padding = fewer conflicts)
was WRONG for this stride/tile geometry — 80 bytes (20 dwords, gcd(20,32)=4, period 8) mod the 128-byte
bank cycle already conflicts at a real but modest 1/8, and pushing to 96 bytes (24 dwords, gcd(24,32)=8,
period 4) made it a 1/4-periodic, 4-way-heavier pattern — moved to the WORSE side of the modulus, not the
better side. The naive "+1 element" pad trick is not safe here because the WMMA fragment-load addressing
(`rdna3_wmma_output_coord` / `wmma_fragment_loads` in `extra/llm_research/kernel_lds.py`) is a 2D lane→row/col
mapping, not a simple linear row scan, so the actual conflict-minimizing stride has to be derived from
that lane mapping (or swept empirically over the 16-byte-aligned choices under `max_lds_bytes=65536`), not
guessed. **PADDING IS EXHAUSTED — that suggested stride sweep is void.** `kernel_lds.py:238` requires "K row must
contain whole b128 vectors" and the candidate declares `load_vector_width=store_vector_width=8` (8 halves
= 16 B), so the stride MUST be 16-byte aligned. Under that constraint every stride is a multiple of 4
dwords, hence `gcd(S,32) >= 4` always:

| stride | dwords | gcd(S,32) | 16B-aligned | |
|---:|---:|---:|---|---|
| 64 | 16 | 16 | yes | far worse |
| 72 | 18 | 2 | **no** | illegal |
| **80** | 20 | **4** | yes | **current, 12.5% — optimal by padding** |
| 84 | 21 | **1** | **no** | conflict-free but illegal |
| 88 | 22 | 2 | **no** | illegal |
| 96 | 24 | 8 | yes | tested -> 50%, -5.5% |
| 112 | 28 | 4 | yes | same as 80, no gain |

The conflict-free strides are exactly the ones b128 forbids. **80 B is already the best achievable by
padding**, so the remaining move is an **XOR swizzle** — permute banks with an address XOR above the
16-byte boundary, preserving b128 alignment while decorrelating banks. That is the standard technique
precisely when vector-load alignment blocks padding.

**Prize sizing (upper-bound sketch):** sensitivity measured in the wrong direction is 5.5% throughput per
37.5pp of conflict ~= 0.15%/pp, so eliminating all 12.5pp extrapolates to roughly **+1.8%** — same order
as Theory 3, NOT the large win the 79.6%-of-runtime share suggests. Extrapolated from one point; treat as
a ceiling.

**CLOSED 2026-07-24 — conflict eliminated to bit-exact ZERO; throughput did not move. Theory 2 is worth
~+0.1%, not +1.8%. The "conflicts are on the critical path" finding above is WRONG.**

*The bank model, derived and then validated to the cycle.* RDNA3 services a wave32 b128 LDS access eight
lanes at a time (8 lanes x 4 dwords = the 32 x 4 B banks). A b128 slot only cares which aligned dword
*quad* it lands in, so `Q(row, slot) = (row*(stride//16) + slot) mod 8`, i.e. **`Q = (5*row + slot) mod 8`**
at stride 80. An octet is conflict-free iff its eight slots give eight distinct `Q`. Feeding the *real*
address math (`tinygrad/codegen/opt/kernel_lds.py:497-504` stores, `:514-516` fragment loads) through that
model reproduces every known datapoint exactly: stride 64 -> 72%, **80 -> 12.50%**, **96 -> 50.00%**,
112 -> 12.50%, 128 -> 86%. The `gcd(S,32)` model in the section above is *not* the mechanism (it happens to
rank these five strides in the same order); the quad-group invariant is.

*Where the conflicts actually were.* Not in the WMMA fragment loads — those are **already perfectly
conflict-free** at stride 80, because a load octet is eight consecutive rows at one fixed slot and
`5*row mod 8` is a bijection. **100% of the conflict is in the cooperative store**, which is 50%
conflicted: a store octet is 2 rows x 4 slots, and adjacent rows give `{c..c+3} u {c+5,c+6,c+7,c}` — `c`
twice, `c+4` unused. Loads are 768 of the 1024 LDS cycles per wave-epoch and stores 256, so
`0.5 * 256/1024 = 12.5%` — the measured number, from first principles.

*XOR swizzle: PROVABLY DEAD, and it was the wrong target.* The only address XOR available above the 16 B
boundary lives in the two slot-index bits, and the fragment load is one index plus a 16-wide `CONTRACT`
(32 contiguous bytes), so a swizzle may only relocate 32 B fragment halves, never individual b128 slots.
Exhaustive search over that space: the best pure bit-5 XOR is the identity (12.50%, no gain — it can only
permute slots *within* a row, and a store octet's per-row slot **set** is invariant under any such
permutation); the best expressible *layout* of any kind at stride <= 128 is 6.67%; a fully conflict-free
layout needs stride >= 144 B = 73728 B LDS, over the 64 KiB `max_lds_bytes`. Padding was exhausted and
swizzling is exhausted **because the conflict is not a property of the layout at all.**

*The actual fix (shipped): re-elect the store's lane->row map, not the layout.* Rows at distance **4**
give `{c..c+3} u {c+4..c+7}` = all eight. Rotating the low three row bits (`q -> ((q&1)<<2)|(q>>1)`, so
lane-quads 0..7 take rows 0,4,1,5,2,6,3,7) pairs rows four apart inside every octet while leaving each
lane-quad on one contiguous 64 B source row — so the wave issues the *same* global-memory segments, the
LDS layout is byte-identical, LDS bytes are unchanged, and the term is loop-invariant so it hoists out of
the K loop. `cooperative_store_row{,_rotation}` in `tinygrad/codegen/opt/kernel_lds.py`; the pure-Python
model in `extra/llm_research/kernel_lds.py` follows it. `candidate-set.json` is untouched (identities unchanged).

*Measured, paired same-session, ctx512 PMC (attention row as an unchanged-path control):*

| | lds_conflict_pct (all four GEMM roles) | SQC_LDS_BANK_CONFLICT | SQC_LDS_IDX_ACTIVE | attention control |
|---|---|---|---|---|
| naive election | 12.50000% (bit-exact) | 6291456 | 50331648 | 5.14681% |
| rotated election | **0.00000%** | **0** | 44040192 | 5.14681% |

`IDX_ACTIVE` fell by exactly **7/8**, matching the model's predicted 1024 -> 896 cycles to the cycle.
`SQ_INSTS_VALU` rose 0.11% (the hoisted address setup). Numerics `max_abs_err=6.104e-05 PASS` at Hd=64
and Hd=128; `8B: SDPA=198 FUSED=198 MATCH PASS`.

*Throughput: nothing.* Paired, interleaved, same-session, 2 reps (whole-prefill authority, tok/s):

| arm | conflict | LDS/WG | pp512 | pp1024 | pp2048 | pp4096 | vs base80 |
|---|---|---|---|---|---|---|---|
| base80 (naive) | 12.5% | 40960 B | 3574 / 3570 | 3477 / 3466 | 3290 / 3288 | 2979 / 2979 | — |
| **elect80 (shipped)** | **0.0%** | 40960 B | 3574 / 3573 | 3478 / 3476 | 3294 / 3297 | 2981 / 2984 | **+0.13%** |
| base112 (control) | 12.5% | 57344 B | 3566 / 3545 | 3469 / 3453 | 3287 / 3276 | 2977 / 2966 | −0.31% |

**+0.13% mean, sign-consistent but under the 0.59% back-to-back noise floor. The doc's +1.8% projection
overshot by ~14x.** The error was assuming linearity. Marginal sensitivity is a *threshold*, not a slope:
896 -> 1024 LDS cycles (+14%) costs 0.13% (0.009%/%), while 1024 -> 1792 (+75%, the stride-96 arm) costs
5.5% (0.073%/%) — 8x steeper. At the baseline operating point the LDS pipe has slack and the 12.5%
conflict was entirely hidden behind the GEMM's VMEM/VALU/WMMA work; stride 96 pushed it past saturation
into being co-critical. **So "throughput tracked the counter in lockstep" was a single-point coincidence,
not a slope.** The `base112` control also rules out LDS *footprint*/occupancy as the stride-96 mechanism:
+40% LDS bytes at identical conflict costs only 0.31%, not 5.5%.

The change is shipped anyway because it is free (zero LDS, +0.11% VALU, gates green) and it removes a
permanently misleading PMC signal — but **the 79.6%-of-runtime GEMM path does not have a bank-conflict
needle in it. Theory 2 is closed.**

## THEORY 3 — causal tile skipping (bounded, independent)

The kernel iterates **every** KV tile and masks in the softmax; it never skips a fully-masked tile.
Predicted waste: 1.94x at chunk 0, 1.14x at chunk 3, 1.06x at chunk 7, 1.121x (12.1%) aggregate over a
4096 prefill. **NOTE: those are ATTENTION-LOCAL ratios, not whole-model deltas** — the first version of
this doc put them next to a pp512 row and invited reading 1.94x as a 94% pp512 gain.

**IMPLEMENTED AND CONFIRMED (`c44905a18`, `PREFILL_CAUSAL_TILE_SKIP`, default off).** Expressed as a
dynamic KV-loop bound, not an exec-mask skip: `group` (gidx0) is already a runtime SPECIAL, so
`tiles_needed = ceildiv(query_start + q_tile*16 + 16, 16)` is a runtime UOp and `cstyle.py`'s RANGE rule
renders it with no renderer change. Exact (masked tile => weight 0, alpha 1), numerics 6.104e-05 at Hd=64
and Hd=128, 8B parity 198==198. Cost: +9 instructions/wave for the bound, independent of trip count.

Whole-model, **three independent same-session A/B pairs**:

| pair | pp512 | pp4096 |
|---|---|---|
| 1 | 3607 -> 3680 (+2.02%) | 3003 -> 3054 (+1.70%) |
| 2 | 3424 -> 3481 (+1.66%) | 2880 -> 2921 (+1.42%) |
| 3 | 3404 -> 3459 (+1.62%) | 2858 -> 2911 (+1.85%) |
| **mean** | **+1.77%** | **+1.66%** |

Back-to-back noise on the same config is 0.59%, so the signal is ~3x noise. Attention-local (PMC, ctx4096)
**1.133x aggregate**, matching the 1.121x prediction. Estimate was +2.8%; actual +1.66% — the projection
ignored the +9 instructions/wave. Still default-off: promotion needs a `route_manifest.py` row and an
authority gate, not a flag flip.

## Bounded and known
Free-transpose V via a **transposed V KV cache** (write once at generation time so prefill reads the
layout it wants for free): ceiling ~3111 tok/s at pp4096 vs llama 3160. Not shippable as the per-call
permute that was measured and refuted today.

## What to drop
Further micro-optimization of individual instruction buckets in the attention loop. Today measured the
cost of that approach directly. Also settled dead ends (do not re-run): VGPR/occupancy compaction
(1.43%/2.46% SLOWER), phase-ABI LDS state (added 2048 B LDS, reduced zero VGPRs), barrier removal
(<0.2%), and the P-repack LDS hypothesis (10 LDS instrs/tile; attention lds_conflict 5.1% vs the GEMMs'
12.5%). **G2 GQA K/V LDS sharing (1.65% slower) has been MOVED out of this list** — it is not an unrelated
dead end, it is Theory 1's refutation; see there.

## Measurement methodology (learned the hard way today)

1. **Absolute throughput on this box drifts ~5% across a long session.** The same config back-to-back
   varies 0.59%, but baselines hours apart differ by 5% (3607 vs 3424 at pp512). So a recorded baseline is
   NOT a valid comparator — **every small-win claim needs a same-session PAIRED A/B**, ideally repeated.
   The three Theory 3 pairs agree on +1.7% across a 5% absolute shift; that is what makes them credible.
2. **Never convert an instruction-count delta into a cycle delta at one measured rate.** `d16` = 3.02
   cyc/instr, `b128` = 6.61 (2.2x). This produced the 1.96x-vs-1.20x V-gather error.
3. **Price the machinery, not just the work removed.** Every local win projected today came in below
   estimate because the delivery cost was omitted: V-transpose (a strided scatter, ~7% not ~3%), causal
   skip (+9 instrs/wave).
4. **A bandwidth estimate badly understates a strided scatter.** bytes/HBM-peak is only valid for a
   streaming copy.
5. **Distinguish SHIPPED from default-off arms in every quoted number.** This doc got that wrong once.
