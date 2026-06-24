# Matmul-PV Diagnostic Candidate — Result

Date: 2026-06-21 (corrected; supersedes the first pass — see "Correction" below)

Scope: `docs/low-level-decode-attn-attribution-result-20260621.md` (`LOW_LEVEL_ATTRIBUTION_FIXABLE_CODEGEN`). The ISA
attribution named a bounded, codegen-justified lever: the dominant decode-attention kernel `flash_partial_coop_vec`
(PV = `prob @ V`, 24.7µs @ctx1024) emits **scalar fp16 V loads, 0 `v_dot2`, 0 LDS** (latency-bound, 201 GFLOPS),
while tinygrad's q·k matmul is fast **because** the tiled-GEMM codegen applies. Hypothesis: route the PV through the
tiled-matmul codegen instead of the scalar partial. **Diagnostic candidate, not a promised promotion** (predicted EV
~1.16× attention ≈ 3–4% whole-decode, W==D-marginal).

## Decision: **`MATMUL_PV_BLOCKED_BY_LAYOUT`**  (lifecycle gate verdict: `FAIL_LOCAL_AB`)

The diagnostic's codegen claim is **CONFIRMED** — the tiled-matmul PV is genuinely faster than the scalar partial
**when it does the fair amount of work**: at **ctx4096 the candidate WINS 1.13×** (≈ the predicted ~1.16×),
byte-band-correct. **But the lever is blocked at the gate context (ctx1024) by a layout/shape constraint, not by
codegen quality and not by "skinny M":** tinygrad's tiled-GEMM codegen fires only on a **concrete** contraction dim,
and the only concrete-K per-split PV matmul requires a **concrete** split count `Smax = MAXC/L = 32` (the symbolic
decode length `Tc` cannot be reshaped into a symbolic-count `(S, L)` tiled batched matmul — it raises `eval failed to
be a single number`). So at ctx1024/512 the matmul computes all 32 splits (the full MAXC KV) regardless of `Tc` →
4×/8× wasted work → **0.94×/0.88×**, missing the ≥1.05×@ctx1024 gate. The symbolic-`Tc` single-matmul form (which
respects `Tc`) is **not tiled at all** (13 GFLOPS).

Lifecycle gate → `FAIL_LOCAL_AB` (ctx1024 0.94× < 1.05×). Project verdict → `MATMUL_PV_BLOCKED_BY_LAYOUT`: the
codegen lever works (1.13×@ctx4096) but is unreachable Tc-proportionally at the gate context. Final disposition is
unchanged: **do not promote, no W==D, the bounded matmul-PV lever is exhausted.**

### Correction (why this supersedes the first pass)
The first pass measured a **non-split** PV (`prob[Hkv,G,Tc] @ V[Hkv,Tc,Hd]`, kernel `r_2_8_16_4_4_256_4`), batched
over **Hkv=8 only**, which **collapses the KV-split parallelism** (the decode-T=1 principle) → ~50 GFLOPS, and from
that wrongly concluded "matmul-PV is worse than the scalar partial; the skinny M=G=4 GEMM defeats tinygrad's tiling,"
and rested. That attribution is **refuted here by measurement**: the **split-preserving** per-split matmul (the same
M=G=4) tiles at **~1078 GFLOPS** and **wins 1.13× at ctx4096**. Tiling does not fail on skinny-M; the prior form just
underfilled the GPU (8 workgroups). The real blocker is the **symbolic split count** (below). The bottom-line verdict
(no promotion, rest bounded decode) is the same; the **root cause and the ledger refutation are corrected** so future
work is not told the false thing "tiled matmul can't help decode PV."

## Phase 0 — design audit (written before coding, per scope)

**Why reopened on new ISA evidence.** The prior coop-qk-preserving closure was timing-only ("combine ~1µs, no
delta"). The 2026-06-21 ISA attribution is **new evidence**: a *specific* fixable inefficiency — `flash_partial_coop_vec`
is a hand-rolled scalar reduction (0 `v_dot2`, 0 LDS) while matmul-shaped ops get the tiled codegen.

**Why it differs from prior closures.** Path A fused the softmax+V *tail* (kept the scalar partial); the
warp/north-star tiles replaced coop's *matmul q·k* with a slower hand dot. This candidate is the inverse: keep coop's
matmul q·k and softmax/prob production, replace **only** the scalar PV partial with a tinygrad matmul.

**Exact PV expression and tensor shapes.** PV per (kv-head, split `s`, query-head `g`):
`pout[kvh,g,s,d] = Σ_{j<L} prob[kvh,g,s,j] · v[kvh, s·L+j, d]`. As a batched matmul, batch = `(kvh, s)`, M=`G`=4,
K=`L`=128, N=`Hd`=128: `A[kvh,s,g,j] @ V[kvh,s,j,d] → PV[kvh,s,g,d]`, where `A` = `prob`.reshape`[Hkv,G,S,L]`.permute`(0,2,1,3)`.
Denominator `l[kvh,s,g] = Σ_j prob` (= `A.sum(-1)`). Downstream `flash_gmax` + **lean natural-layout** `den`/`combine`
(read `PV[Hkv,S,G,Hd]` and `l[Hkv,S,G]` directly — no pout `cat`/permute) do the LSE merge.

**How tinygrad tiled-matmul codegen is invoked.** Plain `Tensor @ Tensor` (the op that makes q·k fast). GEMM tiling
(LDS staging + vectorized loads) fires **only on a concrete reduce dim**. K=`L`=128 is concrete ⇒ tiled (1078 GFLOPS).
That forces a concrete split count `Smax` (the blocker).

**Expected kernel-count / materialization tradeoff.** Replaces 1 scalar partial with 1 small prob-permute copy
(~0.3µs) + 1 tiled PV matmul + 1 `l` reduce (+2 kernels). The lean natural-layout `den`/`combine` avoid a pout
`cat`+permute (an earlier non-lean assembly with those copies was 0.43/0.47/0.66× — the copies erased the win).

**First gate & stop condition.** Local A/B vs `gqa_coop_vec`: total attention ≥1.05×@ctx1024, no ctx4096 regression.
If PV improves but total misses → `FAIL_LOCAL_AB`/`DIAGNOSTIC_ONLY`, stop before any W==D. (It missed at ctx1024.)

### Phase-0 questions answered
1. **Current `flash_partial` in/out shape?** in: `prob[Hq,MAXC]` (f32, 0 outside `Tc`), `vc[Hkv,MAXC,Hd]` (f16).
   out: `pout[(h·S+s)·W+d]`, `W=Hd+1` (col `Hd` folds the denom via 1-augmented V). `gqa_coop_vec` maps `d` to LOCAL
   threads (coalesced V loads) but is still **scalar** (0 `v_dot2`, 0 LDS).
2. **Can `prob @ V` be a matmul without layout copies that erase the win?** Math yes; the *concrete-K* tiled form
   needs a **concrete split count** (symbolic `Tc` is not reshapeable into `(S,L)`). The lean form needs only one
   small `prob`-permute copy (~0.3µs); the matmul is tiled. The non-lean `cat`+pout-permute copies *do* erase it — avoided.
3. **Preserves correctness & current softmax semantics?** Yes — exact stable softmax (per-split max + LSE), byte-band
   identical to coop (rel_rmse 7–8e-4). The only structural change forced by the matmul is the **concrete `Smax`**
   split count (vs coop's symbolic `S=ceildiv(Tc,L)`) — same kernels, more (empty) splits.
4. **Increases materialization / kernel count?** +2 kernels + 1 small prob-permute copy; and — the blocker — it reads
   the **full MAXC** V at every ctx (concrete `Smax`), not just `Tc`.
5. **Expected Amdahl impact?** PV ~35% of ~70µs attention; attention ~23% of decode. 1.16× attention → ~3–4%
   whole-decode = below the ≥5% W==D bar even if local passed (it did not).

## Phase 1 — standalone local A/B (`extra/qk_matmul_pv_diagnostic_ab.py`)

Candidate = coop's matmul q·k + `flash_max` + `flash_prob` (unchanged) → **split-preserving tiled matmul PV**
(concrete `Smax`) + `l` reduce → `flash_gmax` + lean natural-layout `den`/`combine`. Comparator = `gqa_coop_vec`
(canonical). Clock-pinned, throughput (back-to-back, authoritative per the oracle/dispatch-probe method), median-of-3.

### Correctness (vs numpy reference)

| ctx | rel_rmse | max_abs | gate (≤1e-3) |
|---:|---:|---:|---|
| 512  | 7.4e-4 | 3.9e-4 | PASS |
| 1024 | 7.0e-4 | 1.9e-4 | PASS |
| 4096 | 8.0e-4 | 1.7e-4 | PASS |

(≤1e-5 unattainable — the matmul reorders fp accumulation vs the scalar partial; ~7e-4 matches coop's own ~2e-4
fp-reassoc band, so ≤1e-3 is the justified threshold. No layout mismatch.)

### Local A/B (throughput, clock-pinned, vs gqa_coop_vec)

| ctx | candidate µs | gqa_coop_vec µs | **speedup** | note |
|---:|---:|---:|---:|---|
| 512  | 86.1  | 75.7  | **0.879×** | concrete Smax=32 ≫ S=4 → 8× extra split work (full-MAXC reads) |
| 1024 | 91.2  | 85.3  | **0.936×** | concrete Smax=32 ≫ S=8 → 4× extra; **gate FAIL** |
| 4096 | 127.7 | 144.4 | **1.131×** | Smax=32 = S=32 → **fair → tiled matmul PV WINS** |

**Gate FAIL** (needs ≥1.05×@ctx1024; got 0.936×).

### Codegen evidence (the BLOCKED_BY_LAYOUT root cause; `--gflops`)

`DEBUG=2` per-kernel GFLOPS of the PV matmul in three forms:

| PV form | reduce dim K | parallel wg | tiled? | GFLOPS |
|---|---|---:|---|---:|
| **split, concrete `Smax`, K=`L`=128 concrete** | concrete | Hkv·Smax = **256** | **yes (LDS-tiled)** | **~1078** |
| non-split, concrete big-K=`Tc` (`r_2_8_16_4_4_256_4`, the prior form) | concrete | Hkv = **8** | partial (occupancy-starved) | **~50** |
| single full matmul, K=`Tc` **symbolic** | symbolic | — | **no** | **~13** |
| (reference) q·k matmul, K=`Hd`=128 concrete | concrete | — | yes | ~545 |

The q·k is fast because its K=`Hd`=128 is concrete and `Tc` is its *output (N)* dim. The PV's contraction is over the
*keys* — concrete only per-split (`L`), which forces a concrete split count `Smax`. **tinygrad cannot express a
symbolic-count tiled batched matmul**, and the symbolic-K single matmul is not tiled (13 GFLOPS). So the tiled lever
is reachable only at concrete `Smax` = full MAXC, fair only when `Tc ≈ MAXC` (ctx4096, where it wins). The first
pass's "skinny M=4 defeats tiling" is wrong — the split form has the *same* M=4 and tiles at 1078 GFLOPS; the prior
non-split form was **parallelism-collapsed** (8 wg), the project's own decode-T=1 anti-pattern.

## Phase 2 — lifecycle

decode_eval candidate `matmul_pv_diagnostic` (family `attention_split`, `ab_script`) → **`FAIL_LOCAL_AB`**
(0.936×@ctx1024) → `refute_candidate`. Refutation corrected to `matmul_pv_symbolic_split_layout_wall` (root cause =
symbolic-split-count layout limit; prunes "PV as a tiled matmul for decode" reopens that ignore the blocker — but
records that the tiled lever *works* and wins at ctx4096, so it is BLOCKED, not codegen-incapable).

## Phase 3 — W==D: **NOT reached** (local A/B failed — discipline = stop). No env-gated route added.

## Interpretation — for the remaining llama gap

The matmul-PV diagnostic **validates the ISA diagnosis** (tiled-matmul PV codegen is real and beats the scalar
partial — 1.13×@ctx4096) and pins the bounded lever's limit precisely: it cannot be invoked Tc-proportionally at the
gate context because tinygrad lacks a **symbolic-count tiled batched matmul**. This is a layout/codegen-capability
limit, the same family as the deep fused-flash blocker — the full llama-class win still requires the **deep LDS-tiled
fused-flash codegen capability** (a single fused kernel with concrete tiling over the symbolic KV), which would also
unblock this. The bounded matmul-PV lever is **exhausted** (strongest lean form measured; the only fix is the deep
capability). Per discipline: **stopped before any W==D route**.

## Acceptance gates

| gate | result |
|---|---|
| G1 scalar PV issue restated from ISA evidence | PASS (Phase 0; 0 `v_dot2`/0 LDS scalar partial) |
| G2 candidate uses tiled-matmul path or explains why not | PASS (tiled concrete-K matmul used, 1078 GFLOPS; symbolic-K/non-split untiled explained) |
| G3 correctness measured | PASS (rel_rmse 7–8e-4) |
| G4 local A/B vs gqa_coop_vec measured | PASS (0.879/0.936/1.131×) |
| G5 candidate through decode_eval/lifecycle | PASS (`FAIL_LOCAL_AB`) |
| G6 no W==D unless local passes | PASS (not added) |
| G7 no default/model route change unless gated | PASS (no `tinygrad/`, no model route) |
| G8 no closed lane reopened beyond ISA-justified PV diagnostic | PASS |
| G9 policy guard passes | PASS |
| G10 tree clean after commit | PASS (commit below; pre-existing unrelated dirty `structure/.../performance-primitive-research-principles.md` listed) |

## Next action

**Do not iterate matmul-PV variants** (the strongest split-preserving lean form is banked; the blocker is a tinygrad
capability gap — symbolic-count tiled batched matmul — not a tunable). The bounded decode lever space is exhausted.
Remaining options: the **deep LDS-tiled fused-flash codegen** capability (multi-week; symbolic tiling over KV — would
also unblock this), or **`REST_DECODE`** with this counter+codegen-level evidence. The llama oracle stays the
validated target / non-promotable reference.

## Changed files
`extra/qk_matmul_pv_diagnostic_ab.py` (rewritten: split-preserving primary + non-split contrast + `--gflops`),
`bench/qk-matmul-pv-diagnostic/` (artifacts), `bench/qk-decode-eval/candidates.json`,
`bench/qk-lifecycle-search/refutations.json`, this doc + handoff/READMEs.

## Boundary
No `tinygrad/` change, no model route/default, no W==D route, no closed lane reopened, no tuning sweep, no
weak-baseline benchmarking. Clock-pinned diagnostic; perf-state restored to `auto`. The ctx4096 1.13× is a standalone
diagnostic number (not a promotion — the gate is ctx1024).
