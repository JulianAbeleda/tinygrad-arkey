# Decode — Post-Matmul-PV Strategic Exhaustion Scope

Date: 2026-06-21

Scope/audit only (no kernel, no model/default route, no flag tuning, no closed lane reopened). After the corrected
Matmul-PV result (`MATMUL_PV_BLOCKED_BY_LAYOUT`), this doc exhausts the three remaining project options and decides
the next month of work:

1. `REST_DECODE` + pivot to v2/search/tooling cleanup.
2. Symbolic-count tiled batched matmul (the specific blocker Matmul-PV surfaced).
3. Full LDS-tiled fused-flash codegen (the llama-class target).

## Decision: **`STRATEGY_RECOMMEND_FULL_FUSED_FLASH`** → next project **`POST_MATMUL_PV_FULL_FUSED_FLASH`** (gate-first)

The bounded decode space is **fully exhausted** and only a **deep codegen capability** can close the 5–6× standalone
attention gap to llama. Of the two deep capabilities, **option 3 (fused-flash) subsumes option 2 (symbolic-count
tiled matmul)** — a fused-flash kernel that tiles over the symbolic KV *requires* the symbolic-count tiled schedule —
and **option 2 standalone is W==D-marginal** (matmul-PV's ceiling is ~1.13× attention ≈ 3–4% whole-decode, below the
≥5% bar even with the full-MAXC overread removed). So option 2 is never the right *standalone* next project; it is a
sub-capability built *inside* option 3 where it has a real payoff. **Option 1 (rest+v2) is premature**: the cheap,
decisive fused-flash **first gate has never been run**, and "a search/v2 system without a llama-beating primitive is
infrastructure, not completion" (north star). The disciplined move is **not** to fund a blind multi-week compiler
project, and **not** to rest blind, but to **fund the cheap first gate of fused-flash** (a concrete-shape toy
LDS-tiled fused microkernel) with a **hard stop**: it either opens the deep project or triggers `REST_DECODE`+v2.
Matmul-PV gives new leverage — tinygrad's tiled-GEMM codegen **does** emit fast LDS-staged tiled code (1078 GFLOPS)
for concrete shapes — so the building block exists; the open question (resolved cheaply by the first gate) is whether
a *fused* q·k+softmax+PV tile can be expressed and beat `gqa_coop_vec`.

**Fallback is explicit:** if the fused-flash first gate fails (can't express LDS-tiled fused dataflow, or expressible
but slower than `gqa_coop_vec`), the recommendation collapses to `POST_MATMUL_PV_REST_DECODE_V2` with the full v2
transition scope (sketched in Phase 2 Option 1 below) — decode is then capped at tinygrad's backend ceiling and the
project pivots to v2/search consolidation.

---

## Phase 0 — final decode evidence/closure table

| lane | result | key evidence | final verdict | reopen condition |
|---|---|---|---|---|
| baseline decode (`gqa_coop_vec`, FLASH_L=128) | 68.1/66.4/60.7 tok/s @512/1024/4096 ≈ **~67% llama** | `qk_decode_runtime_overhead`; handoff §1 | **shipped default**; the comparator | n/a (it is the baseline) |
| weight-GEMV / MMVQ | parity (llama mmvq == tinygrad GEMV ~7.9ms) | refutations; `decode-gap-is-attention-not-weight-gemv` | closed (not the gap) | new int8 lifecycle only |
| FFN activation fusion | byte-exact, ~0% (work-conserved, not launch-recoverable) | `decode-fusion-build-result-20260620` | closed | — |
| attention micro/reduce-stat fusion | no-go (intrinsic O(KV) QK/softmax) | same | closed | — |
| q8 FFN opt-in | ~1.06× W==D, dNLL 0.0029 ≤ 0.01 | `decode_q8_model_route_timing_audit` | **opt-in, default-off** | no default promo |
| FLASH_L=64 (more KV-splits) | local ~1.08×@1024 byte-exact; W==D +1.8%@1024 / **−1.2%@4096** | `decode-vector-flash-tile-realigned` | `LOCAL_PASS_WD_FAIL`; **not promoted** | — |
| raw fused flash tile | byte-exact but slower | refutations | closed | — |
| scalar LDS+GQA tile | 0.21× (workgroup collapse / per-lane redundancy) | refutations | closed | — |
| warp-cooperative tile | flat ~163µs > coop matmul (hand dot latency-bound) | `north-star-decode-attention-redesign-audit` | closed | — |
| vector/compact/stream-k combine | combine ~1µs negligible; not the lever | redesign audit | closed | — |
| north-star flash_attn_tile (hand) | `FAIL_LOCAL_AB` 0.58/0.89× | `north-star-flash-attn-tile-execution` | closed (hand form) | — |
| Path A fused softmax+V tail | `FAIL_LOCAL_AB` 0.725/0.876× (per-lane exp redundancy; full online-max BLOCKED_BY_IDIOM) | `fused-softmax-v-tail-candidate` | closed; **do not iterate** | — |
| llama flash_attn_tile **oracle** | llama **5.87/5.71/4.77× faster standalone** (pure GPU) | `llama-flash-attn-tile-oracle` | `PASS_ORACLE_LOCAL_AB`; **non-promotable target** | n/a (reference) |
| low-level ISA attribution | scalar `flash_partial` (0 `v_dot2`, 0 LDS, 100% occ, 0 spill) vs llama LDS `v_dot2` tile | `low-level-decode-attn-attribution` | `FIXABLE_CODEGEN` (diagnostic) | n/a |
| **Matmul-PV diagnostic** | gate `FAIL_LOCAL_AB` 0.88/0.94/**1.13×**; split tiled PV **1078 GFLOPS, wins@ctx4096**; blocked by symbolic split count | `matmul-pv-diagnostic-result` | **`BLOCKED_BY_LAYOUT`**; bounded lever exhausted | only via deep symbolic-tiled / fused-flash capability |
| HCQ live-counter tooling | rocprofv3 blind to HCQ (direct-ring/doorbell); rocprof-compute broken; PMC backend reads 0 | `tinygrad-hcq-profiling-visibility` | `USE_NATIVE_ATTRIBUTION_ONLY` | live counters become essential (low EV) |
| native attribution (ISA/resources/ProfileGraphEvent/ATT) | sufficed for every recent verdict | same | **standing tool** | n/a |

**Conclusion:** every *bounded* decode lane is refuted or shipped; the only open levers are deep codegen capabilities.

---

## Phase 1 — remaining capability map

| capability | evidence it's the blocker | expected benefit | affected contexts | complexity | first gate | risk |
|---|---|---|---|---|---|---|
| **symbolic-count tiled batched matmul** | matmul-PV: concrete-K split PV tiles @1078 GFLOPS & wins@ctx4096, but symbolic Tc can't reshape→(S,L); symbolic-K=13 GFLOPS | recover matmul-PV @ctx512/1024 without full-MAXC overread → ~1.13× attention ≈ **3–4% decode** (W==D-marginal); plus other dynamic-shape matmuls | ctx512/1024 (ctx4096 already fair) | medium-high (linearizer shape specialization / JIT-bucket caching) | µkernel: symbolic-K matmul emits tiled path & beats current symbolic path >10× | compile/JIT explosion; W==D below bar |
| **full LDS-tiled fused-flash** | llama oracle 5–6× standalone; ISA shows fused `v_dot2`+LDS single tile; tinygrad emits 6 scalar kernels | the actual **5–6×** attention path → the only route to **beat llama** decode | all (512/1024/4096) | high (multi-week compiler/renderer; maybe AMDGCN escape hatch) | toy concrete-shape LDS-tiled fused µkernel beats `gqa_coop_vec` @ctx1024 ≥1.05× | inexpressible without raw ASM; Path-A-style fusion loss; idiom walls |
| **llama source/HSACO byte-level oracle** | source bounded (no WMMA/cp_async, gfx1100 HSACO) per oracle Phase 0 | re-runnable byte-level target for codegen validation | n/a (reference) | medium (bounded port, deferred) | port compiles + byte-exact vs profiling oracle | artifact dependency; non-promotable |
| **native attribution / tooling only** | sufficed for all recent verdicts; HCQ live counters dead-end | keeps verdicts trustworthy without live counters | n/a | low (exists) | already passing | none (standing) |
| **v2 / search / tooling cleanup** | north star requires clean v2 + closed lifecycle search | maintainable execution surface; required for *completion* (not for beating llama) | n/a | medium (audit + migration) | v2 runs W==D within noise of repo | premature if decode story unresolved |

**Key structural fact:** the **fused-flash capability subsumes the symbolic-count tiled matmul** (a fused tile over
symbolic KV needs the symbolic-count tiled schedule). So funding symbolic-tiled-matmul *standalone* buys a marginal,
W==D-failing matmul-PV patch; funding it *inside* fused-flash buys it where the payoff is real.

---

## Phase 2 — exhaustive option analysis

### Option 1 — `REST_DECODE` / pivot to v2-search-tooling

- **Thesis:** bounded decode is exhausted and the deep capabilities are high-risk/unfunded; bank the won ground
  (oracle + refutation map), stop decode experiments, and build the clean `tinygrad-v2` execution repo + hardened
  lifecycle search the north star requires anyway.
- **Why now:** every bounded lane is closed; auto-clock W==D variance (<0.6%) and the evaluator/lifecycle loop are
  already built; v2 is a stated completion component; consolidating preserves provenance and reduces repo entropy.
- **Why NOT now:** **premature** — the cheap, decisive fused-flash *first gate* has never been run. Resting before
  that gate consolidates v2 around an *unresolved* decode story, and "a search system that cannot beat llama is not
  enough / infrastructure, not completion" (north star). Rest is correct only *after* the fused-flash gate fails.
- **Expected upside:** certain, modest — maintainable surface, faster future iteration, clean docs/artifacts. **No
  llama win.**
- **Cost:** v2 audit + migration (keep/drop, parity gates) — ~1–2 weeks.
- **Risk:** low technical; strategic risk of declaring decode "done" while the last cheap experiment is untried.
- **Files likely touched (if chosen):** new `tinygrad-v2/` workspace; `bench/README.md`, `docs/README.md`,
  handoff; migration manifests; **no** `tinygrad/` decode change.
- **Evaluator/lifecycle integration:** preserve `extra/qk_decode_eval.py`, `qk_lifecycle_search_loop.py`,
  `candidates.json`, `refutations.json`, `binding_templates.json`, the llama oracle candidate, and W==D harnesses as
  first-class v2 citizens.
- **First gate:** v2 runs full W==D decode within measurement noise of the research repo; lifecycle-search candidate
  generation + decode_eval run in v2; quality gates pass.
- **Stop condition:** v2 parity fails (a migration bug) → fix before cutover; never delete research provenance.
- **What would prove it wrong:** the fused-flash first gate passing (then rest was premature).
- **North-star effect:** advances pillars 2 (method) & 3 (clean repo) but **not** pillar 1 (beat llama).

### Option 2 — symbolic-count tiled batched matmul

- **Thesis:** target the exact blocker matmul-PV surfaced — make tinygrad emit a *tiled* matmul whose batch/split
  count is **symbolic** (dynamic over ctx) while keeping the concrete-K tiled schedule, so the split PV (1078 GFLOPS
  @ctx4096) becomes reachable Tc-proportionally at ctx512/1024 without full-MAXC overread.
- **Why now:** it is the *specific, named* capability gap; the win at ctx4096 (1.13×) proves the codegen works; the
  building block (concrete-K tiled matmul) exists.
- **Why NOT now:** **dominated.** (a) Standalone payoff is **W==D-marginal**: even with the overread fixed, matmul-PV
  caps at ~1.13× attention ≈ 3–4% whole-decode < the ≥5% bar → `LOCAL_PASS_WD_FAIL` at best (same class as
  FLASH_L=64). (b) It is a **sub-capability of fused-flash** — building it alone, then again inside fused-flash, is
  wasteful; build it where the payoff is real. (c) Risk of concrete-JIT bucket explosion (one tiled graph per ctx
  bucket) for a marginal gain.
- **Expected upside:** recover ctx512/1024 matmul-PV to ~1.1× attention; **~3–4% decode** ceiling alone.
- **Cost:** medium-high — linearizer/renderer shape specialization + JIT bucketing — ~2–4 weeks for a marginal
  result.
- **Risk:** compile/cache explosion; broad shape-system changes leaking into other paths; local A/B may still miss if
  the symbolic tiling carries overhead; **W==D below bar even on success.**
- **Files likely touched:** `tinygrad/codegen/*`, `tinygrad/uop/*` (shape/range specialization), JIT cache;
  `extra/qk_matmul_pv_diagnostic_ab.py` rerun. **Deep `tinygrad/` change** — higher blast radius.
- **Evaluator/lifecycle integration:** rerun `matmul_pv_diagnostic` through decode_eval; expected
  `LOCAL_PASS_WD_FAIL`.
- **First gate:** a microkernel where a symbolic-K (or symbolic-split-count) matmul emits the **tiled** path and
  beats the current symbolic path **>10×** (≈13→>130 GFLOPS); then matmul-PV rerun clears local A/B @ctx1024 ≥1.05×
  **or** proves the Amdahl cap.
- **Stop condition:** requires a broad shape-system rewrite; causes compile explosion; local A/B remains <1.05×; or
  W==D <5% (the likely outcome) → fold the capability into the fused-flash project instead.
- **What would prove it wrong:** matmul-PV W==D ≥5%@1024 after the fix (unlikely given the Amdahl math).
- **North-star effect:** small; a marginal decode bump, not a llama-beating primitive. Best realized *inside* option 3.

### Option 3 — full LDS-tiled fused-flash codegen

- **Thesis:** build the llama-class primitive directly — one fused decode-attention kernel: LDS-staged K/V, dense
  vector/`v_dot2` q·k, register online-softmax `(m,l,acc)`, GQA/query-head column packing, many KV-split blocks, an
  efficient combine, **no WMMA** — matching llama's validated `flash_attn_tile`.
- **Why now:** it is the **only** path to the project's pillar-1 goal (beat llama, 5–6× standalone gap); the target is
  **validated** (oracle, non-promotable) and the source is **bounded** (no WMMA/cp_async, gfx1100 HSACO); matmul-PV
  newly proves tinygrad emits **fast LDS-tiled** code (1078 GFLOPS) for concrete shapes, so the building block exists;
  and the **first gate is cheap** (a concrete-shape toy fused µkernel), bounding the risk before any month-long bet.
- **Why NOT now (honest counter-evidence):** discouraging priors — Path A (fused softmax+V tail) **lost** to per-lane
  exp redundancy; the full online-max removal is `BLOCKED_BY_IDIOM` (two-granularity store wall); warp/north-star/
  scalar-LDS hand tiles all lost; the `flash_fused_multireduce_linearizer_wall` exists (the coupled kernel *verifies/
  runs* via the `UOp.set/.after` idiom but is not proven to emit *fast LDS-tiled* code). It may need an AMDGCN/HSACO
  escape hatch with a non-trivial integration path. These are exactly why the project is **gate-first** (see scope).
- **Expected upside:** the real prize — up to llama-class attention (~5–6× the attention kernels) → potentially
  **beat llama decode** under W==D (pillar 1).
- **Cost:** high — multi-week compiler/renderer (+ possible raw-ASM escape hatch). **But the first gate is days, not
  weeks**, and gates the rest.
- **Risk:** highest — inexpressibility, fusion loss (Path A precedent), idiom walls, ASM integration. Mitigated by the
  hard-stop gate ladder.
- **Files likely touched:** `extra/` (toy µkernel + A/B + oracle reuse), then if it passes `tinygrad/codegen/*`,
  `tinygrad/renderer/*`, possibly an AMDGCN/HSACO escape hatch; `decode_eval` north-star binding (already exists).
- **Evaluator/lifecycle integration:** reuse `north_star_flash_attn_tile_v0` binding + `decode_eval` reference
  oracle; register the native candidate; W==D promotion via the standing harness.
- **First gate:** a toy **concrete-shape** (fixed ctx1024, single bound start_pos) LDS-tiled fused-flash decode
  µkernel — q·k + online softmax + PV in **one** kernel, emitting vectorized/`v_dot2` + LDS-staged K/V (no WMMA),
  built on the concrete-K tiled-matmul building block — that is **value-correct (rel_rmse ≤1e-3)** AND **≥1.05× vs
  `gqa_coop_vec` standalone @ctx1024**.
- **Stop condition:** can't express LDS-tiled fused dataflow without raw ASM *and* no integration path; expressible
  but **slower** than `gqa_coop_vec` (Path-A/warp precedent); or concrete passes but symbolic generalization needs
  the symbolic-count tiled matmul which *itself* can't be expressed → narrow or `REST_DECODE`.
- **What would prove it wrong:** the toy fused tile loses to `gqa_coop_vec` at concrete ctx1024 (then the deep tile is
  unreachable in tinygrad → rest).
- **North-star effect:** directly targets pillar 1 (beat llama); subsumes option 2; feeds the lifecycle search a real
  llama-beating route to maintain.

---

## Phase 3 — ranked recommendation

| rank | option | recommendation | why | first gate | stop condition |
|---:|---|---|---|---|---|
| **1** | **Full LDS-tiled fused-flash** | **PURSUE (gate-first)** | only path to beat llama (pillar 1); subsumes option 2; building block (1078-GFLOPS tiled LDS matmul) exists; **cheap first gate bounds the risk** | toy concrete-ctx1024 fused LDS tile value-correct + ≥1.05× vs `gqa_coop_vec` | toy tile inexpressible w/o ASM, or slower than coop, or symbolic generalization blocked → REST |
| 2 | REST_DECODE / v2 | **FALLBACK** (do iff rank-1 first gate fails) | bounded decode exhausted; v2/search required for *completion* but is "infra without a llama-win"; correct *after* the cheap gate, not before | v2 W==D parity within noise; lifecycle-search runs in v2 | v2 parity bug → fix before cutover |
| 3 | symbolic-count tiled matmul | **DO NOT do standalone** | dominated: W==D-marginal (~3–4% decode < 5% bar) and a sub-capability of rank-1; build it *inside* fused-flash | symbolic-K matmul emits tiled path, >10× the symbolic path | W==D <5% (expected) → fold into rank-1 |

**Decision enum: `STRATEGY_RECOMMEND_FULL_FUSED_FLASH`.** Next project: **`POST_MATMUL_PV_FULL_FUSED_FLASH`** (gate-first,
with `POST_MATMUL_PV_REST_DECODE_V2` as the documented fallback on first-gate failure).

---

## Phase 4 — chosen scope: `POST_MATMUL_PV_FULL_FUSED_FLASH` (gate-first)

### Objective
A native tinygrad LDS-tiled **fused-flash decode-attention** primitive that beats `gqa_coop_vec` in local A/B and
then llama under W==D — built **gate-first** so a cheap concrete-shape µkernel decides whether the deep project is
funded. **No default/model route until in-model W==D passes; default-off and shape-guarded when shipped.**

### Exact files to read first
- `docs/matmul-pv-diagnostic-result-20260621.md` (the 1078-GFLOPS tiled-LDS-matmul building block + the symbolic-split blocker)
- `docs/low-level-decode-attn-attribution-result-20260621.md` (llama's fused `v_dot2`+LDS tile vs tinygrad's 6 scalar kernels)
- `docs/llama-flash-attn-tile-oracle-result-20260621.md` (validated target, bounded source, non-promotable oracle)
- `docs/fused-softmax-v-tail-candidate-result-20260621.md` (Path A failure mode: per-lane exp redundancy; the BLOCKED_BY_IDIOM two-granularity store wall)
- `docs/native-fused-flash-linearizer-scope-20260621.md` (the coupled online-softmax+V kernel verifies/runs via `UOp.set/.after`; the idiom pitfalls)
- `docs/gpu-low-level-control-tooling-reference-20260621.md` (when/how to drop to AMDGCN/HSACO; escape-hatch rule)
- `structure/Development/performance-primitive-research-principles.md` (decode T=1 parallelism rule; whole-primitive gates)
- `extra/qk_flash_decode.py` (the `gqa_coop_vec` comparator + the split/softmax/combine kernels to fuse)
- `extra/qk_matmul_pv_diagnostic_ab.py`, `extra/qk_llama_flash_attn_tile_oracle_ab.py` (A/B + oracle harness patterns)
- `extra/qk_decode_eval.py`, `bench/qk-decode-eval/{candidates.json,binding_templates.json}` (`north_star_flash_attn_tile_v0` binding to reuse)

### Phases & gates
- **Phase 1 — FIRST GATE (cheap, days): concrete-shape toy fused tile.** One kernel, **fixed ctx1024** (single bound
  start_pos, concrete K), LDS-staged K/V + vectorized/`v_dot2` q·k + register online-softmax `(m,l,acc)` + PV, built
  on the concrete-K tiled-matmul building block. **Gate:** value-correct (rel_rmse ≤1e-3 vs numpy) **AND** ≥1.05× vs
  `gqa_coop_vec` standalone @ctx1024 (clock-pinned throughput, median, the oracle method). **Stop if:** inexpressible
  without raw ASM (and no integration path), or expressible but <1.05× (Path-A/warp precedent).
- **Phase 2 — symbolic generalization (only if Phase 1 passes): the option-2 sub-capability.** Make the KV-tile loop
  run over **symbolic** ctx while preserving the tiled schedule (no full-MAXC overread). **Gate:** symbolic kernel
  within ~10% of the concrete kernel's perf at ctx1024 and correct at ctx512/1024/4096; no JIT-bucket explosion
  (bounded number of ctx graphs). **Stop if:** symbolic tiling can't preserve the schedule, or compile/cache
  explosion.
- **Phase 3 — close toward llama:** GQA query-head column packing + many KV-split blocks + efficient combine.
  **Gate:** standalone attention ≥1.5× vs `gqa_coop_vec` @ctx1024 and trending toward the llama oracle (5–6×).
- **Phase 4 — in-model W==D:** env-gated default-off route, unsupported shapes fall back. **Gate:** ≥5%@ctx1024 /
  ≥7%@ctx4096 whole-decode, no ctx512 regression, median-of-5 PROFILE-off; greedy byte-exact or dNLL ≤0.01.
- **Phase 5 — lifecycle/policy:** register the native candidate (reuse `north_star_flash_attn_tile_v0`); centralize
  the route + fallback; promotion only via decode_eval/W==D.

### Artifacts
`bench/qk-fused-flash-native/` (per-phase JSON), decode_eval run artifacts, per-phase result docs
(`docs/fused-flash-native-phaseN-result-*.md`), refutation entries on any failed phase.

### Lifecycle/evaluator integration
Reuse `decode_eval` `ab_script` runner + `north_star_flash_attn_tile_v0` binding; the native candidate flows through
correctness → local A/B → W==D → policy exactly like prior candidates; the llama oracle stays the non-promotable
target/byte-level oracle.

### Stop conditions (project-level)
Any phase gate fails per its stop condition → record a refutation, **fall back to `POST_MATMUL_PV_REST_DECODE_V2`**
(decode capped at tinygrad's backend ceiling; pivot to v2/search). Never ship a slower/lossy path; never change the
default before Phase 4 passes.

### Rollback / no-change boundaries
No `tinygrad/` change in Phase 1 (extra/ µkernel + custom_kernel only). `tinygrad/codegen|renderer` changes only from
Phase 2, behind the gate, with `git diff tinygrad/` reviewed each phase. Any AMDGCN/HSACO escape hatch is default-off,
shape-guarded, with a tinygrad-codegen fallback, and must clear the same W==D + quality gates.

### Final decision enum for that future project (choose one at its end)
`FUSED_FLASH_PROMOTABLE` · `FUSED_FLASH_LOCAL_PASS_WD_FAIL` · `FUSED_FLASH_FAIL_LOCAL_AB` ·
`FUSED_FLASH_BLOCKED_BY_CODEGEN` · `FUSED_FLASH_NEEDS_ASM_ESCAPE_HATCH`.

### What success / failure means
- **Success** (`FUSED_FLASH_PROMOTABLE`): tinygrad beats llama decode under W==D via a native fused-flash primitive →
  pillar 1 of the north star achieved → then v2 cutover consolidates the win.
- **Failure** (any stop): decode is proven capped at tinygrad's current backend ceiling with a concrete reason →
  `REST_DECODE` + v2 is then the *evidence-backed* correct pivot (not a premature one).

---

## Phase 5 — fallback v2 transition sketch (used iff the fused-flash first gate fails)
Per north star §"Clean Repo / v2": new `tinygrad-v2/` execution repo; **keep** core tinygrad+AMD runtime, Qwen3
Q4_K/Q6_K decode + `PREFILL_V2`/`PREFILL_GRAPH_GEMM`/`PREFILL_TC_ATTN`/`PREFILL_CONCRETE_KV`, q8 opt-in, the
lifecycle-search stack (`qk_decode_eval`, `qk_lifecycle_search_loop`, `candidates.json`, `refutations.json`,
`binding_templates.json`), W==D + dNLL/greedy harnesses, and the canonical docs + **decode refutation/oracle archive**;
**drop/archive** one-off probes, superseded scopes, duplicated benchmark scripts, dead q8/Tensile fragments. **Migration
gates:** v2 W==D within noise of the repo, prefill policy checks run, lifecycle-search runs, quality gates pass, old
repo kept as provenance. **No performance promises**; reopen decode only if a new capability (fused-flash / symbolic
tiled) or new timed evidence appears.

## Reopen conditions (decode, after this decision)
- The fused-flash first gate **passes** → fund the full `POST_MATMUL_PV_FULL_FUSED_FLASH` project (Phases 2–5).
- A new tinygrad capability (symbolic-count tiled matmul, or a renderer that emits LDS-tiled fused reductions) lands.
- New timed evidence overturns a closure-table verdict.
- Otherwise decode stays rested; do **not** reopen any closed bounded lane or re-tune flags.

## Acceptance gates
| gate | result |
|---|---|
| G1 all recent decode lanes in the closure table | PASS (Phase 0, 18 rows) |
| G2 all three options analyzed exhaustively | PASS (Phase 2) |
| G3 recommendation names first gate + stop condition | PASS (Phase 3/4) |
| G4 chosen scope executable by a future agent | PASS (Phase 4: files, phases, gates, artifacts, enums) |
| G5 no closed lane reopened | PASS (strategy only; reopen is gated) |
| G6 no implementation/kernel/model/default change | PASS (`git diff tinygrad/` empty) |
| G7 policy guard passes | PASS (run pre-commit) |
| G8 tree clean after commit / unrelated dirty listed | PASS (pre-existing unrelated dirty `structure/.../performance-primitive-research-principles.md` listed) |

## Boundary
Scope/audit only. No `tinygrad/` change, no model/default route, no kernel built, no flag tuned, no closed lane
reopened. The fused-flash recommendation funds a **cheap first gate**, not a blind multi-week project; `REST_DECODE`+v2
is the explicit, evidence-backed fallback. The llama oracle stays non-promotable; refutations stay the search map.
