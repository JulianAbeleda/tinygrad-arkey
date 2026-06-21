# docs/ — map

Single navigation source-of-truth for this fork's docs. The AMD-decode work produced a long
chronological probe log; the **verdicts are folded into the syntheses below** — start there, treat
the dated `*-plan/-result/-probe.md` files as provenance, not current state.

## ⭐ Start here (canonical, post-bank)

- **`current-project-state-handoff-20260621.md`** — ⭐⭐ CANONICAL CURRENT STATE (read first). One short page:
  canonical numbers, decided policies (global `PREFILL_V2` OFF; `auto`/server/q8 opt-in), closed lanes (prefill
  kernels, prefill default, bounded decode fusion, bounded decode vector-tile, the `87.6` ambiguity). **Bounded
  decode work is RESTED** — the only remaining decode lever is the north-star full `flash_attn_tile` lifecycle.
  Guardrail: `extra/qk_policy_consistency_check.py` fails if a canonical doc re-opens these.
- **`decode-evaluation-harness-hardening-result-20260621.md`** — ⭐ MACHINE-SEARCH EVALUATOR BUILT. `extra/qk_decode_eval.py`
  is the automated lifecycle ladder (correctness→local A/B→W==D→policy) emitting schema'd verdicts; it reproduces the
  historical classifications (baseline→REST, flash_l_64→LOCAL_PASS_WD_FAIL, warp_tile→FAIL_LOCAL_AB, q8→PASS_OPT_IN)
  and proved whole-decode W==D auto-clock variance is **<0.6% ≪ 5% margin** → `EVALUATOR_READY_FOR_LIFECYCLE_SEARCH`
  (GPU-state tooling not needed). Measurement-only; no defaults changed.
- **`lifecycle-search-loop-v0-result-20260621.md`** — ⭐ LIFECYCLE-SEARCH LOOP v0 BUILT. `extra/qk_lifecycle_search_loop.py`
  is the first closed `generate→evaluate→prune` loop on the evaluator: runs valid candidates through `decode_eval`
  (4 executed, verdicts match) and **prunes invalid ones before benchmarking** (WMMA-decode reopen, FLASH_L=64
  default-promotion). No kernels, no defaults; propose-only ledger; surfaced + fixed a q8 auto-clock confound.
  `LIFECYCLE_SEARCH_V0_READY`.
- **`candidate-template-generation-v0-result-20260621.md`** — ⭐ TEMPLATE GENERATION v0 BUILT.
  `extra/qk_candidate_template_gen.py` expands 4 templates into 9 legal decode candidate specs (policy metadata,
  deterministic) that flow **through the loop**: 3 executable (verdicts match) + 6 pruned/deferred (closed-lane
  reopens, default-promotion attempts, north-star `flash_attn_tile` deferred). No kernels/flags/defaults.
  `TEMPLATE_GENERATION_V0_READY`.
- **`north-star-evaluator-binding-templates-result-20260621.md`** — ⭐ NORTH-STAR BINDING TEMPLATE BUILT.
  `bench/qk-decode-eval/binding_templates.json` specifies what an executable `flash_attn_tile` candidate must
  declare/run/produce vs `gqa_coop_vec` (comparator, T=1 artifact fields, runners, no-WMMA, gates, stop conditions).
  `gen_north_star_flash_attn_tile` is now a precise `PRUNE_NEEDS_TEMPLATE` (blocked only on a kernel + runners); the
  loop distinguishes missing-template / no-runner / executable; a no-GPU selftest proves the binding path.
  `NORTH_STAR_BINDING_TEMPLATE_READY`.
- **`north-star-flash-attn-tile-execution-result-20260621.md`** — ⭐ FIRST NORTH-STAR ATTEMPT EXECUTED + REFUTED.
  `extra/qk_north_star_flash_attn_tile_ab.py` (warp-cooperative q·k partial + many-wg combine) ran the local A/B vs
  `gqa_coop_vec` → **0.58×@1024 / 0.89×@4096** (byte-exact) → `FAIL_LOCAL_AB`; stopped before W==D.
  `gen_north_star_flash_attn_tile` now EXECUTEs → FAIL_LOCAL_AB → refute_candidate. `NORTH_STAR_FAIL_LOCAL_AB`.
  (Combine-bandwidth claim CORRECTED by the redesign audit below — the combine is NOT the ceiling.)
- **`north-star-decode-attention-redesign-audit-20260621.md`** — ⭐ BOUNDED TILE LEVER EXHAUSTED.
  A throughput probe corrects the diagnosis: the combine is negligible (pout ~1 µs); the "combine cost" was
  2nd-raw-dispatch overhead. The real ceiling is the **cooperative-dot q·k partial** (flat ~163 µs vs coop's
  scaling 75–144 µs); **coop's matmul q·k is near-optimal for tinygrad primitives**. No bounded combine/compact-state
  tile passes @ctx1024; the 10× gap to llama is **codegen quality**. `REDESIGN_AUDIT_POINTS_TO_CODEGEN_DATAFLOW`
  (no build). Do not build another bounded tile.
- **`decode-codegen-dataflow-capability-scope-20260621.md`** — ⭐ CAPABILITY SCOPE: `CODEGEN_SCOPE_LLAMA_ORACLE_FIRST`.
  Native codegen (single fused flash kernel) is a multi-week linearizer project (`spec.py:163-165` single-op REDUCE +
  store-group idiom; pre-refuted `flash_fused_multireduce_linearizer_wall`) with no validated target. The llama audit
  numbers are **in-model only** — llama's kernel was never measured **standalone** vs coop. Next project ports
  llama's `fattn-tile.cuh` (source on disk) as a **non-default reference oracle**, measures standalone throughput vs
  `gqa_coop_vec` (first gate ≥1.05× @ctx1024) via the existing `ab_script` binding, resolving standalone-vs-in-model
  before any codegen surgery.
- **`llama-flash-attn-tile-oracle-result-20260621.md`** — ⭐⭐ CENTRAL QUESTION ANSWERED: `LLAMA_ORACLE_LOCAL_AB_PASS`.
  Pure-GPU-time A/B (llama rocprofv3 trace vs coop tinygrad ProfileGraphEvent): **llama 10.2/12.2/27.7 µs vs coop
  59.9/69.9/132 µs → llama 5.87/5.71/4.77× faster STANDALONE** @ctx512/1024/4096. The 10× decode-attention gap is a
  **standalone kernel-codegen target, not only in-model integration** → native fused-flash codegen is justified, with
  llama's `flash_attn_tile` as the validated target. Profiling oracle (full port BOUNDED, deferred); registered as a
  `reference_oracle` decode_eval candidate (`PASS_ORACLE_LOCAL_AB`, non-promotable).
- **`native-fused-flash-linearizer-scope-20260621.md`** — ⭐⭐ PREMISE CORRECTED: `NATIVE_FLASH_LINEARIZER_SCOPE_READY`.
  An empirical probe **REFUTES the "compiler expressiveness wall"** — a coupled online-softmax+V fused decode kernel
  (`m,l,acc` coupled) **verifies + runs value-correct in ONE kernel TODAY** via the existing `UOp.set`/`.after`
  idiom; no `spec.py`/linearizer change. The 6-kernel split was an idiom pitfall (mirror-slot RAW / GROUP-shape-index
  `ops.py:372` / two-ENDs `linearizer.py:81`), not a wall. So the next step is a **bounded kernel-build** (Path A:
  coop's matmul q·k + ONE fused softmax+V), first gate = value-correct (met) + A/B vs `gqa_coop_vec`. The real 5–6×
  gap is **in-kernel-q·k codegen QUALITY** (deep, deferred). `flash_fused_multireduce_linearizer_wall` refutation
  corrected. No `tinygrad/` change; no llama port first.
- **`fused-softmax-v-tail-candidate-result-20260621.md`** — Path A EXECUTED + REFUTED:
  `FUSED_SOFTMAX_V_TAIL_FAIL_LOCAL_AB`. The achievable fused tail (coop matmul q·k + inline-exp partial, keep
  flash_max) is value-correct but **0.725×@1024 / 0.876×@4096** vs `gqa_coop_vec` — fusing exp into the partial makes
  W=129 lanes recompute exp (vs coop's once/key), so it loses. Tail fusion doesn't help; coop's hoisted-exp split is
  near-optimal. Full online-max removal `BLOCKED_BY_IDIOM` (two-granularity pm+pout store). The 5–6× gap stays
  in-kernel-q·k codegen quality (deep). No W==D; banked.
- **`decode-frontier-decision-after-path-a-20260621.md`** — ⭐ FRONTIER DECISION: `FRONTIER_LOW_LEVEL_TOOLING_FIRST`.
  A per-kernel breakdown corrects the framing: coop's 70µs = `flash_partial` 24.7µs + **matmul q·k 13.9µs (≈ llama's
  whole 12µs tile)** + softmax ~28µs — the **q·k is NOT the bottleneck**; the gap is the softmax+V multi-kernel that
  Path A can't fuse. We lack counter/ISA attribution → deep q·k codegen (A) is mis-targeted, llama port (B) premature.
  Next = **diagnostic-first low-level tooling** (rocprof-compute counters + ISA disasm of `flash_partial` vs llama
  tile) to name the inefficiency, then a targeted codegen fix OR `REST_DECODE` with proof. Bounded decode exhausted.
- **`low-level-decode-attn-attribution-result-20260621.md`** — ⭐ ATTRIBUTION DONE: `LOW_LEVEL_ATTRIBUTION_FIXABLE_CODEGEN`.
  ISA/resource attribution rules OUT occupancy/registers/spills (tinygrad kernels at **100% occupancy, ≤13 VGPR, 0
  spills**) and fundamental limits. Root cause = **codegen quality**: `flash_partial` (PV, 24.7µs) emits scalar fp16
  loads, **0 `v_dot2`, 0 LDS** (latency-bound) vs llama's LDS-tiled `v_dot2` fused tile; the q·k matmul is fast
  because it's tiled (LDS=64). Fixable lever = route the PV through tiled-matmul codegen (~1.16× attention, but
  W==D-marginal; full win = deep LDS-tiled fused-flash codegen). Counters tooling-opaque (rocprof-compute broken,
  rocprofv3 blind to HCQ) but binaries sufficed. Next = scope the matmul-PV diagnostic, else REST_DECODE.
- **`tinygrad-hcq-profiling-visibility-result-20260621.md`** — `HCQ_VISIBILITY_USE_NATIVE_ATTRIBUTION_ONLY`. rocprofv3
  is inherently blind to HCQ (tinygrad writes PM4/AQL to a hardware ring + doorbell, never `hsa_queue_create`, so
  rocprof's HSA interception misses it; reproduced: 0 traces). rocprof-compute fix = unbounded deps + 0-counter
  backend → not worth it; bounded HCQ-counter paths already killed. Native attribution (ISA/resources + ProfileGraphEvent
  + ATT intervals) exists and suffices. Live HCQ counters = a deep native-profiled-HCQ project, deferred (low EV).
- **`matmul-pv-diagnostic-result-20260621.md`** — ⭐ `MATMUL_PV_BLOCKED_BY_LAYOUT` (gate `FAIL_LOCAL_AB`) → REST_DECODE.
  The **split-preserving** tiled matmul PV (K=L=128 concrete, Hkv·Smax=256 wg) **TILES at ~1078 GFLOPS and WINS
  1.13×@ctx4096** — the ISA lever is CONFIRMED on the merits — but loses ctx1024/512 (**0.94/0.88×**) because tinygrad
  cannot express a symbolic-count tiled batched matmul, so the concrete-K form needs concrete Smax=32 → reads the full
  MAXC KV (4–8× extra split work). **Corrects** an earlier non-split form (~50 GFLOPS, parallelism-collapsed) that
  wrongly blamed "skinny M=4." The bounded lever is exhausted (win unreachable Tc-proportionally without a
  symbolic-count tiled matmul / the deep fused-`v_dot2`+LDS single-tile codegen). Decode bounded space exhausted.
- **`post-matmul-pv-decode-strategic-scope-20260621.md`** — ⭐ STRATEGY: `STRATEGY_RECOMMEND_FULL_FUSED_FLASH` → next
  project `POST_MATMUL_PV_FULL_FUSED_FLASH` (gate-first). Exhausts the 3 remaining options (rest+v2 / symbolic-count
  tiled matmul / full fused-flash). Bounded decode is dead; only deep codegen closes the 5–6× llama gap. Symbolic-count
  tiled matmul is **dominated** (W==D-marginal + a sub-capability of fused-flash); rest+v2 is **premature** (the cheap
  fused-flash first gate is untried). **Recommendation: fund the cheap concrete-ctx1024 toy LDS-tiled fused µkernel
  first gate (value-correct + ≥1.05× vs `gqa_coop_vec`), hard-stop → if it fails, fall back to REST_DECODE+v2.**
  Includes the full decode closure table, capability map, and the executable fused-flash scope + v2 fallback sketch.
- **`project-north-star-llama-and-lifecycle-search-20260620.md`** — PROJECT COMPLETION DEFINITION. The project is
  complete only when tinygrad both beats the current llama.cpp decode reference and has a closed lifecycle
  machine-search system that can find/maintain that win, then cuts over into a clean `tinygrad-v2` execution repo.
  The search target is route/fusion/materialization/policy lifecycle, not just single-kernel schedules.
- **`prefill-increment0-shipped-result-20260620.md`** — CURRENT PREFILL STATE. Increment 0 shipped
  `PREFILL_CONCRETE_KV` as an opt-in server/long-prompt path with precompile-at-load. Combined with Branch B,
  concrete prefill holds **73–111% of llama** across KV 512–3584. Rerun: A2 warm prefill `4941 -> 343 ms` when both
  chunks are concrete, and A1 first-generation prefill `9.41 -> 3.44 s` after load-time precompile; byte-identical
  tok0, default path untouched.
- **`prefill-branch-b-tc-attention-result-20260620.md`** — Branch B is promoted default-on under `PREFILL_V2`/gfx1100:
  concrete first-chunk explicit attention reruns at **3394 pp512 tok/s** (`112.7%` llama), byte-identical. The win is
  attention-reduce fusion, not WMMA.
- **`prefill-flash-increment2-result-20260620.md`** — Increment 2 flash prefill kernel is correct (`rel_rmse ~1e-7`)
  but **~15× too slow** at KV512 and not shipped. Recommendation: rest prefill here unless a multi-day WMMA/key-tiled
  flash build is explicitly reopened.
- **`prefill-policy-integration-result-20260620.md`** — ⭐CURRENT PREFILL STATE (policy shipped). Three advertised
  profiles: **universal default** (any card, slow long prompts), **`PREFILL_V2=auto`** (24GB+, ~5–15× faster prefill,
  VRAM-gated so it never OOMs small cards), **`PREFILL_SERVER_PROFILE=1`** (servers/long/repeat, best warm prefill
  0.17–1.6s). Shipped: VRAM-aware `PREFILL_V2=auto` (Phase 1), concrete-KV server policy (Phase 2), the 32-token
  fallback fix `PREFILL_REMAINDER_FIX` (Phase 3, default-on, byte-identical, up to 14× on prefix-cache resume),
  CLI hints (Phase 4). Per-phase: `prefill-{v2-auto-policy,concrete-kv-policy,route-schedule}-result-20260620.md`;
  scope `prefill-policy-integration-scope-20260620.md`; VRAM-reduction design `prefill-v2-vram-reduction-scope-20260620.md`.
  **POLICY (DECIDED 2026-06-21): prefill is kernel-solved and the opt-in fast paths are shipped, but the global
  default `PREFILL_V2` STAYS OFF — do NOT flip to `auto`** (it would keep +14GB fp16 prefill weights resident during
  decode for zero decode benefit; the common decode/short-prompt user must not pay that). Fast path is one flag away
  (`PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1`), and the CLI hints it on large GPUs. **Decode remains the frontier.**
- **`decode-prefill-headline-reconciliation-result-20260621.md`** — ⭐HEADLINE RECONCILED. `87.6` is a NUMERIC
  COINCIDENCE: a real ctx≈0 decode **tok/s** AND, separately, a real ctx4096 decode **ms/token** (=11.4 tok/s); the
  reported "87.6 tok/s" was the genuine ctx≈0 rate (reruns ~85–86), not the ms mislabeled. Clean-wall reruns
  reproduce the canonical table exactly (68.1/66.4/60.7 @512/1024/4096); prefill policy (`auto`/server) does **not**
  regress decode (<1%, identical output). **Decode headline stays `~67% llama` @ctx (≈86% @ctx≈0).** **DECIDED
  2026-06-21: global `PREFILL_V2` default stays OFF — not flipped to `auto`** (it holds +14GB resident during decode
  for zero decode benefit); fast paths stay opt-in. Current-state: `current-project-state-handoff-20260621.md`.
- **`decode-role-tensor-kernel-attribution-solution-scope-20260620.md`** — CURRENT DECODE NEXT SCOPE. Decode remains
  below llama: default route is still the banked `~67%` llama class, while q8 FFN is a hardened default-off opt-in
  route rerun at **72.9/71.1 tok/s @ctx 512/1024** (`~1.064×`, host-sync `0.0%`). Next work is role/tensor/kernel
  attribution, not q8 lifecycle.
- **`decode-fusion-build-result-20260620.md`** — BOUNDED DECODE FUSION CLOSED. A real FFN activation producer-fusion
  kernel was built and byte-exact, but produced `~0%` speedup; the activation cost is work-conserved, not launch
  recoverable. Attention reduce/stat microfusion is a no-go because the dominant costs are intrinsic O(KV) QK /
  softmax work and the real fully fused flash path is linearizer/codegen-walled. Keep current decode defaults:
  baseline `60.8-68.0 tok/s`, q8 opt-in `64.5-72.8`.
- **`decode-latency-hiding-lifecycle-codegen-scope-20260621.md`** — (Claude-1 lane; superseded by the fused-coop
  roadmap below). The decode frontier is latency hiding / larger lifecycle codegen, not micro-fusion. The `87.6`
  headline is RECONCILED (`decode-prefill-headline-reconciliation-result-20260621.md`; decode is the curve / ~67%
  llama); then either prototype a fully fused flash-decode tile, prove a GEMV latency-hiding schedule, or compiler-backlog.
- **`decode-fused-coop-primitive-roadmap-scope-20260621.md`** — DONE (historical). Returned `LINEARIZER_FIRST`,
  which led to the vector-tile build; that build is now rested (see the realigned result below). Not current work.
- **`canonical-policy-handoff-audit-result-20260621.md`** — CLAUDE-2 LANE CLOSED. Hardened the canonical
  policy/headline state: audited the policy commits (no junk; recorded the swept Claude-1 files), swept stale
  references, wrote `current-project-state-handoff-20260621.md`, and added the guardrail
  `extra/qk_policy_consistency_check.py`. Scope: `canonical-policy-handoff-audit-scope-20260621.md`.
- **`llama-decode-primitive-difference-audit-scope-20260621.md`** — DONE (historical). Oracle audit; verdict
  below.
- **`llama-decode-primitive-difference-audit-result-20260621.md`** — DECODE PRINCIPLE CORRECTED. llama decode
  attention is **not WMMA**; the gap is a non-WMMA vector `flash_attn_tile` that wins by many KV-split parallel blocks,
  LDS K/V staging, and GQA query-head column packing. New rule: for decode `T=1`, reuse must not collapse occupancy.
  WMMA and MMVQ stay closed. Canonical rule lives in
  `../structure/Development/performance-primitive-research-principles.md`.
- **`decode-vector-flash-tile-realigned-result-20260621.md`** — ⭐ BOUNDED DECODE VECTOR-TILE RESTED (current
  decode state). Applying the T=1 principle to the existing winner `gqa_coop_vec` (lower `FLASH_L` → more splits)
  passed the standalone gate (`FLASH_L=64` ~1.08× attention @ctx1024, byte-exact) but **failed W==D promotion**
  (+1.8%@1024, −1.2%@4096). New hand-tiles were byte-exact but slower than the matmul q·k. **Decision:
  `REST_DECODE` for bounded work; do not promote `FLASH_L=64` by default.** The only remaining decode lever is the
  north-star full `flash_attn_tile` lifecycle with an efficient many-split / stream-k combine — not a bounded
  patch.

### Historical / Superseded Prefill Provenance

- **`prefill-RECONCILIATION-source-of-truth-20260619.md`** — ⭐PREFILL SOURCE-OF-TRUTH. Settles the contradictory
  prefill results under one controlled interleaved matrix. Verdict: concrete-KV = 1.24x byte-identical (shipped,
  ~47% llama); **+Tensile (external .co, research) = 1.76x over concrete = ~86% llama, REPRODUCED** (the old
  "4770/1.76x" is REAL; the "0.997x no-advantage" runs were a high-WMMA-clock outlier — tinygrad WMMA prefill is
  clock-volatile 1449-2675, Tensile is clock-stable ~2640). Supersedes prefill-matmul-RECONCILED / tensile-land /
  transpose-free "Tensile-no-advantage". Artifact: `artifacts/prefill-reconciliation-matrix-20260619.json`.
- **`prefill-occupancy-lever-result-20260619.md`** — ⭐⭐CANONICAL prefill verdict — RESOLVES THE WHOLE ARC. The P0
  kernel-identity gate (`extra/qk_prefill_kernel_identity.py`) found the "WMMA bimodality" was a **flag-leak BUG** in
  `qk_tensile_ab_measure.py`: `TinyJit` captures on the 2nd call, and the harness left `PREFILL_TENSILE_GEMM=True`
  (from `build(True)`) during `joff`'s capture → the "OFF"/WMMA arm silently routed **Tensile** (3 `tensile_*`
  kernels found in its graph). **TRUTH: WMMA prefill = ~1433 (~47% llama) CONSISTENT (no lottery); Tensile = ~2673
  (~87% llama) CONSISTENT, byte-identical (rel_err 0) = a REAL 1.84× win.** The original reconciliation (1.83×) was
  right; the `0.997× no-win` and every "fast 2674 WMMA" were leaked-Tensile. **Prefill decision (clean): accept the
  vendored rocBLAS Tensile `.co` dependency → 87% llama byte-identical, OR rest dependency-free at WMMA ~47% (POWN
  ~42 TFLOPS codegen wall) + shipped concrete-KV 1.24×.** No occupancy/clock/boost lever exists or is needed. Fixed
  the harness; production unaffected (env flag set once, never toggled).
- **`prefill-primitive-pmc-result-20260619.md`** — ⭐WHY Tensile prefill is faster, DISASM + PMC MEASURED. Both paths
  use RDNA3 WMMA (corrects the old "Tensile = FMA-only, no WMMA" claim: the in-model gateup kernel
  `MT128x128x16_MI16x16x16` = 80 `v_wmma` + 256 `v_fma_mix`). Same gateup GEMM, measured per-primitive: LDS staging →
  WMMA reads DRAM **6.6× more** (`GL2C_MC_RDREQ` 29.8M vs 4.5M; WMMA LDS=0 vs Tensile 24.5KB); stalls → WMMA **6.5×
  more** `SQ_WAIT_ANY` (Tensile prefetch-overlaps); occupancy 1 vs 32 waves/wg; **VALU overhead ~EQUAL (NOT the
  cause)**. Verdict: the gap is **memory dataflow** (LDS staging + the stalls it removes), not codegen-VALU or the
  matrix op. PMC captures the vendored Tensile kernel ✅; no WMMA-busy counter (TFLOPS-only). Driver
  `extra/qk_prefill_primitive_pmc.py`; scope `prefill-primitive-pmc-scope-20260619.md`.
- **⚠ RETRACTED as the-same-bug** (kept for provenance/methodology only — NOT current state):
  `prefill-boost-resolution-{scope,result}-20260619.md` ("bimodal/boost/occupancy lottery"),
  `prefill-clock-dpm-authority-{scope,result}-20260619.md` ("clock authority / not-clock-its-occupancy"),
  `prefill-occupancy-lever-scope-20260619.md` (the scope whose premise P0 invalidated). All three chased artifacts of
  the `ab_measure` flag-leak. Methodology bankable: a TinyJit A/B toggling a routing global must capture each jit
  before changing it + assert on the captured graph's kernels; real GFXCLK via separate-process `--showgpuclocks`.
- **`decode-native-tooling-readiness-scope-20260619.md`** — SCOPE for the missing decode-native tooling gate before
  scheduler/renderer implementation. Current state: ATT/HCQ visibility exists and timing policy exists, but no
  timing-grade `>=30us` feature attribution exists for native q8/MMVQ. Defines the readiness artifacts,
  authority labels, q8 `ffn_gate/up` role-join gap, bucket-classification requirements, and exact start criteria for
  N2/native backend work.
- **`decode-native-tooling-readiness-result-20260619.md`** — DTR-0 executed. Adds
  `extra/qk_decode_native_tooling_readiness.py` and `bench/qk-decode-native-tooling/*`. Verdict:
  `TOOLING_NOT_READY`: N2 candidate count `0`, max attributed movement `14.087us`, and missing rows are q8
  `ffn_gate/up` role-joined body evidence, `>=30us` timing-grade feature attribution, counter/timing feature join,
  and bucket classification for q8 `ffn_gate/up`.
- **`decode-native-tooling-completion-scope-20260619.md`** — execution scope for the remaining tooling phases
  DTR-1..DTR-4. Immediate next step: extend `extra/qk_att_inmodel_role_join.py` to capture `ffn_gate`, `ffn_up`, and
  `ffn_gateup_pair`; then build a feature joiner and ablation matrix before any scheduler/renderer implementation.
- **`decode-native-tooling-completion-result-20260619.md`** — DTR-1..DTR-3 executed. `ffn_gate` and `ffn_up`
  now have in-model ATT body attribution (`q4k_gemv_partial_12288_4096_1`, ~47k body packets each), clearing the
  missing role-visibility row. DTR-2 feature join and DTR-3 ablation matrix still find `0` N2 candidates; max
  timing-grade movement remains `14.087us`. Verdict: `TOOLING_NOT_READY_FOR_N2`; remaining blocker is
  counter/timing-grade scheduler-resource attribution for the `73.109us` q8 native-to-oracle gap.
- **`decode-native-tooling-pass-scope-20260619.md`** — exhaustive scope for making the tooling verdict pass. Defines
  every acceptable route to `TOOLING_READY_FOR_N2`, `ROADMAP_ONLY`, or `BROAD_BACKEND_ACCEPTED`: PMC blob decoding,
  SQTT/timeline attribution or formal decoder blockage, same-binary timing joins, scheduler/resource ablation rows,
  W==D projection, and exact N2 start conditions. Current best guess is `ROADMAP_ONLY`, but the P1-P4 tooling rows must
  prove it.
- **`decode-native-tooling-pass-result-20260619.md`** — pass list completed. Adds the P1-P5 tooling scripts/artifacts
  and updates readiness to final `ROADMAP_ONLY`: q8 `ffn_gate/up` role evidence exists, no N2 candidate exists, max
  isolated movement remains `14.087us`, PMC/SQTT are blocked as counter/timeline attribution from current artifacts,
  and no native W==D projection is justified. Native q8 scheduler/renderer is roadmap/broad-backend work only.
- **`amd-broad-backend-roadmap-scope-20260619.md`** — roadmap scope for the only remaining native AMD path after the
  decode tooling pass: explicit `BROAD_BACKEND_ACCEPTED` or stay `ROADMAP_ONLY`. Defines the reusable backend
  capabilities, tracks, phases, gates, artifacts, and stop conditions for a scheduler/resource project spanning q8
  decode and prefill WMMA/Tensile, while disallowing a q8-only native patch.
- **`amd-broad-backend-roadmap-result-20260619.md`** — BB-0/BB-1 executed after accepting the broad backend roadmap.
  Adds `extra/qk_amd_broad_backend_roadmap.py`, `extra/qk_amd_schedule_metadata_probe.py`,
  `extra/qk_amd_wait_resource_probe.py`, `extra/qk_amd_software_pipeline_probe.py`,
  `extra/qk_amd_bb5a_renderer_allocator_scope.py`, `extra/qk_amd_bb5a1_pipeline_ir_scope.py`,
  `extra/qk_amd_bb5a1_pipeline_ir_probe.py`, `extra/qk_amd_bb5a_full_plan.py`,
  `extra/qk_amd_bb5a_execute_plan.py`, `extra/qk_amd_bb5a2_solution_scope.py`,
  `extra/qk_amd_bb5a2_lds_stage_plan_probe.py`, `extra/qk_amd_bb5a2_lowering_hook_probe.py`,
  `extra/qk_amd_bb5a2_render_isa_evidence_probe.py`, `extra/qk_amd_bb5a2_real_lowering_integration_probe.py`,
  `extra/qk_amd_bb5a2_pipelined_dataflow_probe.py`, `extra/qk_amd_bb5a3_wait_scheduler_integration_probe.py`,
  `extra/qk_amd_bb5a4_allocator_resource_probe.py`, `extra/qk_amd_bb5a5_resource_policy_probe.py`,
  `extra/qk_amd_bb5a6_correctness_probe.py`, `extra/qk_amd_bb5a7_performance_gate_probe.py`,
  `extra/qk_amd_bb5a8_tensile_mapping_probe.py`, `extra/qk_amd_bb5a8_authority_kernel_capture_probe.py`,
  `extra/qk_amd_bb5a9_causal_delta_package.py`,
  `tinygrad/renderer/amd/schedule.py`, `tinygrad/renderer/amd/elf.py`, and
  `bench/amd-broad-backend-roadmap/*`. Verdict:
  `BROAD_BACKEND_ACCEPTED_BB5A9_CAUSAL_DELTA_DONE_PARALLEL_IMPLEMENTATION_READY_Q8_BLOCKED`:
  authority/oracle suite, schedule metadata IR, semantic wait-scheduler probe, and resource accounting probe pass;
  BB-5 formally blocks because software-pipelined prefill still needs real renderer/allocator integration; BB-5a now
  scopes that missing layer; BB-5a.1 passes as a read-only two-stage pipeline IR surface; the full BB-5a plan was
  executed through the roadblock sequence: BB-5a.2 double-buffered LDS lowering passes with gated source/ELF evidence,
  BB-5a.3 wait scheduler integration passes, BB-5a.4 resource control passes, BB-5a.5 policy passes, and BB-5a.6
  correctness passes. BB-5a.7 blocks because pure tinygrad authority prefill is `42.0 TFLOPS`, below the `60.0 TFLOPS`
  gate. BB-5a.8 completes the static Tensile-to-tinygrad mapping and captures the timing-equivalent tinygrad authority
  kernel (`43.026 TFLOPS`, `64` `v_wmma`, `0` LDS bytes, `0` `ds_load_b128`) as source/ELF/disassembly/resource
  evidence. BB-5a.9 proves the causal deltas and makes LDS layout, K-loop scheduling, and resource policy parallel
  implementation tracks. Q8 transfer remains disallowed.
- **`amd-broad-backend-bb5a9-causal-delta-package-20260619.md`** — BB-5a.9 handoff. Root cause now proven at
  same-kernel level: captured tinygrad authority uses WMMA but no LDS staging, while Tensile uses WMMA plus
  LDS-staged wide reads/stores, explicit prefetch, and wait/barrier scheduling. Defines parallel tracks A-F and the
  BB-5a.10 implementation backlog.
- **`amd-lds-research-consolidation-20260619.md`** — LDS/WMMA consolidation checkpoint before BB-5a.10. Reconciles
  old LDS refutations with the current Tensile-class renderer target. Closed: plain LDS tiling, multi-wave hand-LDS
  tuning, waves/tile/BLOCK_K/no-LDS sweeps, and manual UOp prefetch. Open only: renderer-level staged LDS layout,
  `ds_load_b128`, software-pipelined K-loop, semantic waits/barriers, and resource policy.
- **`amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md`** — BB-5a.10 focused Tensile layout audit. Isolates
  the selected rocBLAS MT128 authority function from `/tmp/td_all.txt` and proves candidate-spec readiness, not
  bit-exact Tensile layout readiness. Correction: this selected function uses `ds_store_b64` for LDS writes and
  `ds_load_b128` for WMMA operand reads; require selected-kernel-compatible LDS stores, not `ds_store_b128`, for the
  first pure-tinygrad candidate.
- **`amd-broad-backend-bb5a10-implementation-plan-20260619.md`** — BB-5a.10 full phase list. P0 is complete; P1-P5
  run as one coordinated implementation batch over LDS layout, renderer lowering, K-loop staging, waits/barriers, and
  resource policy. P6/P7/P8 are structural/correctness/performance gates. P9 keeps q8 transfer blocked until P8 passes.
- **`amd-broad-backend-bb5a10-p1-layout-spec-20260619.md`** — BB-5a.10 P1 result. Derives the first non-bitexact
  selected-layout spec from the audit: two logical LDS regions, selected-kernel-compatible stores (`ds_store_b64`
  accepted), `ds_load_b128` reads, WMMA handoff requirement, dependency metadata, and spill rejection. Next is P2-P5.
- **`amd-broad-backend-bb5a10-p2-p6-structural-result-20260619.md`** — BB-5a.10 P2-P6 result. Structural ISA/ELF
  candidate passes renderer LDS store/read, K-loop stage order, wait/barrier schedule, resource policy, and combined
  structural gate. This is not correctness or performance yet; P7 executable correctness is now the frontier.
- **`amd-broad-backend-bb5a10-p7-correctness-scope-20260619.md`** — BB-5a.10 P7 scope. Splits executable correctness
  into P7a hardware LDS-WMMA smoke, P7b executable wrapper, P7c small numeric correctness, P7d authority-shape
  correctness, and P7e P8 handoff. Key blocker: P6 has no output store and is not a complete K-loop matmul yet.
- **`amd-broad-backend-bb5a10-p7a-p7c-correctness-result-20260619.md`** — BB-5a.10 P7a-P7c result. Known-good
  LDS-WMMA smoke passes, the structural candidate has an executable wrapper with output, and a selected-compatible
  `ds_store_b64 -> ds_load_b128 -> WMMA` small tile is numerically correct (`RMSE 0.000209`). Next is P7d.
- **`amd-broad-backend-bb5a10-p7d-authority-correctness-result-20260619.md`** — BB-5a.10 P7d result. The selected-compatible
  `ds_store_b64 -> ds_load_b128 -> WMMA` path passes a full authority-K subset (`16x16x4096`, RMSE `0.0001915`).
  Next is P7e P8 handoff packaging; P8 timing remains blocked until the handoff exists.
- **`amd-broad-backend-bb5a10-p7e-p8-handoff-result-20260619.md`** — BB-5a.10 P7e result. Packages the P7d
  correctness/source/resource metadata and exact P8 command. No performance claim.
- **`amd-broad-backend-bb5a10-p8-blocked-result-20260619.md`** — BB-5a.10 P8 result. P8 correctly blocks because
  full authority `M=512,N=12288,K=4096` launch mapping is not implemented yet; do not time the single-tile smoke.
- **`amd-broad-backend-bb5a10-p8-tta-scope-20260619.md`** — BB-5a.10 P8 TTA scope. Defines tile-to-authority
  launch mapping: TTA1 `16x16` full-grid correctness bridge, TTA2 sampled authority correctness, TTA3 `128x128`
  macro-tile performance candidate, TTA4 P8 timing gate. Next is TTA1.
- **`amd-broad-backend-bb5a10-p8-tta-completion-scope-20260620.md`** — BB-5a.10 P8 TTA completion scope.
  Freezes the full address formulas, artifact list, blocked continuations, and P8/P9 ordering through completion.
  Next remains TTA1 implementation.
- **`amd-broad-backend-bb5a10-p8-tta1-full-grid-correctness-result-20260620.md`** — BB-5a.10 P8 TTA1 result.
  Full authority grid `(768,32,1)` over `16x16x4096` sampled tiles passes (`max RMSE 0.0002276`). This is a
  correctness bridge only; next is TTA2 sampled authority correctness.
- **`amd-broad-backend-bb5a10-p8-tta2-authority-sample-correctness-result-20260620.md`** — BB-5a.10 P8 TTA2
  result. Full authority launch sampled first/middle/last row and column tiles passes (`max RMSE 0.0002276`);
  next is TTA3 `128x128` macro-tile candidate.
- **`amd-broad-backend-bb5a10-p8-tta3-macro-candidate-result-20260620.md`** — BB-5a.10 P8 TTA3 result.
  The `128x128` macro helper has the right grid `(96,4,1)`, WMMA, `ds_load_b128`, and scratch/private `0`,
  but blocks because it uses `ds_store_b128` instead of selected-compatible `ds_store_b64`. Next is TTA3a.
- **`amd-broad-backend-bb5a10-p8-tta3a-ds64-macro-conversion-result-20260620.md`** — BB-5a.10 P8 TTA3a
  result. Converts the `128x128` macro helper from `4` `ds_store_b128` stores to `8` selected-compatible
  `ds_store_b64` stores, repatches the K-loop branch, and TTA3 then passes. Next is P8 timing.
- **`amd-broad-backend-bb5a10-p8-performance-result-20260620.md`** — BB-5a.10 P8 result. The converted
  `128x128` DS64 macro candidate is sampled-correct and scratch/private free, but reaches only `18.38 TFLOPS`
  best versus the `60 TFLOPS` gate. P9/q8 transfer remains blocked.
- **`amd-broad-backend-bb5a10-p8-bottleneck-classification-result-20260620.md`** — BB-5a.10 P8 bottleneck
  classification. Original B128 macro (`21.47 TFLOPS`) and converted DS64 macro (`20.93 TFLOPS`) are both far
  below gate; DS64 is not the root cause. The bottleneck is the LDS-staged family itself. Next is global-direct /
  IC-served WMMA candidate decision; q8 remains blocked.
- **`amd-broad-backend-bb5a10-p8-global-direct-candidate-decision-result-20260620.md`** — BB-5a.10 P8 global-direct
  decision. Existing no-LDS candidates are correct but best is only `17.88 TFLOPS`; do not reopen q8. Next is
  P8 timing-authority reconciliation against the prior `~43 TFLOPS` global-direct artifact.
- **`amd-broad-backend-bb5a10-p8-timing-authority-reconciliation-result-20260620.md`** — BB-5a.10 P8 timing-authority
  reconciliation. The prior `43.026 TFLOPS` captured authority kernel remains valid for that kernel only; it does
  not validate current P8 hand-ASM candidates because kernel identity and timing harness differ. Current P8 authority
  remains the synchronized custom-kernel harness. Next is a same-harness authority timing bridge; q8 remains blocked.
- **`tensile-primitive-transfer-matrix-scope-20260620.md`** — PTM-0 scope for decomposing Tensile into primitive
  transfer rows instead of looping on "Tensile" as one object. Uses official Tensile docs plus local artifacts to
  split problem form, tile/WMMA, vector reads, LDS, K-loop prefetch, waits, resource policy, library logic, and timing
  authority. Standalone LDS is closed; next is the same-harness authority bridge.
- **`tensile-roadmap-scope-20260620.md`** — full Tensile roadmap built on PTM-0. Splits the work into three tracks
  (Explanation: why the selected `MT128x128x16` kernel works + two cross-file corrections — the 43 TFLOPS authority
  is tinygrad global-direct NOT Tensile-LDS, and per-body v_wmma is 80 not 13810; Prefill Transfer: native vs
  external `.co`; Decode Applicability: q8 gates) and phases PTM-1..PTM-5. Encodes the stop rules (no standalone LDS,
  no mixed-harness TFLOPS, name the row). Artifact `bench/qk-tensile-primitive-transfer/roadmap.json`. Next is PTM-1
  same-harness authority bridge.
- **`llama-relative-promotion-reconciliation-20260620.md`** — promotion reconciliation across prefill, decode, and
  attention relative to llama. Keeps decode defaults and concrete-KV prefill banked, marks external Tensile prefill as
  the only immediate policy-ready large promotion, keeps q8 artifact as research-only, and blocks native q8/Tensile
  transfer until their explicit gates pass.
- **`q8-ffn-artifact-promotion-scope-20260620.md`** — promotion scope for taking the q8 FFN handwritten/artifact
  route from default-off research to default candidate. Defines Q8P-1..Q8P-6: quality breadth, default safety,
  coverage, W==D performance, artifact ownership, and lossy model-policy decision. First gate is multi-window quality.
- **`q8-ffn-artifact-promotion-result-20260620.md`** — Q8P-1..Q8P-6 executed. The q8 FFN route passes multi-window
  quality (`max dNLL 0.002225`), default-safety, coverage, W==D performance (`1.0507x` min), artifact ownership, and
  model-policy gates. Promoted from research-only evidence to hardened opt-in behind `Q8_FFN_HANDWRITTEN=1`; default
  remains off.

- **`amd-decode-banked-20260616.md`** — THE entry point. Final decode state (~64 tok/s / 63% llama),
  the full lever map (shipped / tapped / refuted / gated), the machine-search system, resume pointers.
- `amd-decode-beyond-llama-roadmap.md` — the lever map with live statuses (parity vs beyond-llama).
- `gpu-performance-first-principles.md` — **canonical** bytes/math/overhead + roofline reference;
  diagnose the bucket BEFORE optimizing.
- **`../bench/README.md`** — the benchmark results index: every current number, its artifact, and the
  exact command to reproduce it. **Includes "Which harness for decode tok/s — READ FIRST"** (use the clean
  `model.generate`-path CLI/W==D harnesses; the flash auto-bench's ~54 is contaminated, not a tok/s number).
- `qk-decode-banked-reproduce-20260618.md` — banked decode line reproduced on HEAD (68.2/66.4/60.7, W==D,
  host-sync 0%, whole stack default-on) + the harness lesson.
- `amd-decode-capstone.md` — the decode ledger (23 → ~64 tok/s arc).
- `amd-decode-arc-synthesis.md` — synthesis through the primitive lens.

## 8B decode-attention + MMVQ frontier (2026-06-17 → 18) — latest state

The work after the decode bank. Closeouts/results are canonical; the many dated `qk-*` arc docs are provenance.

- **`what-makes-a-performance-primitive-efficient-20260618.md` — READ THIS FIRST for the performance-primitive model and gap.**
  Consolidated source of truth for what makes a performance primitive efficient, using llama.cpp vs tinygrad as the
  case study: decode, lm_head, MMVQ, attention, spec, prefill, machine-search lessons, and every remaining path
  marked shipped/refuted/deferred/open.
- `performance-primitive-external-research-audit-20260619.md` — second-round external research audit across
  arXiv/OpenReview/ChinaXiv. Cites each paper/source, records the claim, checks whether it is true/applicable to this
  tinygrad gfx1100 project, and maps it to current or future primitive rows.
- `primitive-local-observability-search-scope-20260619.md` — scope for building primitive-local tooling instead of a
  generic profiler: read-only ledger first, then schema validators, runner wrappers, deterministic failure
  classifiers, guided search memory, and optional rocprof/SQTT counter plugins.
- `primitive-local-observability-search-result-20260619.md` — **executed PLO-1..PLO-6.** Adds
  `extra/qk_primitive_ledger.py`, a read-only primitive ledger/validator/classifier/search-memory/trace-plugin
  inventory that reconstructs current verdicts from existing artifacts without hardware execution.
- `primitive-local-observability-audit-20260619.md` — replay audit over the primitive ledger, including the TPE-7a
  rebindable-node artifact. Confirms graph-protocol prerequisite PASS while keeping in-model capture and artifact
  policy as remaining gates.
- `primitive-ledger-analysis-audit-20260619.md` — uses the primitive ledger for the intended analysis pass: decode is
  q8/MMVQ lifecycle-limited; prefill is graph/artifact-boundary-limited; broad kernel search is not supported by the
  current evidence.
- `primitive-lifecycle-search-scope-20260619.md` — scope + executed PLS-1..PLS-4 ledger for lifecycle-level search:
  producer placement, activation/weight format, consumer primitive, routing boundary, quality gate, fallback, runner
  bindings, policy candidates, generator, and refutation memory. Adds `extra/qk_lifecycle_search.py` and
  `bench/qk-lifecycle-search/*`; current frontier is q8 decode artifact/native transfer and Tensile prefill
  artifact/native transfer, not broad kernel search.
- `primitive-coverage-gap-scope-20260619.md` — coverage audit scope after the latest decode/prefill integration
  learning. Names rows missing from the map, not necessarily missing implementations: decode B2 runtime/cache identity,
  decode MMVQ artifact/import, prefill transpose-free layout, long-context KV/attention, serving, alternative quant,
  CUDA portability, and tooling visibility.
- `primitive-coverage-map-20260619.md` — **executed PCG-0.** Consolidates the current row map into
  `bench/qk-primitive-coverage/rows.json` with 12 validated rows. Key update: Tensile prefill is refuted as an e2e
  speed route after transpose-free `0.997x`; prefill now points to non-matmul overhead, while decode B2 is closed and
  the remaining decode choice is large project-level MMVQ contract work or the small q8 research flag.
- `decode-large-small-paths-scope-20260619.md` — split decode closeout into the large parity-scale path and small
  research path. Large path: MMVQ contract preservation/source import, `~1.187x` measured target but project-level.
  Small path: q8 FFN artifact route, `1.051-1.063x` and dNLL `+0.002887`, default-off research flag.
- `decode-mmvq-artifact-import-discovery-result-20260619.md` — large-path L1 inventory. llama.cpp has MMVQ source and
  build objects, but no standalone Tensile-like HCQ code-object family; direct TPE-style decode extraction is closed as
  a bounded route.
- `decode-mmvq-large-project-scope-20260619.md` — funded large decode MMVQ project scope. Splits the work into
  source/object import first and native renderer/scheduler transfer second, with P0-P8 gates. Target remains
  `44% -> 54%` in-model HBM over the weight-GEMV bucket, about `1.187x` decode.
- `decode-mmvq-large-project-p0-contract-inventory-result-20260619.md` — **executed P0.** The llama.cpp gfx1100
  `mmvq.cu` object contains `22` Q4_K/Q6_K candidate functions and `22` `.kd` descriptors with `144` byte kernargs.
  Next gate is P1: named-descriptor HCQ load smoke, no HIP runtime, no launch yet.
- `decode-mmvq-large-project-p1-loader-smoke-result-20260619.md` — **executed P1.** Selected Q4_K and Q6_K low-VGPR
  llama descriptors load through tinygrad HCQ (`0x74840`, `0x74e40`), no unsupported relocations, no HIP runtime, no
  kernel launch. Next gate is P2: capture real llama kernargs/grid/local in a separate HIP-only process.
- `decode-mmvq-large-project-p2-kernarg-capture-result-20260619.md` — **executed P2.** Versioned LD_PRELOAD capture
  over llama-bench records `7` real Q4_K/Q6_K `mul_mat_vec_q` launches and reconstructs the `144` byte kernargs.
- `decode-mmvq-large-project-p3-p4-q4-result-20260619.md` — **executed P3/P4 for Q4_K.** Imported llama Q4_K MMVQ is
  correct on `blk.0.attn_output.weight` (`max_abs 1.43e-6`) and reaches `903.9 GB/s` / `94.2%` HBM with single-submit
  HCQ timing. Next gate is P5: q8_1 producer/reuse plus one-role in-model routing.
- `decode-mmvq-large-project-p5-p6-result-20260619.md` — **executed P5/P6 for Q4_K.** Real activation -> q8 producer
  -> imported Q4 consumer is correct and clears the lifecycle device gate (`50.8%` HBM-equivalent vs current
  `attn_q/o` ~`29%`), and the same Q4 template generalizes to `ffn_gate/up`. Next gate is graph-safe Q4 routing; Q6
  remains a parallel coverage track.
- `decode-mmvq-large-project-p7a-graph-route-result-20260619.md` — **attempted P7a.** Runtime-cache graph adapter was
  built, but TinyJit replay faults even with persistent side buffers. Imported Q4 remains valid in eager HCQ; graph use
  now requires first-class raw-kernarg rebind support or native lowering. **Superseded by P7b.**
- `decode-mmvq-large-project-p7b-raw-kernarg-rebind-scope-20260619.md` — P7b scope for making imported raw kernargs
  graph-safe: raw template + declared pointer patches, staged through CPU-side args-buffer proof, eager parity, graph
  micro-smoke, real activation graph proof, then one-block route decision.
- `decode-mmvq-large-project-p7b-raw-kernarg-rebind-result-20260619.md` — **executed P7b.** Raw-kernarg rebind support
  passes: offsets `0/8/56` bind q4/q8/out VAs, eager parity is `max_abs 1.43e-6`, and TinyJit replay of real
  `blk.0.attn_output` activation is stable for `5/5` calls with zero diff vs eager.
- `decode-mmvq-large-project-p7c-one-role-route-scope-20260619.md` — P7c scope for moving the imported Q4 route from
  probe-only to one real model role behind `DECODE_MMVQ_IMPORT_Q4=1`, with persistent q8/out side buffers and no default
  behavior change.
- `decode-mmvq-large-project-p7c-one-role-route-result-20260619.md` — **executed P7c.** `blk.0.attn_output` routes
  through the imported Q4_K path in `model.py`; smoke output shape is `[1,1,4096]`, routed blocks `[0]`. Next gate is
  clock-controlled one-role timing, then q8 quality/dNLL.
- `decode-mmvq-large-project-p7d-one-role-timing-scope-20260619.md` — P7d scope for timing the imported route on the
  true pre-`attn_output` activation with same-process interleaved TinyJit A/B.
- `decode-mmvq-large-project-p7d-one-role-timing-result-20260619.md` — **executed P7d.** Imported route is correct,
  replay-stable, and model-branch reachable, but slower for `blk.0.attn_output`: `0.1396ms` vs baseline `0.1064ms`
  (`0.763x`). Do not expand `attn_output`; next valid diagnostic is `ffn_gate/up` q8 amortization.
- `decode-mmvq-large-project-p7e-gateup-amortization-scope-20260619.md` — P7e scope for the fresh favorable Q4 case:
  `ffn_gate/up`, `12288` rows each, one q8 producer shared by two imported Q4 consumers.
- `decode-mmvq-large-project-p7e-gateup-amortization-result-20260619.md` — **executed P7e.** Imported route remains
  replay-stable but loses for `ffn_gate/up`: `0.2264ms` vs baseline `0.1685ms` (`0.744x`). Imported Q4 decode route is
  closed as a local timing win; remaining value is oracle/native-transfer evidence, not model-wide artifact routing.
- `decode-mmvq-large-project-p8-fused-lifecycle-scope-20260619.md` — P8 scope for the full 1-4 sequence after P7e:
  lower-bound model, current native expressibility, handwritten prototype evidence, and final decision.
- `decode-mmvq-large-project-p8-fused-lifecycle-result-20260619.md` — **executed P8.** Fused q8+gate/up is
  build-worthy by lower bound (`56.83us` vs `153.22us` gate); current native COMGR/DSL attempts fail; the hipcc/LLD
  artifact route clears local gate (`115.24us`, `1.46x`) and graph route passes. Decision: artifact research flag or
  project-level native renderer transfer.
- `decode-q8-two-lane-scope-20260619.md` — post-P8 two-lane closeout scope: harden the default-off q8 artifact
  research flag and separately define the native renderer/scheduler transfer start gate.
- `decode-q8-two-lane-result-20260619.md` — **executed two-lane closeout.** Artifact lane is ready as
  `Q8_FFN_HANDWRITTEN=1` research flag (`1.051-1.063x`, dNLL `+0.002887`, no HIP runtime); native lane is project-level,
  with no bounded `>=30us` q8-specific patch identified.
- `decode-q8-both-lanes-execution-scope-20260619.md` — execution scope for the "do both" decision: accept the q8
  artifact dependency for research-flag use and charter the native AMD scheduler project separately.
- `decode-q8-both-lanes-execution-result-20260619.md` — **executed both lanes.** `Q8_FFN_HANDWRITTEN=1` is accepted as
  research-only/default-off; native transfer is chartered as N0-N4 project-level backend work with a `>=30us` start gate.
- `decode-next12-execution-result-20260619.md` — **executed high-level decode next steps 1-2.** The q8 artifact route is
  the completed research answer; native scheduler work is active as a project charter with N0 complete and N1 now closed
  with no bounded N2 start.
- `decode-n1-attribution-scope-20260619.md` / `decode-n1-attribution-result-20260619.md` — native q8 scheduler
  attribution. Full oracle gap is `73.109us`, but largest bounded attribution is `14.087us`; SQTT capture works but
  local RDNA3 HCQ decode fails, so remaining scheduler/resource movement is project-level tooling/backend work.
- `amd-scheduler-tooling-backend-project-scope-20260619.md` — concrete scope for that project-level fork. Track T
  funds RDNA3 HCQ attribution tooling first; Track B funds the reusable AMD scheduler/resource backend only after a
  measurable feature or explicit backend investment decision.
- `amd-scheduler-tooling-backend-t0t4-b0-result-20260619.md` — first combined execution. T0/T2/T3 pass, SQTT replay is
  structurally decodable but maps only `S_ENDPGM` and no q8 body instructions, so T4 finds no bounded feature; B0 oracle
  suite passes with q8 and Tensile targets.
- `amd-scheduler-tooling-t1-body-mapping-proof-20260619.md` — focused T1 proof. Sweeps baseline, `SQTT_MODE=3`,
  `SQTT_TTRACE_EXEC=1`, and both; all capture q8 wave lifecycle packets but `0` raw body instruction packets, so the
  local register-knob fix is refuted.
- `amd-scheduler-tooling-t1b-att-aqlprofile-result-20260619.md` — **executed both requested T1b paths.** ROCprofiler SDK
  and AQLprofile are installed under `/opt/rocm-7.2.4`; external `rocprofv3 --att` remains blocked by the missing/unstable
  ATT decoder path, while AQLprofile command recovery passes. Transplanting recovered `MASK/TOKEN/CTRL` values into HCQ
  changes trace volume but still yields `0` body instruction packets, so the missing piece is a broader command-sequence
  or ROCprofiler-service detail, not a simple register value.
- `amd-scheduler-tooling-t1c-att-decoder-repair-result-20260619.md` — **executed local ATT decoder repair.** Inspected
  available ROCm packages and tested decoder aliases (`librocprofiler-sdk.so`, legacy `libatt_plugin.so`); no candidate
  produced `rocprofv3 --att` output. External ATT is a ROCm packaging/toolchain blocker until a real
  `librocprof-trace-decoder.so` is installed or built.
- `amd-att-decoder-blocker-scope-20260619.md` — concrete reopen scope for the known ATT blocker: binary decoder
  acquisition, source build from `ROCm/rocm-systems`, known-good ROCm environment, and why an ABI shim is not the first
  path. Gates require `rocprofv3 --att` payloads before returning to tinygrad HCQ SQTT body attribution.
- `amd-att-decoder-solution-result-20260619.md` — **executed D0/D1 and solved the external ATT oracle blocker.**
  ROCprof Trace Decoder `0.1.6` binary passes once HIP controls are compiled/linked coherently against ROCm 7.2 instead
  of Ubuntu HIP 5.7. `rocprofv3 --att` now emits `.att`, decoded UI files, wave JSON, and result JSON for the HIP control.
- `amd-sqtt-oracle-hcq-diff-scope-20260619.md` — next scoped tooling phase after the decoder pass: archive the working
  ROCprofiler ATT oracle, reproduce tinygrad HCQ lifecycle-only SQTT, diff setup/order/targeting, try one env-gated
  command-sequence patch if a bounded delta is found, and close with body-attribution pass/kill.
- `amd-sqtt-oracle-hcq-diff-result-20260619.md` — **executed O0-O5; verdict `KILL_PATCH_NO_BODY`.** ROCprofiler ATT is
  valid and instruction-rich (`110446` decoded wave instruction records), but the only bounded HCQ patch
  (`SQTT_ORACLE_TARGET_CU=1`, with/without AQLprofile raw regs) still produced zero body packets. Track T is closed as a
  small primitive-observability patch; reopening requires broader ROCprofiler command-service integration.
- `amd-rocprofiler-thread-trace-audit-result-20260619.md` — source audit of what that broader integration actually
  means. Verdict: ROCprofiler ATT depends on a profiled HSA queue + AQLprofile-generated vendor AQL packet lifecycle
  (`hsa_amd_profiling_set_profiler_enabled`, profiler-active queue packet, trace-control buffer, code-object markers),
  not one missing SQTT register. Reopen only as AQLprofile packet import/replay or native profiled-HCQ work.
- `amd-rocprofiler-reopen-tracks-scope-result-20260619.md` — scoped and executed the first phase for all three reopen
  options. Verdict: split tooling is the default usable path; AQLprofile packet replay is the only bounded reopen;
  native profiled-HCQ is project-level and should not start from another register sweep.
- `amd-rocprofiler-r1p1-aqlprofile-replay-result-20260619.md` — **executed Track 1 R1-P1.** Forcing tinygrad
  `AMD_AQL=1` is stable but still lifecycle-only (`0` body packets). AQLprofile has nonzero gfx1100 command material,
  but the old command-buffer output is not a direct HCQ replay blob. Remaining reopen requires a v2 AQLprofile packet
  exporter with tinygrad-owned trace/control buffers, or native profiled-HCQ.
- `amd-rocprofiler-r1p2-v2-exporter-scope-20260619.md` — scope for that remaining bounded reopen: v2
  `aqlprofile_att_create_packets` exporter, allocation callback table, tinygrad-mappable buffers, one HCQ AQL dispatch
  replay, and strict body-packet pass/kill gates before any native profiled-HCQ work.
- `amd-rocprofiler-r1p2-v2-exporter-result-20260619.md` — **executed R1-P2 P0.** Corrected the local v2 ABI:
  `aqlprofile_att_profile_t.agent` is `hsa_agent_t`, not `aqlprofile_agent_handle_t`. With the real HSA GPU agent,
  `aqlprofile_att_create_packets` passes for all swept ATT profiles, returns nonzero start/stop packets, and exposes the
  allocation callback table. Next boundary is P1/P2: bind those buffers to tinygrad-submittable GPU VAs and replay around
  one HCQ dispatch.
- `amd-rocprofiler-r1p2-hcq-replay-result-20260619.md` — **executed R1-P2 P1/P2; verdict
  `PASS_BODY_ATTRIBUTION`.** A separate HSA helper exports v2 AQLprofile vendor packets, tinygrad allocates HCQ-owned
  control/command/trace buffers, and the probe patches both raw 64-bit VAs and PM4 `VA >> 12` page-address fields. Full
  `start -> tinygrad body kernel -> stop` replay syncs and yields decodable SQTT body packets (`98,269` body-like
  packets), proving ROCprofiler ATT can be imported into tinygrad HCQ without HIP/HSA in the tinygrad process.
- `amd-att-primitive-attribution-scope-20260619.md` — next scope after ATT replay passed: use imported ATT on real
  tinygrad primitives, first decode MMVQ contract attribution (`76%` standalone HBM -> `44%` in-model) and then prefill
  non-matmul residual attribution. Strictly observability-only; timing authority remains clock-controlled A/B + PMC.
- `amd-att-primitive-attribution-result-20260619.md` — **executed the ATT primitive atlas; verdict
  `PASS_ATT_PRIMITIVE_ATTRIBUTION`, interpretation `ATT_USABLE_NOT_DECISIVE_FOR_INMODEL_GAP`.** ATT now
  body-attributes native tinygrad Q4_K coop (`168,693` body-like packets), imported llama Q4_K MMVQ (`163,942`), and
  pp512 SDPA (`135,442`). This clears the tooling blocker but does not change timing conclusions; next decode use is a
  role-joined in-model ATT pass.
- `amd-att-inmodel-role-join-scope-20260619.md`, `amd-att-inmodel-role-join-result-20260619.md` — **executed first
  role-joined in-model ATT pass; verdict `PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP`.** `blk.0.attn_output` launches the
  intended `q4k_coop_partial_4096_4096` plus stage-2 reduce/glue in-model, with `16,137` body-like ATT packets. This
  closes runtime/cache identity for that Q4_K role; next ATT target, if any, is higher-share Q6_K `ffn_down`/`lm_head`.
- `decode-standalone-retention-staged-attack-scope-20260619.md` — staged attempt to recover more of tinygrad's
  `~76%` standalone decode MMVQ efficiency in-model. Starts with Q6_K role-joined ATT (`ffn_down`, then `lm_head`),
  then reduce/glue Amdahl, one direct-output/reduce-fusion proof if gated, q8 lifecycle only if still justified, and
  finally project-level scheduler/resource work if all bounded routes fail.
- `decode-standalone-retention-stage1-q6-role-join-result-20260619.md` — **executed Stage 1 Q6_K role join via
  explicit `q6_surface_fallback` after full model load hit a 4.68GB allocation failure.** Both Q6 surfaces launch the
  intended native coop programs (`q6k_coop_partial_4096_12288`, `q6k_coop_partial_151936_4096`) plus reduce/glue, with
  ATT body attribution. No bounded Q6 fallback/wiring fix found; proceed to reduce/glue Amdahl ledger.
- `decode-complete-tooling-scope-20260619.md` — complete tooling scope for the remaining decode lifecycle question:
  join role identity, ATT body attribution, lifecycle accounting, timing authority, reduce/glue Amdahl, and llama
  comparison into one atlas before funding any direct-output/reduce-fusion or scheduler/resource build.
- `decode-complete-tooling-result-20260619.md` — **executed DCT-0..DCT-7.** Adds
  `extra/qk_decode_complete_tooling.py` and `bench/qk-decode-complete-tooling/*`. Verdict:
  `COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS`; reduce/glue is visible but does not clear the build gate, Q6 surface
  equivalence is accepted for visibility not timing, and ATT remains body evidence rather than timing authority.
- `decode-native-mmvq-scheduler-renderer-full-scope-20260619.md` — full scope for the remaining dependency-free
  native decode route after the tooling atlas: a project-level AMD scheduler/renderer path to preserve the MMVQ
  lifecycle contract in-model. Defines NSR-0..NSR-8, start criteria, gates, kill conditions, expected potential, and
  the boundary between q8 research-flag hardening and true native compiler ownership.
- `decode-q8-research-route-hardening-result-20260619.md` — small-path hardening pass. Consolidates W==D, dNLL,
  artifact hashes, fixed-launch boundary, and policy gate; verdict `PASS_RESEARCH_HARDENED_EXISTING_EVIDENCE`.
- `decode-fused-mmvq-integration-next-path-scope-20260619.md` — next base-decode path after the PMU convergence:
  tinygrad's standalone GEMV is stronger than llama's, but in-model weight-GEMV falls to `~44%` vs llama `~54%`.
  Scopes activation/Q8 reuse plus occupancy/launch-shape preservation, starting with measurement-only FMI-1/FMI-2.
- `decode-fused-mmvq-integration-fmi1-fmi2-result-20260619.md` — **executed FMI-1/FMI-2.** The in-model GEMV loss
  atlas passes (`44% -> 54%` projects `1.187x` if recovered across the weight-GEMV bucket), and llama/tinygrad launch
  contract diff passes. Decision: build Track B first, the byte-identical occupancy/launch-shape route.
- `decode-fused-mmvq-integration-fmi4-b1-result-20260619.md` — **executed FMI-4 B1.** Existing env launch-shape knobs
  (`Q4K_COOP_RT`, `Q6K_COOP_RT`, coop on/off) do not move a high-share role by `>=10%`; B1 is closed. Track B remains
  live only as runtime/cache identity or renderer/scheduler work.
- `decode-integration-diagnostic-result-20260619.md` — prefill-style decode localization. Verdict:
  **no single transpose-like tax**; Q4_K stage2 reduce is real but insufficient, q8 lifecycle is capped/lossy, env knobs
  fail, and the remaining large gap is MMVQ in-model contract preservation.
- `decode-fused-mmvq-integration-b2-runtime-cache-result-20260619.md` — **executed PCG-1/FMI-4 B2.** Runtime/cache
  identity closes: in-model decode and direct same-process role calls use the same program/launch identities for
  `attn_q/o`, `ffn_gate/up`, `ffn_down`, `lm_head`, and `attn_k/v`. The hidden wiring-bug route is closed.
- `primitive-pmu-observability-scope-20260619.md` — scope for using installed ROCm profiler tooling as the PMU oracle
  and building only the tinygrad primitive-local attribution layer needed around HCQ.
- `primitive-pmu-observability-result-20260619.md` — PMU-1..PMU-3 result: ROCm PMU works on HIP controls, but tinygrad
  HCQ is invisible to `rocprofv3` in the smoke; redirects to a tinygrad-native HCQ attribution adapter.
- `primitive-hcq-attribution-scope-20260619.md` — PMU-4 scope: tinygrad-native HCQ attribution for eager launches and
  graphs, producing Level-3 runtime/graph evidence without pretending to have PMU counters.
- `primitive-hcq-attribution-result-20260619.md` — PMU-4a..c result: probe-local attribution captures eager HCQ
  launches, HCQGraph construction/replay, and a Tensile runtime row; classifies `rocprof_hcq_visibility_gap` +
  `graph_rebind_ok`.
- `amd-schedule-codegen-exhaustion-scope-20260619.md` — cross-primitive scope for exhausting AMD scheduler/codegen by
  oracle, not as an open-ended compiler ambition. Uses q8 decode and Tensile prefill as authority cases.
- `amd-schedule-codegen-exhaustion-result-20260619.md` — **executed SCE-0/SCE-1.** Builds
  `bench/amd-schedule-codegen-exhaustion/oracle_matrix.json`: 7 feature rows are project-level, 1 artifact-only,
  1 bounded graph/rebind row, 1 tooling-blocked, 1 not worth owning, 1 already expressible. Native q8/prefill
  schedule generation is exhausted as a bounded primitive; remaining native work is a reusable AMD backend project.
- `prefill-address-lowering-renderer-arc-plan-20260619.md` — dependency-free prefill renderer arc. CG-W1.5 validates
  the real warmstarted in-model ffn matmul uses WMMA but is ALU-overhead-bound; CG-W2/2b then refute kernel-level
  coalesced/wide-copy fixes. The only remaining no-deps lever is renderer/opt-level fp16 load vectorization or
  hand-asm WMMA, both project-level.
- `route-a-a3-lds-multiwave-scope-20260619.md` — continuation scope for dependency-free RDNA3 WMMA hand-asm:
  LDS-staged, multi-wave GEMM to chase LLVM/Tensile after single-wave A2 stayed below LLVM.
- `route-a-a3-lds-multiwave-result-20260619.md` — **executed A3 P0/P1 gates.** P0 LDS tile smoke passes
  (RMSE `0.000209`); P1 multi-wave LDS GEMM faults even at `128^3`, so the next valid step is a smaller
  store/load-only address-mapping debug probe before any P2 pipeline/tuning.
- `prefill-tensile-research-measurement-scope-20260619.md` — complete Option A execution scope for Claude: finish the
  bounded JIT-dim step, route extracted Tensile prefill behind `PREFILL_TENSILE_GEMM=1`, and measure pp/dNLL as
  research-only evidence.
- `prefill-tensile-tpe7a-rebindable-node-result-20260619.md` — TPE-7a result: one extracted Tensile kernel object
  can be rebound to current buffers through graph-style kernarg filling; correctness/protocol proof only.
- **`performance-frontier-exhaustion-20260619.md` — latest exhaustion checkpoint.** Bounded decode primitives are
  exhausted; q8/RMSNorm is codegen-deferred; hand-LDS WMMA is refuted; external BLAS ceiling is measured; the bounded
  no-deps prefill WMMA sweep is refuted; EBT-1 kills the HIP-runtime bridge; the only material prefill route left is
  Tensile primitive extraction through HCQ or a codegen/Tensile-class rewrite.
- `qk-decode-per-role-delta-audit-20260618.md` — the quantitative per-role decode gap table (traffic/%peak/time-share/
  Amdahl/status); summed ceilings ~+27–30% ≈ the whole 1.47× llama gap, all behind one q8/full-MMVQ wall.
- `qk-machine-search-primitive-rows-20260618.md` — current machine-search rows (live + closed); supersedes the
  06-17 rows doc. Live/deferred: q8 side-channel, ffn coop sub-gate, attention residual audit, LDS flash-prefill,
  external/raw-HIP boundary/control; closed: quant-weight-reuse-8b, broad mmvq_q4k/q6k, decode_block_fusion,
  hand-LDS WMMA as the prefill lever, and bounded pure-tinygrad WMMA issue/occupancy.
- `q8-mmvq-lifecycle-deep-scope-20260618.md` — deep scope for the only remaining decode MMVQ lifecycle reopening:
  producer-side q8 from fused RMSNorm/apply into Q4_K ffn_gate/up int-dot. Explains what "q8/MMVQ lifecycle"
  means, what is already refuted, phase gates, and why this is low-EV/deep rather than a kernel tweak.
- `q8-mmvq-lifecycle-deep-result-20260619.md` — **executed it: Q8L-0/1 pass, Q8L-2 KILL.** The fused
  per-row→per-32 multi-output producer is NOT expressible via the store-group idiom (needs an LDS-reduction
  flash-style kernel); q8 side-channel is **deferred behind a codegen capability**, not a buildable arc — closes
  the last bounded decode research question.
- `q8-ffn-handwritten-oracle-scope-20260619.md` — research-only oracle scope for the q8 decode reopening: use
  handwritten kernels to test whether the deferred fused RMSNorm→q8 producer plus llama-style Q4_K int-dot consumer
  actually clears correctness, lifecycle speed, block EV, and dNLL gates before funding tinygrad codegen. Includes
  Q8H-0 preflight, Q8H-1 real-GGUF handwritten MMVQ correctness PASS, and Q8H-3/4 producer+lifecycle PASS
  (1.23x gate+up isolated), plus Q8H-5 EV PASS (~1.05x decode model); remaining gate is q8-lossy dNLL/W==D.
- `q8-dual-track-route-and-codegen-scope-20260619.md` — splits q8 into complementary tracks: Track A handwritten/
  backend research route for dNLL/W==D truth, and Track B tinygrad codegen transfer for owning the fused producer and
  q8 MMVQ lifecycle. Adds `extra/q8_ffn_quality_proxy.py`; Track A A0 quality proxy PASS with 160-token dNLL
  +0.00165, so next is HCQ-launchable handwritten route.
- `q8-ffn-fast-artifact-and-codegen-transfer-scope-20260619.md` — forward scope after A2: one-block q8 route is
  correct but COMGR-HCQ artifacts are too slow (`~195us` vs `<=129us` gate). Scopes the two remaining paths:
  hipcc-quality artifact loading through HCQ (`unknown AMD reloc 10` first) and tinygrad-owned raw/codegen transfer.
- `q8-ffn-fast-artifact-vs-raw-code-result-20260619.md` — **executed the two paths.** Path A hipcc/LLD artifact
  loading through HCQ passes when expressed as `producer + fused gate/up consumer` (`114.12us`, correct, no HIP runtime
  in-process). Path B current COMGR/raw-code route remains correct but too slow (`194.80us`). Reopens A3 graph/in-model
  routing only for the fast fused artifact route.
- `q8-ffn-fast-artifact-a3-route-result-20260619.md` — **A3 result.** Fast artifact one-block route passes eagerly
  (`121.38us`, correct vs q8 proxy). Initial Tensor-visible injection faulted; the contract audit found optimized-away
  input buffers and a wrong Q4_K dummy dtype/shape. After fixing both, eager injected node and TinyJit replay PASS
  (`max_abs 0.00137`, no HIP runtime). W==D decode is next.
- `q8-ffn-handwritten-a4-decode-result-20260619.md` — **A4 final gate PASS_RESEARCH.** `Q8_FFN_HANDWRITTEN=1`
  routes dense decode FFN gate/up through the graph-injected q8 artifact. W==D decode improves
  `1.051-1.063x` across ctx 128/512/1024/4096, and actual-route dNLL is `+0.002887` over 160 tokens. Default remains
  off; remaining question is artifact dependency vs codegen/ASM transfer.
- `q8-ffn-codegen-asm-transfer-scope-20260619.md` — **Track B scope + B0/B1 audit.** Disassembles the passing
  hipcc/LLD oracle and slower COMGR route. Both consumers already emit 16 `v_dot4_i32_iu8`; the gap is fused gate/up,
  producer shape, scheduling, and q8 side-channel lifecycle, not a missing dot intrinsic. Next build is a tinygrad-owned
  fused gate/up consumer (`<=60us`) before funding producer renderer work.
- `q8-ffn-codegen-b2a-comgr-fused-result-20260619.md` — **B2a COMGR fused-C result: FAIL_PERF.** The tinygrad-owned
  COMGR fused gate/up consumer is correct (`max_abs <=1.43e-6`) but slow (`146.88us` vs `<=60us`; lifecycle
  `177.72us` vs `<=129.2us`). Closes source-level C reshuffles; remaining B2 path is explicit AMD DSL/ASM or renderer
  scheduling work.
- `q8-ffn-codegen-b2b-asm-consumer-scope-20260619.md` — **B2b AMD DSL/ASM consumer scope + smoke PASS.** Adds
  `extra/q8_ffn_asm_gateup_smoke.py`, which emits `v_dot4_i32_iu8` through `Ops.PROGRAM` and HCQ with no C/hipcc path
  and stores the expected result. Next is a sliced hand-owned fused gate/up consumer: address skeleton -> q8/Q4 load
  skeletons -> one-block dot -> full fused gate/up, gated at `<=60us`. Final B2b verdict: **correctness PASS /
  PERF FAIL**. Full real-GGUF fused gate/up ASM consumer is correct (`max_abs <=1.43e-6`) but slow (`166.649us` vs
  `<=60us`), so native decode ownership is closed as project-level AMD scheduling/compiler work.
- `q8-ffn-amd-scheduler-work-scope-20260619.md` — next-layer scope after B2b: compiler/scheduler work, not primitive
  search. Defines S0-S5: disassembly accounting, reduction audit, address/scale-min audit, load/wait/dot scheduling,
  descriptor/local-id capability, and the decision gate for local hand schedule vs AMD DSL feature vs project-level
  scheduler. Recommendation: run S0 first only.
- `q8-ffn-amd-scheduler-s0-result-20260619.md` — **executed S0 and closed native q8 decode ownership.** tinygrad ASM
  emits the same 16 dot4 ops as hipcc/LLD and fewer static instructions (`218` vs `336`) but is still `166.649us` vs
  `<=60us`; visible deltas are load shape/address scheduling, not a bounded primitive edit. Verdict:
  `S0_CLOSE_PROJECT_LEVEL_SCHEDULER`.
- `q8-ffn-dynamic-scheduler-observability-scope-20260619.md` — scope for option 2 after S0: a tinygrad-native HCQ
  trace/counter bridge for the q8 visible gap. Defines DSO-0..5: q8 HCQ attribution rows, resource/occupancy metadata,
  controlled variant ladder, optional built-in AMD PMC/SQTT attempt, and final classifier
  (`load_shape_bound`, `wait_scheduler_bound`, `closed_project_level`, etc.).
- `q8-ffn-dynamic-scheduler-observability-result-20260619.md` — **executed DSO-0..5.** Classifier:
  `wait_scheduler_bound`. The decisive ladder is body-insensitive: reduction-only/synthetic-dot/load-only variants all
  remain ~0.151-0.153ms vs full ASM 0.166ms, so the visible q8 gap is broader AMD scheduling/work-decomposition/codegen,
  not a bounded load-shape primitive.
- `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md` — complete next-layer scope after DSO: Route A native
  tinygrad AMD scheduler/codegen transfer, Route B artifact/import research route, and Route C schedule-import training
  data. Defines gates for when to reopen q8 producer ownership vs keeping decode closed as compiler roadmap.
- `q8-ffn-artifact-import-route-result-20260619.md` — **executed Route B.** Reproducible hipcc/LLD artifact build,
  fixed-launch HCQ loader, graph injection, and maintenance boundary all pass as **research-only / policy-bound**.
  Isolated lifecycle `115.24us`; graph replay max_abs `0.001373`; default off; no in-process HIP runtime.
- `q8-ffn-route-a-scheduler-codegen-result-20260619.md` — **executed Route A A0/A1.** Oracle contract extraction
  passes, but AMD DSL capability map finds no bounded A2 feature: vector loads ~14us, wait grouping ~0.8us, reduction
  rewrite ~13us, dot4 already solved. Native q8 ownership stays project-level scheduler/codegen roadmap.
- `q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md` — **post-A1 evidence gate.** tinygrad HCQ-level PMC/SQTT
  collection works for the q8 ASM path (`2` PMC events, `12` SQTT events, ~1.78 MB trace), but local SQTT decode fails
  on the captured RDNA3 blobs and no bounded `>=30us` A2 feature is identified. Route A remains closed for q8 decode
  except as a project-level AMD scheduler/codegen effort.
- `spec-decode-bandwidth-amortization-scope-20260619.md` — reopens spec decode only under the PMU-backed
  weight-read-amortization framing. Keeps the old `decode_spec_verify_shortcut` closed; defines the new
  `decode_spec_weight_amortization_lifecycle` row, whose hard gate is T=K+1 verify `<=1.5x` one T==1 pass plus
  low-sync accept/commit and greedy byte-exactness.
- `spec-decode-bandwidth-amortization-sdb1-sdb2-result-20260619.md` — **executed SDB-1/SDB-2.** Current spec remains
  non-viable (`~0.52x` before runtime with 0.6B K=4) because verify is `4.65x`; reaching `<=1.5x` requires a
  `67.8%` verify cut across Q4_K, Q6_K/lm_head, and attention/reduces. No bounded shared primitive; spec is
  project-level T-cheap batched-forward work.
- `spec-decode-tcheap-batched-forward-project-scope-20260619.md` — project-level decode-only scope for making spec
  viable after SDB-1/SDB-2: a short-block target verify forward, low-sync accept/commit, exact KV protocol, and
  T=K+1 verify `<=1.3-1.5x` one pass. Explicitly not a prefill route and not a bounded kernel edit.
- `spec-decode-tcheap-batched-forward-tbf0-tbf2-result-20260619.md` — **executed TBF-0..TBF-2.** Short-block verify
  IR contract is defined, but current component gates all fail: Q4_K `2.916x`, Q6_K/lm_head `5.831x`,
  attention/reduces `3.061x`, linears group `3.523x` vs the `<=1.5x` T-cheap gate. Stops before TBF-3 until a
  concrete component route exists.
- `spec-decode-component-route-candidates-scope-20260619.md` — next decode-only scope after TBF-0..2: candidate
  routes for grouped short-block quantized linears, short-block causal verify attention, and their combined
  projection. No implementation until a candidate passes its component gate.
- `spec-decode-component-route-candidates-result-20260619.md` — **executed SCR-0..SCR-4.** Candidate attention
  generalization has no bounded proof surface, grouped short-T linears have no shared Q4_K/Q6_K bounded schedule, and
  combined projection has no passing ceilings. Verdict: `PROJECT_LEVEL_CLOSE`; do not build TBF-3 unless a new measured
  component candidate clears `<=1.5x`.
- `llama-kernel-residual-primitive-audit-scope-20260619.md` — scope for auditing llama.cpp's **own** remaining
  primitive headroom: MMVQ residual-to-peak, q8 quant, attention, small-op fusion, graph boundaries, and prefill.
  Separate from the tinygrad-vs-llama gap explanation.
- `llama-kernel-residual-primitive-audit-20260619.md` — result of that audit. llama is not theoretically optimal,
  but fresh `rocprofv3` traces show prompt-free decode is 85.6% MMVQ; q8/RMSNorm is the only moderate non-MMVQ
  decode lifecycle candidate, graph launch overhead is already solved by HIP graphs, and pp512 prefill is 74.4%
  quantized MMQ/matmul rather than attention-limited.
- **Decode-attention wins SHIPPED (byte-identical greedy, default-on):**
  - `qk-8b-attention-fusion-result-20260617.md` — flash-decode threshold 1024→512 (+12.8% ctx520).
  - `qk-8b-flash-variant-result-20260617.md` — `hoisted` exp + L=128 default (+11–29% across ctx).
  - `qk-gqa-coop-vector-load-result-20260617.md` — `gqa_coop_vec` default → decode-attention slope gap CLOSED.
- **Q4_K MMVQ int-dot line — CLOSED:** `qk-mmvq-int-dot-closeout-20260618.md` (**read this**) — the
  consolidated bank. SHIPPED `_sdot4`→native signed dot4 via `__builtin_amdgcn_sudot4` (fixed a latent
  unsigned-bug; value-tested; used by no default path); 128-thread/row sudot4 kernel 57% correct (beats opaque
  52%) but whole-linear REFUTED by the q8-pack wall (reuse ceiling 2 + ~7µs pack floor); int-dot FFN refuted.
  - Key sub-arcs (provenance): `qk-dot4-isa-audit-20260618.md` (the sudot4 fix + RDNA3 dot4 ISA map),
    `llama-q4k-mmvq-scheduler-audit-20260618.md` (llama's MMVQ decomposition),
    `qk-mmvq-llama-scheduler-probe-verdict-20260618.md`, `qk-mmvq-sudot4-full-linear-arc-20260618.md`,
    `qk-q8-activation-lifecycle-verdict-20260618.md`, `qk-mmvq-{codegen,deep-linearizer,fused-coop-row}-*`.
- **Current decode standing:** ~66–69% of llama via the shipped coop + flash-decode routes. Residual MMVQ gap =
  per-thread codegen (tinygrad-internals). 14B/32B pivot deferred per standing preference.

## Active / open frontiers

- `prefill-wmma-lds-tiling-scope-20260619.md` — provenance for the now-refuted Branch A. After decode closed, the surviving high-EV arc:
  PREFILL_V2 forward is ~74% fp16 WMMA matmul emitted with LDS=0; the lever is WMMA operand LDS-tiling (~1.6× pp).
  Decision-first: Phase PWLT-0 is the authority call — Branch A (tinygrad hand-LDS, **triple payoff**: also unblocks
  q8 producer + flash-prefill attention) vs Branch B (external hipBLASLt/rocBLAS, prefill-only). Both feasible
  (assets/libs present); recommendation A-first, B as fallback control.
- `prefill-wmma-lds-tiling-result-20260619.md` — **executed Branch A: PWLT-A1 pass, PWLT-A2 KILL.** Hand-LDS WMMA
  = 1.02× the default matmul (both ~34% peak) → **LDS-tiling is NOT the lever** (IC-served on gfx1100, like decode
  attention). Real headroom is dense WMMA issue / Tensile-class scheduling, not LDS staging.
- `prefill-external-blas-result-20260619.md` — **ceiling/control measured.** Host-only C++ avoids the split-HIP
  compile issue; hipBLASLt reaches 69.8 TFLOPS on ffn_gate/up (1.71× tinygrad) and rocBLAS reaches 70.9/76.7 TFLOPS
  on ffn_down/attn_q/o. This proves a higher GEMM ceiling, but routing remains an external-dependency + HCQ-vs-HIP
  runtime boundary.
- `prefill-external-rawhip-tensile-boundary-scope-20260619.md` — broad external/raw-HIP/Tensile boundary scope
  before EBT-1.
  Starts with the authority decision, then EBT-1 tinygrad-buffer pointer interop, EBT-2 bridge/shape overhead,
  EBT-3 one-block transfer, EBT-4 full warm pp, and fallback lanes for Tensile HSACO or raw-HIP kernels. It also
  states the key gate conflict: strict >=1.5x full pp likely stops because the measured ceiling caps around
  1.4-1.45x before overhead. Superseded as the active plan by the Lane B scope below after EBT-1 killed Lane A.
- `prefill-external-bridge-ebt1-result-20260619.md` — **executed EBT-1: Lane A KILL.** HIP runtime and tinygrad
  HCQ/KFD are mutually exclusive in one process, so in-process rocBLAS/hipBLASLt on tinygrad pointers is closed.
- `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md` — **current Lane B scope.** Extract the
  selected Tensile primitive and its full launch contract (solution, HSACO, symbol, `.kd`, kernargs, launch geometry,
  workspace) and run it through tinygrad HCQ. Also scopes option 2: only after a working extracted contract exists,
  use it as the target for a tinygrad codegen/Tensile-class schedule transfer.
- `prefill-tensile-tpe4-perf-result-20260619.md` — **executed TPE-4: PASS.** The extracted rocBLAS Tensile
  ffn_gate/up primitive runs through tinygrad HCQ at 66.91 TFLOPS median (0.7703 ms), correct, no copies, no HIP
  runtime in-process. Lane B is now runnable and fast for one fixed shape.
- `prefill-tensile-tpe5-shape-matrix-result-20260619.md` — **executed TPE-5: PASS.** The extracted Tensile primitive
  generalizes: ffn_gate/up 66.8, ffn_down 68.9 (StreamK, no workspace), attn_q/o 58.9 TFLOPS through HCQ — all correct,
  stable, no workspace/aux/layout-copies, one code object + one pointer convention. Weighted model predicts **~1.40×
  full warm pp512** (→ ~2920 tok/s ≈ 95% of llama) if all three are routed, above the 1.25× gate.
- `prefill-tensile-tpe6-block-transfer-result-20260619.md` — **executed TPE-6: REDIRECT.** A whole FFN block
  (gate+up+silu·up+down) routed through the kernels is **exact** (rel 4.8e-4) and copy-free (weights stay natural
  `[out,in]`, run in `[feature,T]` space, zero per-matmul transposes), and the block matmuls hit 61 TFLOPS = **1.53×
  the PREFILL_V2 plateau on GPU time**. But naive per-op routing adds ~6.2 ms host sync overhead (a JIT-less probe
  artifact) that swamps the win end-to-end → realizing it needs a **single-dispatch graph (HCQGraph/TinyJit) runtime
  helper**. Next: build that helper, re-run the block gate, then TPE-7 (no model default; external-artifact policy pending).
- `prefill-own-wmma-kernel-scope-20260619.md` — pure tinygrad/no-deps scope. Key learning: tinygrad's
  WMMA matmul (41 TFLOPS) only *matches* the non-WMMA ALU matmul (40) — it gets **none** of the tensor-core 2×, so
  WMMA units are **stalled, not the bottleneck**. POWN-0 diagnose (occupancy / accumulator-chain / issue-rate) →
  POWN-1 config sweep (LDS-off since IC-served, chase dense WMMA issue + occupancy) gated ≥1.5×. The result below
  banks the bounded no-deps ceiling.
- `prefill-own-wmma-kernel-result-20260619.md` — **executed POWN-1: KILL.** Best config is the existing
  B128x128x16/W2x2 at 42.0 TFLOPS; more waves, bigger tiles, BK32, and noLDS all regress. No bounded no-deps
  prefill WMMA knob reaches the 62 TFLOPS gate.
- `prefill-external-blas-scope-20260619.md` — **DECLINED (no external deps).** rocBLAS/hipBLASLt ceiling-first plan;
  kept as provenance for the bridge analysis (DEV=AMD HCQ vs HIP-runtime). Its PXB-1 ceiling has now been measured
  in `prefill-external-blas-result-20260619.md`.
- **`amd-decode-prefill-v2-increment1-20260617.md`** — **prefill v2 BUILT & WON: ~13x warm prefill** (189→2486
  tok/s, ~83% of llama) via concrete-ubatch + fp16 + realized-weights + warmstart-TC, gated `PREFILL_V2`,
  decode untouched. Quality gate PASSED (dNLL ~0, 8B). Corrects the Stage-0 gate's premise (lazy weights →
  realize/VRAM; per-shape opts; host-overhead confound). Gate: `amd-decode-prefill-v2-gate-20260616.md`.
- **`amd-decode-prefill-v2-increment2-20260617.md`** — **flash-prefill attention: GATED (banked)**. Attention
  is the next prefill bottleneck at long ctx (~51% @ sp=3072) but the tractable approaches are refuted.
- **`amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`** — **CORRECTION + kernel-level
  confirmation**: a custom score-free fused attention kernel IS expressible/correct (bridge + capabilities +
  expressibility proven, `test_flash_prefill_custom_kernel*.py`), but **honest DEBUG=2 GPU time REFUTES it on
  perf (~170–760× SLOWER than SDPA**; the earlier ~2.7× were wall-clock artifacts). Score-free w/o LDS reuse =
  memory-bound; real flash-2 needs LDS tiling (BEAM-territory, hangs gfx1100). Flash-prefill banked; prefill v2
  rests at Increment 1. **Methodology lesson: GPU timing via DEBUG=2 `tm`, never wall-clock around `.realize()`.**
- `amd-decode-prefill-plan.md` — the original prefill diagnosis (~2% of llama; LDS cache-blocking). Superseded
  as the active plan by prefill v2 above, but still the canonical root-cause reference.
- Phase-2 decode docs (2026-06-16): `amd-decode-sequential-tax-profile`, `…-overlap-feasibility-spike`,
  `…-overlap-derisk`, `…-two-queue-probe` (**overlap GATED** on a 2nd compute ring), `…-demotion-search`
  (B3 done), `amd-decode-flash-attention-plan` (flash SHIPPED).
- Direction + status: `structure/Development/machine-search-decode-context-plan-2026-06-16.md`;
  running log `structure/Development/session-handoff.md`.

## Machine-search system (shipped this arc)

The bounded search loop, dogfooded on B3. Code: `extra/qk_search_spec.py` (schema authority),
`extra/qk_nll_eval.py` (decode-path dNLL gate), `extra/qk_demote_search.py` (orchestrator). Result:
`amd-decode-demotion-search-20260616.md`.

## Architecture references (live)

- `amd-decode-harness-architecture.md`, `amd-decode-qk-storage-architecture.md`,
  `amd-decode-primitive-v2-design.md`, `amd-decode-bandwidth-roofline.md`,
  `amd-decode-packed-{load-lowering,qk-tile-design,qk-semantic-op}.md`.

## Historical — the decode-arc probe log

Dated scope/result docs whose verdicts are now captured in the syntheses above. Kept for provenance;
**not current state** (several carry a SUPERSEDED header).

- *"current state" docs, now superseded by the bank:* `amd-decode-current-verdicts.md`,
  `amd-decode-methodology-and-roadmap.md`, `amd-decode-final-report.md`, `amd-decode-hypothesis-statement.md`,
  `amd-decode-consolidated-first-principles.md`, `amd-decode-optimization-plan.md`.
- *bottleneck diagnosis & probes:* `amd-decode-rootcause`, `…-fix-plan`, `…-perlayer-plan`,
  `…-validate-plan`, `…-memory-access-audit`, `…-dequant-instruction-count`, `…-latency-vocabulary`,
  `…-dp4a-vocabulary`, `…-prefetch-plan`, `…-mirage-probe`.
- *kernel/TC/GEMM probes:* `…-option1-result`, `…-option1-corrected`, `…-batched-tc-{plan,result}`,
  `…-warmstart-plan`, `…-verify-loop-plan`, `…-fusion-probe-plan`, `…-vdot-amort-plan`,
  `…-amortized-quant-plan`, `…-scale-and-vdot4-plan` (`amd-loop-…`), `…-semantic-family-b`,
  `…-lossy-quant-search`.
- *levers later synthesized:* `amd-decode-speculative-plan` (B5), `amd-decode-prior-art`.

## Flywheel sub-arc (model-to-kernel triage/generation) — concluded

Read the postmortem first; the learned model added no value at the current feature set, the
native-matmul loop substrate works (decoupled from the decode bar).

- `amd-decode-flywheel-postmortem.md` (read first), `amd-decode-loop-substrate.md`,
  `amd-decode-flywheel-proof-plan.md` (2.6k-line plan), `amd-decode-kernel-optimization-flywheel.md`,
  `amd-decode-ansor-direction.md`, `amd-decode-loop-live-plan.md`,
  `flywheel-judging-rewrite-scope.md`, `flywheel-rewrite-ubuntu-handoff.md`,
  `qwen-json-eval-objective-scope.md`, `research-paper-brief.md`.

## Other subsystems

- **PSP / boot** (separate from decode): `amd-kdb-root-cause.md`, `amd-linux-psp-good-trace.md`,
  `amd-ubuntu-boot-prompts.md`, `amd-remote-dropout-investigation.md`.
- **Reference research:** `amd-rocm-llamacpp-research.md` (llama.cpp/ROCm/MMQ deep dive).

## Upstream tinygrad docs (not fork-specific)

`index.md`, `quickstart.md`, `mnist.md`, `nn.md`, `dtypes.md`, `env_vars.md`, `runtime.md`,
`tinygpu.md`, `tinybox.md`, `showcase.md`.
