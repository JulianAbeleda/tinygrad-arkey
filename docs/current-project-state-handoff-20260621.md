# Current Project State — Handoff (2026-06-21)

Canonical, high-signal snapshot. If anything elsewhere contradicts this file, this file (and the linked
reconciliation result) wins. Machine: gfx1100 RX 7900 XTX 24GB, Qwen3-8B-Q4_K_M.

## 1. Canonical numbers (clean-wall, PROFILE=0, auto clock)

| metric | value | source |
|---|---|---|
| decode @ctx≈0 | **~85–86 tok/s** (empty-KV peak; a contextual number, see §3 on `87.6`) | CLI `--warmup --benchmark` |
| decode @ctx 512 / 1024 / 4096 | **68.1 / 66.4 / 60.7 tok/s** (≈ **~67% llama** — the steady-state headline) | `extra/qk_decode_runtime_overhead.py` (W) |
| q8 opt-in @ctx 512 / 1024 / 4096 | **72.8 / 70.9 / 64.3 tok/s** (~+7%, default-OFF, dNLL-gated) | same harness, `Q8_FFN_HANDWRITTEN=1` |
| prefill (opt-in fast path) | concrete-KV **73–111% of llama pp512**; warm prefill **0.17–1.6 s** | `docs/prefill-policy-integration-result-20260620.md` |
| VRAM | default ~5–6 GB; **`PREFILL_V2` adds ~+14 GB fp16** (≈19–21 GB), resident through decode | `docs/decode-prefill-headline-reconciliation-result-20260621.md` |

## 2. Decided policies (do not re-open)

- **Global `PREFILL_V2` default: OFF** (decided 2026-06-21). It is **not** flipped to `auto` — the +14 GB fp16
  prefill state stays resident during decode for zero decode benefit; the common decode/short-prompt user must not pay it.
- **`PREFILL_V2=auto`**: opt-in (VRAM-gated; enables only where it fits — 24GB+ on; ≤16GB / unknown off).
- **`PREFILL_SERVER_PROFILE=1`**: opt-in (⇒ `PREFILL_V2=auto` + concrete-KV precompile; the server/long-prompt profile).
- **`PREFILL_REMAINDER_FIX`**: default-ON but only active under `PREFILL_V2`; byte-identical (kills the 32-token trap).
- **q8 FFN (`Q8_FFN_HANDWRITTEN=1`)**: opt-in, default-off.

## 3. Closed lanes

- **Prefill kernels** — solved. Flash-prefill v2 was correct but ~15× too slow; not reopened.
- **Prefill default-owner call** — closed: global default stays OFF (§2).
- **Bounded decode fusion** — closed. FFN activation producer-fusion built & byte-exact but ~0% (work-conserved,
  not launch-recoverable); attention reduce/stat microfusion is a no-go (intrinsic O(KV) QK/softmax). See
  `docs/decode-fusion-build-result-20260620.md`.
- **Bounded decode vector-tile** — closed/rested (2026-06-21). The corrected T=1 principle was applied to the
  existing winner `gqa_coop_vec` by lowering `FLASH_L` (more KV-splits): **FLASH_L=64 passed the standalone
  attention gate (~1.08× @ctx1024, byte-exact) — validating the principle — but FAILED W==D promotion**
  (+2.8%@512, +1.8%@1024, **−1.2%@4096**; below the ≥5% bar, regresses long context). New hand-tiles (scalar
  fused LDS+GQA, warp-cooperative) were byte-exact but slower than `gqa_coop_vec`'s matmul q·k. **Decision:
  `REST_DECODE` for bounded work.** Do **not** promote FLASH_L=64 by default. See
  `docs/decode-vector-flash-tile-realigned-result-20260621.md`.
- **`87.6` ambiguity** — reconciled. `87.6` is a **numeric coincidence**: a real **ctx≈0 decode tok/s** (~11.4 ms)
  AND, separately, a real **ctx4096 decode ms/token** (=11.4 tok/s). **Never quote `87.6` bare.** The decode headline
  is the **curve** (~86 @ctx≈0 → ~61 @ctx4096), characterized as **~67% llama**, not the ctx≈0 peak.

## 4. Frontier — bounded decode is RESTED; the Method pillar is underway

- **Machine-search evaluator BUILT (2026-06-21).** `extra/qk_decode_eval.py` is the first-class, automated form of
  the lifecycle ladder (correctness → local A/B → whole-decode W==D → policy), emitting schema'd verdicts. It
  **reproduces the project's historical classifications** (baseline→REST, flash_l_64→LOCAL_PASS_WD_FAIL,
  warp_tile→FAIL_LOCAL_AB, q8→PASS_OPT_IN) and answered the key falsifier: **whole-decode W==D auto-clock variance
  is <0.6% ≪ the 5% promotion margin** (`EVALUATOR_READY_FOR_LIFECYCLE_SEARCH`; GPU-state tooling NOT needed). It
  only measures — no defaults change. See `docs/decode-evaluation-harness-hardening-result-20260621.md`, ledger
  contract `bench/qk-lifecycle-search/evaluator_contract.json`.
- **Lifecycle-search loop v0 BUILT (2026-06-21).** `extra/qk_lifecycle_search_loop.py` is the first closed
  `generate → evaluate → prune` loop on the evaluator: it runs valid candidates through `decode_eval` (4 executed,
  verdicts match) and **prunes invalid ones before benchmarking** (a WMMA-decode reopen → `PRUNE_CLOSED_LANE`, a
  FLASH_L=64 default-promotion → `PRUNE_POLICY_VIOLATION`). It builds no kernels, changes no defaults, proposes
  (dedup'd) ledger updates, and surfaced + drove a fix to a q8 auto-clock measurement confound (now reads the
  controlled lane). `LIFECYCLE_SEARCH_V0_READY`. See `docs/lifecycle-search-loop-v0-result-20260621.md`.
- **Candidate-template generation layer v0 BUILT (2026-06-21).** `extra/qk_candidate_template_gen.py` is the
  'generate' step: it expands 4 route/fusion/layout templates into 9 legal decode candidate **specs** (in the
  `search_candidates` schema, deterministic, with policy metadata) that flow **through the loop** unchanged — 3
  executable (bound to existing decode_eval candidates, verdicts match) + 6 pruned/deferred (closed-lane reopens,
  default-promotion attempts, and the north-star `flash_attn_tile` as a `PRUNE_NEEDS_TEMPLATE` deferred candidate).
  No kernels/flags/defaults. Loop gained a one-line `--candidates` option + a `deferred`→`PRUNE_NEEDS_TEMPLATE`
  branch. `TEMPLATE_GENERATION_V0_READY`. See `docs/candidate-template-generation-v0-result-20260621.md`.
- **North-star evaluator-binding templates v0 BUILT (2026-06-21).** `bench/qk-decode-eval/binding_templates.json`
  (`north_star_flash_attn_tile_v0`) now SPECIFIES exactly what an executable north-star flash_attn_tile candidate
  must declare/run/produce vs `gqa_coop_vec` (comparator, T=1-parallelism artifact fields, local-A/B + W==D runners,
  no-WMMA, 5 gates, 7 stop conditions). `gen_north_star_flash_attn_tile` carries this binding and is now a precise
  `PRUNE_NEEDS_TEMPLATE` ("binding exists, missing: kernel + local_ab runner + W==D route"). The loop resolves
  binding metadata into three cases (missing template / present-but-no-runner / executable); a no-GPU
  `north_star_binding_selftest` (`SELFTEST_PASS`) proves the executable binding path with no perf claim. No kernel,
  no defaults. `NORTH_STAR_BINDING_TEMPLATE_READY`.
  See `docs/north-star-evaluator-binding-templates-result-20260621.md`.
- **North-star flash_attn_tile candidate EXECUTED + REFUTED (2026-06-21).** First real north-star attempt through
  the system: `extra/qk_north_star_flash_attn_tile_ab.py` (warp-cooperative q·k partial + many-workgroup combine)
  ran the local A/B vs `gqa_coop_vec` and **MISSED: 0.58×@ctx1024, 0.89×@ctx4096** (byte-exact) → `FAIL_LOCAL_AB`.
  Per discipline, **stopped before any W==D route**. `gen_north_star_flash_attn_tile` now EXECUTEs (was
  `PRUNE_NEEDS_TEMPLATE`) → decode_eval → FAIL_LOCAL_AB → refute_candidate. `NORTH_STAR_FAIL_LOCAL_AB`. **The
  ceiling was then re-audited (next bullet) — the combine is NOT the lever.** See
  `docs/north-star-flash-attn-tile-execution-result-20260621.md` and the redesign audit below.
- **North-star redesign audit (2026-06-21) — bounded tile lever EXHAUSTED; `REDESIGN_AUDIT_POINTS_TO_CODEGEN_DATAFLOW`.**
  A throughput probe (`extra/qk_north_star_dispatch_probe.py`) **corrects the prior diagnosis**: the combine is NOT
  HBM-bandwidth-bound (pout ~1 MB ≈ ~1 µs; the latency-measured "combine cost" was 2nd-raw-dispatch overhead vs
  coop's batched JIT graph). Under fair throughput the candidate is 0.46/0.52/0.87× @ctx512/1024/4096, **flat
  ~163 µs** while coop **scales** 75→144 µs → the ceiling is the **cooperative-dot q·k PARTIAL** (latency/occupancy-
  bound), and **coop's matmul q·k is near-optimal for tinygrad primitives**. No bounded combine/compact-state tile
  can pass @ctx1024. The 10× gap to llama is hand-tuned-kernel **codegen quality**, not a dataflow restructure.
  Do NOT build another bounded tile/combine. See `docs/north-star-decode-attention-redesign-audit-20260621.md`.
- **Decode codegen/dataflow capability scope (2026-06-21) — `CODEGEN_SCOPE_LLAMA_ORACLE_FIRST`.** Decision between
  native tinygrad codegen (A), a llama source-port reference oracle (B), and rest (C). **Chosen: B first.** Native
  codegen is a **multi-week linearizer project** (the single fused flash kernel — coupled `(m,l,acc)` online softmax
  across different range nests — is blocked at `tinygrad/uop/spec.py:163-165` single-op `REDUCE` + the shared-range
  store-group idiom + `linearizer.py:54-82`; pre-refuted as `flash_fused_multireduce_linearizer_wall`, "not bounded")
  **with no validated target** (my fused tile already failed). The **central question is unmeasured**: the llama
  audit numbers are **in-model only** — we have NEVER measured llama's kernel **standalone** vs coop. llama's actual
  source is on disk (`/home/ubuntu/env/llama.cpp/.../fattn-tile.cuh` + the Hd=128 instance) and the raw-HIP→tinygrad-
  Buffer bridge is proven. So the next project ports it as a **non-default reference oracle**, measures standalone
  **throughput** vs `gqa_coop_vec` (first gate ≥1.05× @ctx1024) through the existing `ab_script` binding, and that
  resolves whether the 10× is **standalone kernel-codegen** (→ scope native codegen with a real target) or **in-model
  integration** (→ redirect to the W==D/dataflow frame). See `docs/decode-codegen-dataflow-capability-scope-20260621.md`.
- **Llama flash_attn_tile reference oracle EXECUTED (2026-06-21) — `LLAMA_ORACLE_LOCAL_AB_PASS`. The central question
  is ANSWERED: llama's win is the STANDALONE kernel, ~5-6×.** Pure-GPU-time A/B (llama rocprofv3 trace vs coop
  tinygrad ProfileGraphEvent — both HW timestamps, dispatch confound eliminated): **llama attention 10.2/12.2/27.7 µs
  vs coop 59.9/69.9/132 µs @ctx512/1024/4096 → llama 5.87/5.71/4.77× faster STANDALONE.** So the 10× gap is a
  **standalone kernel-codegen target, NOT only in-model integration** → **native fused-flash codegen IS aiming at the
  right layer**, and llama's `flash_attn_tile` is the validated target. Method = PROFILING oracle (Phase-0 confirmed
  the full source port is BOUNDED — no cp_async/WMMA/broad-ggml — but it's deferred to the codegen follow-up that
  needs a re-runnable byte-level oracle; rocprofv3 doesn't hook tinygrad's HCQ so coop used ProfileGraphEvent).
  Registered as decode_eval `reference_oracle` candidate (`PASS_ORACLE_LOCAL_AB`, **non-promotable** — vendored llama
  reference, never a default route). See `docs/llama-flash-attn-tile-oracle-result-20260621.md`.
- **Native fused-flash linearizer scope (2026-06-21) — `NATIVE_FLASH_LINEARIZER_SCOPE_READY`, premise CORRECTED.**
  An empirical probe **REFUTES the "compiler expressiveness wall"**: the coupled online-softmax+V fused decode kernel
  (running `m`,`l`,`acc[D]` coupled via `corr=exp(m_old−m_new)`) **already VERIFIES, LOWERS, and RUNS value-correct in
  ONE kernel TODAY** via the existing `UOp.set`/`.after`/register-array idiom — **no `spec.py` REDUCE change, no
  linearizer change**. The "6-kernel split / coupled reduces trip the linearizer" was an **idiom pitfall** (same-slot
  RAW → use a mirror slot; GROUP-shape-index `ops.py:372`; two-ENDs-over-one-range `linearizer.py:81` → one `END(j)`),
  not a wall. So `NEEDS_UOP_REDUCE_DESIGN_FIRST` / `NEEDS_LINEARIZER_RANGE_MODEL_FIRST` are **refuted**, and the prior
  "multi-week linearizer project" framing is corrected: the next step is a **BOUNDED kernel-build** (Path A:
  `flash_fused_decode` = coop's matmul q·k + ONE fused softmax+V kernel), first gate = value-correct (met) + local
  A/B + in-model W==D vs `gqa_coop_vec`. **Honest caveat:** Path A fuses only softmax+V (keeps coop's matmul q·k), so
  it tests a likely-MARGINAL fusion win — the 5–6× gap is the **in-kernel-q·k CODEGEN QUALITY** (the warp-tile floor),
  which Path A does not address (deep, deferred until Path A's A/B). No `tinygrad/` change for Path A; profiling
  oracle + numpy suffice (no llama port first). See `docs/native-fused-flash-linearizer-scope-20260621.md`.
- **Path A fused softmax+V tail EXECUTED + REFUTED (2026-06-21) — `FUSED_SOFTMAX_V_TAIL_FAIL_LOCAL_AB`.** Built the
  achievable Path-A kernel (`extra/qk_fused_softmax_v_tail_ab.py`: coop matmul q·k + `flash_prob` fused INTO the
  partial as inline per-d-lane exp, keep `flash_max` + GQA reuse), value-correct (rel_rmse 8e-4). **MEASURED
  0.725×@ctx1024 / 0.876×@ctx4096** vs `gqa_coop_vec` → LOSES → `FAIL_LOCAL_AB`, no W==D (discipline). WHY: fusing exp
  into the partial makes W=129 output lanes RECOMPUTE exp per key vs coop's `flash_prob` (once/key) — the ~129×
  redundant exp outweighs the saved `prob` materialization (same redundancy that sank fused-LDS 0.21×). coop's
  hoisted-exp split is **near-optimal**; **tail fusion does not help**. The FULL online-max removal (fusing
  `flash_max` too) is **BLOCKED_BY_IDIOM** — per-split `pm` + per-d `pout` from one kernel = the Q8L-2 two-granularity
  store wall. So the proven expressiveness does NOT yield a decode win; the 5–6× gap stays the **in-kernel q·k codegen
  quality** (deep, no bounded gate; Path A keeps coop's matmul q·k and doesn't attack it). Decode lever options now:
  the deep q·k-codegen-quality project, or REST. See `docs/fused-softmax-v-tail-candidate-result-20260621.md`.
- **Decode frontier decision (2026-06-21) — `FRONTIER_LOW_LEVEL_TOOLING_FIRST`.** A purely-diagnostic per-kernel
  breakdown **corrects the "deep q·k codegen" framing**: coop's 70µs @ctx1024 is `flash_partial` 24.7µs (35%) +
  **matmul q·k 13.9µs (20%, ≈ llama's WHOLE 12µs tile)** + softmax kernels ~28µs (40%). So the **q·k matmul is NOT
  the bottleneck** (it's llama-tile-class); the 5.7× gap is the **softmax+V multi-kernel** (separate, individually
  inefficient kernels) that Path A proved can't be fused. We lack **counter/ISA-level attribution** of WHY
  `flash_partial`/softmax are slow → a deep q·k-codegen project (A) is **mis-targeted**, and a llama port (B) /
  immediate codegen are premature. **Next = low-level tooling (diagnostic first):** rocprof-compute counters +
  AMDGCN ISA disasm of `flash_partial_coop_vec` vs llama's `fattn-tile` (reuse the existing rocprof/SQTT/ATT tooling)
  to NAME the inefficiency — then either a targeted codegen fix (gated by local A/B + the oracle) or `REST_DECODE`
  with counter-level proof. All bounded decode lanes are exhausted. See
  `docs/decode-frontier-decision-after-path-a-20260621.md`.
  fusion, micro-fusion, launch-removal, scalar fused LDS+GQA tile, warp-cooperative tile, and split-count tuning
  (`FLASH_L=64`). The latest (`FLASH_L=64`) validated the T=1 split principle locally (~1.08× attention @ctx1024)
  but missed W==D promotion (+1.8%@1024, −1.2%@4096). **Do not pursue another bounded tile or flag sweep.**
- **The only remaining decode lever is north-star lifecycle/codegen**, not a tactical patch: the full llama-style
  non-WMMA vector `flash_attn_tile` — many KV-split parallel blocks **with an efficient many-split / stream-k
  combine** at T=1, LDS K/V staging, GQA query-head column packing, K-tile-batched vectorized body, register
  online-softmax. The executed north-star candidate + redesign audit (above) pinned the real ceiling: it is NOT the
  combine (pout traffic is ~1 µs, negligible) — it is the **q·k partial** (a hand-rolled cooperative dot is
  latency/occupancy-bound, flat ~163 µs, while coop's matmul q·k scales efficiently). coop's matmul q·k is
  near-optimal for tinygrad primitives; the 10× gap to llama is hand-tuned-kernel **codegen quality**. The bounded
  standalone-tile / combine lever is **exhausted** (every tile replaces coop's matmul q·k with a slower dot and
  loses); the only remaining lever is llama-quality kernel codegen — a deep capability, funded separately if at all. See
  `docs/llama-decode-primitive-difference-audit-result-20260621.md` and
  `docs/project-north-star-llama-and-lifecycle-search-20260620.md`.
- **Principle:** for decode `T=1`, a primitive must preserve/enlarge parallelism from KV splits and GQA columns;
  fusion/LDS/GQA reuse that collapses workgroups is harmful; compare against `gqa_coop_vec`, not weaker baselines;
  and **apply the principle to the existing winner and its split parameters before building a new hand-tile**.
  Canonical principle doc: `structure/Development/performance-primitive-research-principles.md`.
- **No tactical decode patch.** A decode candidate must clear BOTH gates — standalone ≥1.05× @ctx1024 vs current
  `gqa_coop_vec` AND W==D ≥5%@1024 / ≥7%@4096 with no ctx512 regression — or rest. `FLASH_L=64` cleared the first
  but not the second, so it is **not promoted**.

## 5. Where to start

`docs/README.md` · `bench/README.md` · `docs/decode-prefill-headline-reconciliation-result-20260621.md` (headline
authority) · `docs/llama-decode-primitive-difference-audit-result-20260621.md` (corrected decode primitive) ·
`docs/decode-vector-flash-tile-realigned-result-20260621.md` (bounded vector-tile rested: FLASH_L=64 local-pass /
W==D-fail) · `docs/project-north-star-llama-and-lifecycle-search-20260620.md` (the only remaining decode lever).
There is **no funded bounded decode build** — bounded decode is rested.

**Optional owner knob (not promoted):** `FLASH_L=64` is a measured, byte-exact ~2% short-context decode gain
(+2.8%@512, +1.8%@1024) that **regresses −1.2%@4096** and is below the ≥5% promotion bar. It may remain a
research/owner-call knob for short-context-only use; it is **not** a default and **not** a bounded build to
pursue.

## Consistency guardrail
Run `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py` — it fails if a canonical doc
re-opens a closed question (bare `87.6`, an open `PREFILL_V2=auto` owner call, "flip global PREFILL_V2=auto",
"decode headline 87", or bounded decode fusion as current work).
