# Decode Codegen/Dataflow Capability Scope

Date: 2026-06-21

Scope-only project (no kernels, no model change). Decide the next large move after the north-star bounded-tile
redesign audit (`docs/north-star-decode-attention-redesign-audit-20260621.md`): native tinygrad codegen (A) vs a
llama source-port reference oracle (B) vs rest (C).

## Decision: **`CODEGEN_SCOPE_LLAMA_ORACLE_FIRST`**

Port llama's `flash_attn_tile` as a **non-default reference oracle**, measure it **standalone vs `gqa_coop_vec`**
(throughput, apples-to-apples), and let that one cheap experiment resolve the central unresolved question — **is
llama's 10× decode-attention advantage a standalone kernel-codegen win or an in-model integration win?** — before
committing to a multi-week native-codegen project that currently has **no validated target**.

## Phase 0 — evidence reconciliation (replaces the "HBM-bound combine" hypothesis)

| fact | value | source |
|---|---|---|
| gqa_coop_vec attention (throughput) | 75 / 85 / 144 µs @ctx512/1024/4096 — **scales with ctx** | `qk_north_star_dispatch_probe.py` |
| failed north-star tile (latency A/B) | 0.53 / 0.59 / 0.71× | execution result |
| failed north-star tile (throughput) | **0.46 / 0.52 / 0.87×**, **flat ~163 µs** | dispatch probe |
| q·k partial floor | ~163 µs flat (latency/occupancy-bound: 512 small wg + LDS-load + barrier + ds_bpermute) | redesign audit |
| dispatch sensitivity | candidate = 2 un-batched raw dispatches; coop = batched JIT graph → the latency "combine cost" was 2nd-dispatch overhead | dispatch probe |
| combine traffic (pout) | ~1 MB → **~1 µs at HBM peak — negligible** | traffic accounting |
| ~~"combine is HBM-bandwidth-bound"~~ | **REFUTED** (S-invariant, *decreased* with ctx) | redesign audit |
| llama decode timing | **IN-MODEL only** (`decode_ms_per_tok 9.97 ms` whole-model @ctx1024; attention_total 507 µs/tok over all layers); **standalone-kernel time UNKNOWN** | `bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json` |
| current conclusion | the ceiling is the q·k partial / codegen quality; **coop's matmul q·k is near-optimal for tinygrad primitives**; bounded tiles closed | redesign audit + refutation |

**Open question this scope targets:** we have never measured llama's *actual* kernel **standalone** vs coop. The
9.2µs/layer figure is not even in the artifacts (the audit numbers are whole-model per-token). So "the 10× gap is
kernel codegen" is **inferred, not measured** — my hand-written fused tile (a structural mirror) failed, but that is
*my* codegen. The oracle measures the real thing.

## Phase 1 — capability gap map

| llama requirement | tinygrad current support | gap type | first file/component | risk | possible first gate |
|---|---|---|---|---|---|
| many KV-split blocks, **grow with ctx** (48→144) | gqa_coop_vec fixed `S=ceil(ctx/128)` (4–32), occupancy-starved | dataflow/scheduling | `qk_flash_decode.py:271` | low | tile with parallel_blocks ≫ 8 |
| GQA/query-head column packing | coop packs G=4 in register accumulators (`flash_partial_coop_vec`) | **already supported** | `qk_flash_decode.py:195-219` | — | n/a |
| **q·k mapping quality (single fused kernel)** | coop uses a **separate optimized matmul** (near-optimal); a *fused* single-kernel q·k is **NOT expressible** | **codegen/linearizer** | `spec.py:163-165`, `linearizer.py:54-82` | **HIGH (multi-week)** | linearizer emits coupled-accumulator flash OR a microkernel matches q·k throughput |
| K/V LDS staging | raw-C tiles do it (`warp_tile_src`); coop's matmul reads K directly | renderer (raw-C yes; UOp LDS-stage in a fused reduce no) | `qk_flash_decode.py` warp src | medium | LDS-staged fused kernel |
| **register online-softmax `(m,l,acc)` coupled** | **NOT a `REDUCE`** (single-op contract); hand-rolled via `UOp.set`/REG placeholders → forces the **6-kernel split** | **codegen/linearizer (THE wall)** | `spec.py:163-165`, `ops.py:1064`, `q8-mmvq-lifecycle-deep-result-20260619.md` | **HIGH** | the linearizer capability (`flash_fused_multireduce_linearizer_wall`) |
| in-kernel V accumulation | coop does it (separate `flash_partial` kernel) | already (split) | `qk_flash_decode.py:195` | low | n/a |
| efficient split combine / no bulky combine | coop's graph combine (gmax/den/combine) | **NOT the gap** (refuted: combine ~1 µs) | n/a | — | — |
| **graph/JIT integration** | coop = batched JIT graph (efficient dispatch); raw tiles = un-batched `dev.runtime` dispatches (slow) | runtime/HCQ | `qk_north_star_dispatch_probe.py` | medium | **throughput** (back-to-back) comparison, not per-call latency |
| candidate-template / evaluator binding | exists (`ab_script` → `classify`) | **none (only binding)** | `binding_templates.json`, `qk_decode_eval.py:196-198` | low | register the oracle candidate |

**Two HIGH-risk gaps are codegen/linearizer** (single-op `REDUCE`, shared-range store-group, CFGContext
sibling-ordering `linearizer.py:81` assert) — the project's own refutation `flash_fused_multireduce_linearizer_wall`
pre-classifies closing them as a **"multi-week linearizer project (BEAM-hang class), not a bounded build."** One gap
is a **runtime/dispatch confound** (raw vs JIT graph) that any standalone measurement must control for.

## Phase 2 — path ranking

| | expected value | complexity | advances north-star | uses evaluator/search | first gate | stop condition | files | one-off-artifact risk | why now / not now |
|---|---|---|---|---|---|---|---|---|---|
| **A. Native codegen/dataflow** | high (tinygrad-native primitive) **but unvalidated** — my fused tile already failed; no proof a fused kernel beats coop | **very high / multi-week** (linearizer surgery: `spec.py` REDUCE contract, store-group, CFGContext) | directly | only at the end | linearizer emits a coupled-accumulator LDS-reduction flash, or a microkernel matches coop q·k throughput | BEAM-hang / no win after weeks | `tinygrad/codegen/late/linearizer.py`, `uop/spec.py`, `codegen/__init__.py` | low (it's tinygrad code) | **not now** — multi-week with no validated target; pre-refuted as not-bounded |
| **B. llama oracle port** ✅ | **high** — resolves standalone-vs-in-model; establishes the real target + a correctness oracle; informs A | **medium** — source on disk, bridge proven | indirectly (oracle, not a primitive) | yes (`ab_script` gate) | **standalone llama-tile throughput ≥1.05× vs gqa_coop_vec @ctx1024** | port can't be extracted standalone (ggml too entangled) → fall to A/C | `extra/qk_llama_flash_tile_oracle_ab.py`, `binding_templates.json` | **managed**: reference-only, non-default, isolated | **now** — cheap, de-risks A, answers the open question |
| **C. Rest** | low — leaves the central question unmeasured | none | no | no | n/a | — | — | — | not now — B has a credible bounded first gate |

## Phase 3 — chosen scope: B (llama reference oracle)

### Why this path wins
The native-codegen project is multi-week and, per the project's own refutation, **not bounded** — and critically it
has **no validated standalone target**: my hand-written fused tile failed, so "a fused flash kernel beats coop" is
**unproven**. The llama oracle is the cheap experiment that either (i) proves a standalone flash kernel *can* beat
coop (→ native codegen gets a real target + a byte-level correctness oracle), or (ii) shows llama's standalone
kernel is ~parity with coop (→ the 10× is **in-model integration**, and native-codegen-of-a-standalone-tile is the
*wrong* project — redirect to in-model dataflow / the W==D frame). Either outcome is decisive and saves weeks.

### Execution scope (the NEXT project builds this; here is its contract)

**Source to bridge** (MIT-licensed; reference only): `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh`
(`flash_attn_tile_load_tile` LDS staging, `flash_attn_tile_iter_KQ` vector-FMA q·k, `KQ_max` register online
softmax) + the Hd=128 instance `template-instances/fattn-tile-instance-dkq128-dv128.cu`. Extract the tile kernel to
**standalone HIP** with the Qwen3-8B decode shape hardcoded (Hd=128, Hq=32, Hkv=8, T=1), stripped of ggml template
machinery and the launcher.

**Bridge (proven, reuse the warp-tile path):** `dev.runtime(name, dev.compiler.compile(hip_src))` +
`Buffer("AMD", n, dt).ensure_allocated()` + `prg(*._buf, global_size=, local_size=, vals=)` + `copyin/copyout`.
**One HSACO per kernel** (the multi-kernel-lib trap → silent MMU fault); if llama needs tile + combine + streamk-fixup
kernels, compile each separately.

**Local A/B gate (the first executable gate):** `extra/qk_llama_flash_tile_oracle_ab.py` compares the standalone
llama tile vs `gqa_coop_vec` at ctx512/1024/4096. **Measure THROUGHPUT** (back-to-back, one final sync — reuse
`qk_north_star_dispatch_probe.py`'s method) as the primary metric, NOT per-call latency, to control the raw-vs-JIT
dispatch confound; also report latency for context. Gate: **≥1.05× @ctx1024 AND no regress @ctx4096** (throughput).

**Correctness gate:** vs the numpy/`ref_attn` reference (and the coop output). llama uses fp16/fp32 mixed accumulation
→ require **rel_rmse ≤ 1e-3, max_abs documented** (byte-exact unlikely; document the tolerance).

**Artifact:** `bench/qk-llama-flash-tile-oracle/local_ab_<ts>.json` with the binding's 9 fields (`workgroups_by_ctx`,
`kv_splits_by_ctx`, `query_heads_parallelized`, `combine_kernel_count`, `local_attention_us_by_ctx`,
`comparator_attention_us_by_ctx`, `correctness_error`, `reproducibility_band_pct`, + `throughput_*`).

**Lifecycle/decode_eval binding:** new binding `llama_flash_attn_tile_oracle_v0` in `binding_templates.json`
(role `decode_attention`, comparator `gqa_coop_vec`, `local_ab_runner: ab_script` → the oracle harness,
`concrete_runner_status: implemented`, `is_reference_oracle: true`, `default_eligible: false`). Register a
decode_eval candidate `llama_flash_attn_tile_oracle` (family `reference_oracle`) and a generated candidate so it runs
**through lifecycle-search → decode_eval → `FAIL_LOCAL_AB` | `LOCAL_PASS_*`** like every other candidate. No new
plumbing (the `ab_script` path exists).

**Why reference oracle, not default:** it is **vendored llama code**, not a tinygrad primitive — it establishes the
*target* + a *correctness contract* + the standalone-vs-in-model answer. It must **never** become a default decode
route (the project north-star is tinygrad-native quality; precedent: prefill kept its vendored Tensile `.co`
strictly as an opt-in, not a default).

**How it informs native codegen (A):** if the oracle **wins standalone** (≥1.05×), we have (a) proof the fused-tile
structure beats coop, (b) the exact resource shape to target (LDS 10752 B, VGPR 128, wg 32×4, grid 32×16, KV-splits
grow with ctx), (c) a byte-level correctness oracle — then native codegen scopes against a *validated* target. If it
**ties/loses standalone**, native-codegen-of-a-standalone-tile is refuted and the north-star redirects to **in-model
dataflow integration** (fewer dispatches / fused decode block, measured at W==D), a different and better-targeted
project.

**Stop conditions:** (1) the tile kernel cannot be extracted standalone within bounded effort (ggml template/runtime
entanglement) → bank `NEEDS_DEEPER_PORT`, reconsider A vs C; (2) it compiles but is incorrect (rel_rmse > 1e-3) → fix
or stop; (3) it runs and **loses to coop standalone** → bank the finding (the win is in-model, not the kernel),
redirect to in-model dataflow; (4) any closed lane would be reopened → stop. **No model route, no default, no W==D
route until the local A/B gate passes.**

### What would prove this path wrong
If the llama tile **cannot be extracted standalone** (too entangled with ggml) the oracle is infeasible → the path
is wrong and the choice collapses to A (multi-week, blind) or C (rest). If the port is feasible the path is *not*
wrong regardless of pass/fail — both outcomes are decisive.

## Phase 4 — docs/artifacts (this task)

This task: the scope doc + canonical updates. **No build, no `tinygrad/` change.** The build is the next project.

## Acceptance gates

| gate | result |
|---|---|
| G1 old combine-traffic diagnosis explicitly corrected | PASS (Phase 0 table) |
| G2 capability gap map names concrete missing components | PASS (spec.py:163-165, linearizer.py:54-82, store-group idiom) |
| G3 native / llama-port / rest ranked | PASS (Phase 2) |
| G4 chosen path has a first executable gate | PASS (standalone llama-tile throughput ≥1.05× @ctx1024) |
| G5 no closed bounded lane reopened | PASS (oracle is a new lane; not WMMA/MMVQ/bounded-tile) |
| G6 future executable uses lifecycle/evaluator | PASS (`ab_script` binding `llama_flash_attn_tile_oracle_v0`) |
| G7 no model/default/kernel code change | PASS (scope only) |
| G8 policy guard passes | PASS |
| G9 tree clean after commit | PASS |

## Stop condition (for the chosen path)
If the standalone port is infeasible, or it runs and ties/loses to coop, **do not pursue native codegen on a
standalone-tile premise** — that premise would be refuted. Native codegen is only justified *after* the oracle proves
a standalone fused kernel beats coop.

## Changed files
This doc; `docs/current-project-state-handoff-20260621.md`, `docs/README.md`, `bench/README.md` (pointers). No
`tinygrad/`, no binding/kernel build.

## Boundary
Scope only. No kernel, no model route/default, no closed lane reopened, no benchmarking of weak baselines (comparator
stays `gqa_coop_vec` + the llama reference). The oracle, when built, is non-default and evaluated through the
existing system.
