# bench/ — benchmark results index

Single source for the fork's current benchmarks: each number, where it's recorded, and the exact
command to reproduce it. Most `bench/**` output is gitignored (regeneratable); durable result
artifacts are force-added. Doc map: `../docs/README.md`; **canonical current state: `../docs/current-project-state-handoff-20260621.md`**.

> **Canonical policy (do not re-open — guardrail `extra/qk_policy_consistency_check.py`):** Global `PREFILL_V2`
> default stays **OFF**; `PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1` / q8 are **opt-in**. `87.6` is contextual
> (ctx≈0 tok/s vs a separate ctx4096 ms/token) — **never quote it bare**. The decode headline is the **curve**
> (~86 @ctx≈0 → ~61 @ctx4096 = **~67% llama**), not ctx≈0. **Bounded decode work is RESTED** — fusion, micro-fusion,
> and the vector-tile (FLASH_L=64 passed the local attention gate but failed W==D promotion) are all closed. The only
> remaining decode lever is the **north-star full `flash_attn_tile` lifecycle** (efficient many-split / stream-k
> combine), not a bounded patch. See `../docs/current-project-state-handoff-20260621.md`.
>
> **Machine-search evaluator** (`extra/qk_decode_eval.py`, `qk-decode-eval/`): the automated lifecycle ladder that
> turns a candidate into a schema'd verdict; reproduces the historical classifications and proved whole-decode W==D
> auto-clock variance <0.6% ≪ 5% margin. Measurement-only, no defaults changed.
>
> **Lifecycle-search loop v0** (`extra/qk_lifecycle_search_loop.py`, `qk-lifecycle-search/search_*.json`): the first
> closed generate→evaluate→prune loop on the evaluator — runs valid candidates through `decode_eval` and prunes
> closed-lane / default-promotion candidates before benchmarking. No kernels, no defaults; propose-only ledger.
>
> **Candidate-template generation v0** (`extra/qk_candidate_template_gen.py`, `qk-lifecycle-search/templates.json`,
> `generated/`): the 'generate' step — expands 4 templates into legal decode candidate specs (policy metadata,
> deterministic) consumed by the loop via `--candidates`. Specs only, no kernels/flags/defaults.
>
> **North-star evaluator-binding templates v0** (`qk-decode-eval/binding_templates.json` + `binding_template_schema.json`):
> the contract an executable `flash_attn_tile` candidate must satisfy vs `gqa_coop_vec` (comparator, T=1-parallelism
> artifact fields, local-A/B + W==D runners, no-WMMA, gates, stop conditions). `gen_north_star_flash_attn_tile` is now
> a precise `PRUNE_NEEDS_TEMPLATE`; the loop distinguishes missing-template / no-runner / executable. No kernel.
>
> **North-star flash_attn_tile EXECUTED + REFUTED** (`extra/qk_north_star_flash_attn_tile_ab.py` +
> `qk_north_star_dispatch_probe.py`, `qk-north-star-flash-attn-tile/`): first real attempt vs `gqa_coop_vec` →
> `FAIL_LOCAL_AB` (throughput 0.46/0.52/0.87× @ctx512/1024/4096, byte-exact), no W==D route. A throughput probe
> CORRECTS the diagnosis: the combine is negligible (pout ~1 µs); the ceiling is the **cooperative-dot q·k partial**
> (flat ~163 µs vs coop's scaling 75–144 µs). coop's matmul q·k is near-optimal for tinygrad; bounded tile lever
> exhausted → `REDESIGN_AUDIT_POINTS_TO_CODEGEN_DATAFLOW`. Refutation banked.
>
> **Next decode move scoped** (`docs/decode-codegen-dataflow-capability-scope-20260621.md`):
> `CODEGEN_SCOPE_LLAMA_ORACLE_FIRST` — port llama's `fattn-tile.cuh` (source on disk) as a non-default reference
> oracle, measure standalone **throughput** vs `gqa_coop_vec` (first gate ≥1.05× @ctx1024) through the existing
> `ab_script` binding, resolving standalone-kernel-codegen vs in-model-integration before any multi-week linearizer work.
>
> **Llama oracle ANSWERED the question** (`extra/qk_llama_flash_attn_tile_oracle_ab.py`, `qk-llama-flash-attn-tile-oracle/`):
> pure-GPU-time A/B (llama rocprofv3 vs coop ProfileGraphEvent) → **llama 5.87/5.71/4.77× faster than coop STANDALONE**
> @ctx512/1024/4096 → `LLAMA_ORACLE_LOCAL_AB_PASS`. The 10× gap is standalone kernel-codegen, not just in-model. Native
> codegen justified; llama is the target. `reference_oracle` decode_eval candidate (non-promotable).
>
> **Fused-flash expressiveness REFUTED-as-wall** (`docs/native-fused-flash-linearizer-scope-20260621.md`): a coupled
> online-softmax+V fused decode kernel runs value-correct in ONE kernel TODAY (existing `UOp.set`/`.after` idiom) — no
> `spec.py`/linearizer change. Next = bounded Path-A build (coop matmul q·k + fused softmax+V), A/B vs `gqa_coop_vec`;
> the 5–6× gap is in-kernel-q·k codegen QUALITY (deep). `NATIVE_FLASH_LINEARIZER_SCOPE_READY`.
>
> **Path A fused softmax+V tail REFUTED** (`extra/qk_fused_softmax_v_tail_ab.py`, `qk-fused-softmax-v-tail/`):
> value-correct but **0.725×@1024 / 0.876×@4096** vs `gqa_coop_vec` → `FUSED_SOFTMAX_V_TAIL_FAIL_LOCAL_AB`. Fusing exp
> into the partial makes W=129 lanes recompute exp (vs coop's once/key) → loses; coop's hoisted-exp split near-optimal.
> Full online-max removal `BLOCKED_BY_IDIOM` (two-granularity store). No W==D; banked.
>
> **Decode frontier decision** (`docs/decode-frontier-decision-after-path-a-20260621.md`): per-kernel breakdown shows
> coop's matmul q·k (13.9µs) ≈ llama's whole 12µs tile (q·k NOT the bottleneck); the gap is the softmax+V multi-kernel.
> `FRONTIER_LOW_LEVEL_TOOLING_FIRST` — diagnostic counters + ISA disasm of `flash_partial` vs llama tile before any
> codegen project. Bounded decode exhausted.
>
> **Low-level attribution DONE** (`qk-low-level-decode-attn-attribution/`, ISA/resources/occupancy):
> `LOW_LEVEL_ATTRIBUTION_FIXABLE_CODEGEN`. Rules out occupancy/spills (tinygrad at 100% occupancy, ≤13 VGPR, 0
> spills); root cause = `flash_partial` (PV) emits scalar loads / **0 `v_dot2` / 0 LDS** vs llama's LDS-tiled `v_dot2`
> fused tile. Fixable lever = route the PV through tiled-matmul codegen (W==D-marginal; full win = deep fused-flash
> codegen). Counters tooling-opaque (rocprof-compute broken, rocprofv3 blind to HCQ); binaries sufficed.
>
> **Matmul-PV diagnostic — `MATMUL_PV_BLOCKED_BY_LAYOUT`** (gate `FAIL_LOCAL_AB`; `extra/qk_matmul_pv_diagnostic_ab.py`,
> `qk-matmul-pv-diagnostic/`): the **split-preserving** tiled matmul PV (K=L=128 concrete, Hkv·Smax=256 wg) TILES at
> **~1078 GFLOPS** and **WINS 1.13×@ctx4096** — the ISA lever is CONFIRMED — but loses ctx1024/512 (0.94/0.88×)
> because tinygrad can't express a symbolic-count tiled batched matmul, forcing concrete Smax=32 (full-MAXC reads).
> Corrects an earlier non-split form (~50 GFLOPS, parallelism-collapsed) that wrongly blamed skinny M=4. Bounded
> decode + the ISA-named codegen lever both exhausted (need symbolic-count tiled matmul / deep fused-flash).
>
> **Post-Matmul-PV STRATEGY — `STRATEGY_RECOMMEND_FULL_FUSED_FLASH`** (`docs/post-matmul-pv-decode-strategic-scope-20260621.md`):
> exhausts the 3 remaining options. Next project = `POST_MATMUL_PV_FULL_FUSED_FLASH` (gate-first); symbolic-count tiled
> matmul is dominated (W==D-marginal + sub-capability of fused-flash); rest+v2 is premature until the cheap fused-flash
> first gate (concrete-ctx1024 toy LDS-tiled fused µkernel, ≥1.05× vs `gqa_coop_vec`) is run — hard-stop fallback to
> REST_DECODE+v2 if it fails.
>
> **Fused-flash CONCRETE GATE — EXECUTED + FAILED → REST_DECODE+v2** (`extra/qk_fused_flash_concrete_gate_ab.py`,
> `qk-fused-flash-concrete-gate/`; `docs/fused-flash-concrete-gate-result-20260621.md`):
> `FUSED_FLASH_CONCRETE_GATE_FAIL_LOCAL_AB`. The literature-grounded (FlashAttention/Flash-Decoding/FlashDecoding++/
> FlashInfer) concrete ctx1024 flash-decode pipeline (q·k AND PV ride the tiled-GEMM codegen; S=8 FAIR splits — the
> matmul-PV symbolic-split blocker REMOVED by fixing the shape) is value-correct (rel_rmse 4.9e-4) but **0.965×@ctx1024**
> vs the strict same-shape **concrete** `gqa_coop_vec` (the 1.42× vs the SYMBOLIC comparator is a concreteness artifact,
> not a win). tinygrad renders the decode-shape matmuls **register-tiled (16 wg, 305 GFLOPS, no LDS, no `v_dot2`)**, not
> llama's one-kernel LDS-staged `v_dot2` tile; flat-GEMM under-utilization + 2 extra layout kernels offset the benefit.
> The true single fused LDS-tiled kernel is inexpressible (tiled-GEMM ⊥ `.set/.after` fusion). decode_eval
> `fused_flash_concrete_gate` → `FAIL_LOCAL_AB` (match=True). **Bounded AND concrete-shape decode levers both exhausted.**
>
> **Active-surface reduction** (`bench/qk-active-surface-reduction/`; `docs/perf-probe-active-surface-reduction-result-20260621.md`):
> reference-graph inventory of all perf scripts. Deleted **243 in two waves** (wave 1 = 26 zero-reference scratch;
> wave 2 = 217 owner-approved dated-doc-cited provenance probes, 62 rescued as real test/lib deps via the
> import-safety fixpoint + comprehensive import detection). `extra/qk_*.py` **376 → 133**. Live evaluator/search
> verified intact after both (policy guard PASS; all CLIs; 35 test modules import clean; tinygrad/model untouched;
> independent all-styles whole-repo check = every importer of a deleted script is itself deleted). `build_inventory.py`
> reproduces the classification; final record in `inventory.json`. `ACTIVE_SURFACE_REDUCTION_DELETE_COMPLETE`.
>
> **Harness evaluator-contract audit + applied** (`extra/qk_harness_contract.py`; `docs/harness-contract-audit-20260621.md`):
> audited the live lifecycle harness set against the 13-field contract ("Harnesses Are Performance Primitives Too").
> Systemic gaps = no spread/noise band, no git/dirty stamp, prose-only comparator-why; two distortion risks = the
> llama-oracle hardcodes ctx512/4096 from constants (now provenance-disclosed) and `qk_gateup_sched_ab.py` emits no
> artifact / isn't clock-pinned. Applied: centralized `qk_harness_contract` (provenance + `repro_band` +
> `contract_audit` + `stamp`); `decode_eval` now auto-flags any non-conforming child artifact (`child_artifact_contract`
> + `HARNESS-CONTRACT` note); `qk_fused_flash_concrete_gate_ab.py` upgraded to **CONFORMS 13/13** (reference). No
> verdict/default changed.

**Setup (all commands):** `cd /home/ubuntu/tinygrad-arkey`, interpreter `.venv/bin/python`, `DEV=AMD`,
RX 7900 XTX (gfx1100), models at `/home/ubuntu/models/`. Bar: **llama.cpp ≈ 98–106 tok/s** (8B decode,
depending ctx/harness) and **~3020 pp512 tok/s** (8B prefill).
Measurement discipline (the 3 confounds: cache / launch / clock-ramp) — `../docs/amd-decode-measurement-confounds.md`.

## Which harness for decode tok/s — READ FIRST (don't repeat the 2026-06-18 mistake)

Decode tok/s is only trustworthy from a **clean `model.generate`-class path** (device-token feedback, **no
per-step host `Tensor` creation** — that artifact *halves* the rate; `../docs/qk-runtime-overhead-arc-result-20260617.md`).
Pick the harness by what you're measuring:

| want | use | gives | notes |
|---|---|---|---|
| **production tok/s @ ctx≈0** (default headline) | **`-m tinygrad.llm … --warmup --benchmark`** (`tinygrad/llm/cli.py`) | ~86 tok/s, single point | the production default; `model.generate`, clean path |
| **decode tok/s vs ctx** (512/1024/4096…) | **`extra/qk_decode_runtime_overhead.py`** (in-model **W==D**) | 68.2/66.4/60.7 @ 512/1024/4096 | only harness that sweeps ctx on the clean path; W≈D ⇒ GPU-bound |
| flash-decode **policy** (which path is selected, off/on/auto) | `extra/qk_flash_decode_auto_bench.py` | ~54–56 flat | ⚠️ **NOT a tok/s number** — it builds a host input `Tensor` per step inside the timed loop (contaminated by design); use it for selection/correctness only |

Both clean harnesses agree on one curve: **~86 @ctx≈0 → 68.4/66.9/61.2 @ 512/1024/4096** (default stack,
rerun 2026-06-20; matches banked). See `../docs/qk-decode-banked-reproduce-20260618.md`,
`qk-decode-runtime-overhead/result.json`.
> ⚠️ **`87.6` is ambiguous — never quote it bare.** It is BOTH a real ctx≈0 decode **tok/s** (~11.4 ms) AND a real
> ctx4096 decode **ms/token** (=11.4 tok/s). The decode headline is the *curve* (~86 @ctx≈0 → ~61 @ctx4096 = **~67%
> llama** steady-state), not the ctx≈0 peak. Prefill policy (`PREFILL_V2=auto`/server) does NOT change decode (<1%,
> identical output). See `../docs/decode-prefill-headline-reconciliation-result-20260621.md`.

## Current numbers

| benchmark | value | recorded in | reproduce |
|---|---:|---|---|
| **Decode 8B, default-on** (coop stack + gqa_coop_vec + flash) | **~86 @ctx≈0; 68.4/66.9/61.2 @ctx 512/1024/4096** (~67% llama) | `bench/qk-decode-runtime-overhead/result.json`, `qk-decode-banked-reproduce-20260618.md`, `qk-8b-decode-banked-20260617.md` | CLI `--warmup --benchmark` (ctx≈0); `extra/qk_decode_runtime_overhead.py` (ctx sweep) — see "Which harness" above |
| **Decode 8B, q8 FFN research route** (default-off / opt-in) | **72.9/71.1 tok/s @ctx 512/1024** in `auto`; median `~72.0`, `~1.064×`, host-sync `0.0%` | `bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_rerun_20260620.json`, `docs/decode-q8-model-route-timing-audit-result-20260620.md` | `PYTHONPATH=. .venv/bin/python extra/qk_decode_q8_model_route_timing_audit.py --lanes auto,manual_peak --modes baseline,q8 --ckpts 512 1024 --nmeas 20 --warmups 8 --out bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_rerun_20260620.json` |
| _superseded:_ Decode 8B pre-coop (~55) / +ffn_down demote (64.3) | historical | `amd-decode-banked-20260616.md`, `bench/qk-demote-search/search.json` | folded into the default-on stack above |
| **Demotion frontier** (ffn_down/attn_v accept; lm_head 75.0 but rejected on dNLL +0.051) | see json | **`bench/qk-demote-search/search.json`** (+ `accepted-*.json`) | `python -m extra.qk_demote_search --epsilon 0.01` |
| **Decode 14B** (generated policy) | 40.6 tok/s (62%) | **`bench/qk-shared-storage-20260612/matrix-summary.md`** | harness; see that dir's README |
| **Decode 32B** (generated policy) | 17.2 tok/s (56%) | same matrix-summary | same |
| **Standalone Q4_K GEMV** (int-dot) | **76% of HBM peak** (vs llama 57%) | `amd-decode-capstone.md`, memory `amd-decode-kernel-beats-llamacpp` (raw artifact gitignored) | `extra/q4_k_bench` cold/full-clock; see capstone |
| **Flash long-context** (ctx 3072) | 9.4 → **22.7 tok/s (2.41×)** | `amd-decode-flash-attention-plan.md` (SHIPPED section), memory | `FLASH_DECODE=1` decode at long ctx; `test/external/test_qk_flash_decode.py` |
| **Sequential-tax split** | GEMV 72% / non-GEMV 28% | `amd-decode-sequential-tax-profile-*.md` | `DEBUG=2 JIT_BATCH_SIZE=1 cli --benchmark 8` → strip ANSI → `python -m extra.q4_k_profile_report` |
| **Decode dNLL (quality gate)** | baseline 2.779; ffn_down +0.0005 | `amd-decode-demotion-search-*.md` | `python -m extra.qk_nll_eval --model <gguf> --tokens 128` (±0.01 calib noise) |
| **Overlap feasibility (probe)** | 1.0× (gated: one compute ring) | `amd-decode-two-queue-probe-*.md` | `python -m extra.qk_two_queue_probe` |
| _superseded:_ **Prefill 8B old baseline** | ~67 tok/s (**~2% llama**) — obsolete/outlier | `amd-decode-prefill-plan.md` | replaced by PREFILL_V2 graph route + Branch B + Increment 0 below |
| **Prefill 8B, Branch B concrete first chunk** (`PREFILL_TC_ATTN`, default-on under `PREFILL_V2`/gfx1100) | **3394 pp512 tok/s** (`112.7%` llama) for concrete start_pos=0; byte-identical; no WMMA fired, win is fusion | `bench/qk-prefill-tc-attention/concrete_gate_result.json`, `docs/prefill-branch-b-tc-attention-result-20260620.md` | `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_tc_attn_concrete_gate.py` |
| **Prefill 8B, Increment 0 concrete-KV** (`PREFILL_V2=1 PREFILL_CONCRETE_KV=1`, opt-in) | per-chunk forward **1.7–4.4×** faster than symbolic; **73–111% llama** across KV 512–3584; rerun A2 warm prefill **4941→343 ms** when both chunks are concrete; rerun A1 first-gen prefill **9.41→3.44 s** after precompile-at-load; byte-identical tok0 | `docs/prefill-concrete-kv-increment0-result-20260620.md`, `docs/prefill-increment0-shipped-result-20260620.md` | `DEV=AMD PREFILL_V2=1 [PREFILL_CONCRETE_KV=1] PYTHONPATH=. .venv/bin/python extra/qk_prefill_concrete_kv_a1_precompile_test.py`; same for `extra/qk_prefill_concrete_kv_a2_verify.py` |
| **Prefill flash Increment 2** (not shipped) | correct (`rel_rmse ~1e-7`) but **~15× too slow** at KV512 and worse at long KV | `docs/prefill-flash-increment2-result-20260620.md` | `extra/qk_prefill_flash.py`; `extra/qk_prefill_flash_perf.py` |
| **Prefill POLICY profiles** (shipped, gated) | default (slow long prompts) / **`PREFILL_V2=auto`** (24GB+, ~5–15× faster, VRAM-gated) / **`PREFILL_SERVER_PROFILE=1`** (best warm prefill 0.17–1.6s). `PREFILL_REMAINDER_FIX` default-on kills the 32-token trap (up to 14× on prefix-cache resume, byte-identical) | `docs/prefill-policy-integration-result-20260620.md`, `bench/qk-prefill-policy-integration/*.json` | `extra/qk_prefill_{v2_auto_policy,concrete_kv_policy,route_schedule}_probe.py` |
| **Prefill external BLAS ceiling** (standalone fp16 GEMM control, not routed) | ffn_gate/up: hipBLASLt **69.8 TFLOPS** (1.71× tinygrad); ffn_down: rocBLAS **70.9**; attn_q/o: rocBLAS **76.7** | **`bench/qk-prefill-external-blas/ceiling.json`**, `docs/prefill-external-blas-result-20260619.md` | `g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 ... extra/qk_prefill_blas_ceiling.cpp`; see result doc |
| **Prefill pure tinygrad WMMA sweep** (not routed) | best **42.0 TFLOPS** (34% peak), gate was 62 TFLOPS; more waves/bigger tiles/BK32/noLDS regress | **`bench/qk-prefill-own-wmma/sweep.txt`**, `docs/prefill-own-wmma-kernel-result-20260619.md` | `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_wmma_sweep.py` |

## Reproduce — the two most-cited

```sh
# Decode 8B, default-on @ctx≈0 (production headline) -> ~86 tok/s steady median (drop first ~3, clock-ramp)
DEV=AMD PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 40

# Decode 8B vs ctx (the banked curve) -> 68.4/66.9/61.2 @ 512/1024/4096, in-model W==D, host-sync %
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py

# Prefill 8B concrete-KV opt-in path -> precompile concrete prefill jits at load, then byte-identical faster prefill
DEV=AMD PREFILL_V2=1 PREFILL_CONCRETE_KV=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 1

# The full demotion search (frontier + accepted policies) -> writes bench/qk-demote-search/
DEV=AMD .venv/bin/python -m extra.qk_demote_search --epsilon 0.01 --bench 24 --tokens 128
```

## Notes on the record

- **Committed (raw artifacts):** `qk-demote-search/` (this arc's search), `qk-shared-storage-20260612/`
  (8B/14B/32B matrix + policies). Reproducible from the JSON.
- **Doc-only (artifact gitignored by the prune, regeneratable):** the standalone-kernel 76%, the flash
  ctx-sweep, the older prefill sweep, and some prefill Increment 0 harness outputs — numbers are in the cited
  docs + committed scripts; rerun the command to regenerate.
- **dNLL numbers carry ~±0.01 noise** (128-token calib set) — directionally trustworthy (the gate
  correctly rejected lm_head at +0.051), not precise to 3 decimals.
- Decode tok/s is the **steady-state median** with first ~3 tokens dropped (clock-ramp); the cold first
  token (~100 ms) is not the benchmark.
