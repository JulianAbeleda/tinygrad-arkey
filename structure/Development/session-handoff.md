# Session Handoff

> ## ⭐ 2026-06-23 — LEARNING LAYER REFRAMED: the model is a primitive-space PROPOSER, not a kernel judge
> The learned-model/adapter role in the GPU primitive search system is: emit a **bounded search spec** (`SearchRow`:
> lane / primitive / hypothesis / knobs+bounds / required-evidence / stop-rules), which the **deterministic** runner
> expands and the harness / ISA / correctness / W==D-or-whole-prefill gates decide. **LoRA/SFT first** (structured
> supervised primitive-space generation); **RLVR DEFERRED** until schema + deterministic reward + shadow-mode utility
> are proven. Verdicts: `PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE`, `LORA_FIRST_FOR_PRIMITIVE_SPACE_LEARNING`,
> `RLVR_DEFERRED_UNTIL_SCHEMA_AND_REWARD_STABLE`, `DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`. Doc-update only (no
> training, no search, no source/default change). **Next executable task** = build
> `bench/qk-primitive-space-adapter/dataset-v0` + deterministic scorer `extra/qk_primitive_space_scorer.py`. Also today:
> **ORACLE-GUIDED GPU PRIMITIVE EXPLORER SCOPED** (oracle registry + shared spec + gate stack + ledger over the existing
> backends; unified runner DESIGN-only). Docs: `docs/primitive-space-learning-loop-lora-first-result-20260623.md`,
> `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`,
> `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`.

> ## ⭐⭐ SUPERSEDED 2026-06-23 — owned AMDGCN decode-attention is now the DEFAULT (default_on=true)
> The "Route B State" / "B4 W==D fails / no promotion from B4 / opt-in only" narrative below is **superseded**. The
> owned tile had a **dtype-contract bug** (read the fp32 cache as fp16) and an **over-conservative ctx guard** —
> both fixed — plus FO2 (native fp16 cache). It is now **real-cache byte-identical across the whole decode range**
> and the **DEFAULT** decode attention for gfx1100 / Qwen3-8B / B=1 / T=1 (every other shape/device stays gqa+fp32;
> `DECODE_ATTN_AMDGCN_TILE=0` disables). Canonical W==D harness (real decode tok/s) confirms
> **+12.7/+15.4/+18.7/+22.4% @ctx512/1024/2048/4096** → default decode @ctx1024 ~74→~85 tok/s (~76%→~88% of
> llama.cpp). Candidate `decode_attention_llama_flash_tile_owned_amdgcn_b4`: **`default_eligible=true`,
> `default_on=true`**. Runtime-KV is **deferred (incremental)**. The "attention exhausted / B4/B5 sub-bar /
> runtime-KV next" framing is no longer current.
> Authority: MODE B + PREFILL SEARCH EXECUTED (2026-06-23). Mode B: DECODE_MODE_B_EXECUTED_ORACLE_REMAINS_BEST (14 variants, tile constants optimal, additive templating byte-identical). Prefill: CORRECTION -- clock-pinned repeats show a STABLE ~4-5% gap to Tensile (not the noise '99.5%'); Phase A READY, Phase B tile-config search finds NO recovering config (gap is K-loop SCHEDULING not tile-config) -> ORACLE_REMAINS_BEST. `docs/{decode-mode-b-search,prefill-search}-result-20260623.md`. Prior: NATIVE-CODEGEN MICROSEARCH EXECUTED (2026-06-23): TARGET_FOUND -- LDS+vector-loads native, v_dot2+cross-lane(ds_bpermute) are the confirmed renderer gaps (ISA-evidenced, 5 correct candidates). `docs/native-codegen-microprimitive-search-result-20260623.md`. Prior: PROJECT MACHINE-SEARCH ROADMAP READY (2026-06-23): repo is search-capable. Decode search DONE (oracle best); PROJECT SEARCH LEDGER built (extra/qk_project_search_ledger.py, 9 entries); prefill search GATED (at-rest); native-codegen microsearch SCOPED+ALLOWED-NOW (targets v_dot2/cross-lane); cross-shape DEFERRED (14B owner-gated). `docs/project-wide-machine-search-roadmap-result-20260623.md`. Prior: DECODE SEARCH EXECUTED (Mode A, 2026-06-23): DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST -- 5/6 PASS gates, none beat oracle outside spread (default S48/base is policy-optimal), W==D-only, artifacts CONFORMS 13/13, no default flip. `docs/decode-machine-search-execution-result-20260623.md`. Prior: DECODE MACHINE-SEARCH READINESS PACKAGE READY (2026-06-23): froze buffer-identity default as oracle (W==D 90.6/89.3), built gate+checkers+runner (extra/qk_decode_search_*.py), SEARCH_RUNNER_READY smoke (oracle PASS, bad REJECTED). Decode NOT worth searching for 8B speed; READY for regression-safe/cross-shape/codegen/portability. `docs/decode-machine-search-readiness-package-result-20260623.md`. Prior: PREFILL AT PARITY (2026-06-23): synced whole-prefill RETIRES the 66% headline -- graph-GEMM ~96-99.5% of Tensile, at/above llama; shipped kv_proj de-WG-starve fix (BN64 small-N, +3-4% whole-prefill, dependency-free, byte-identical). Prefill AT REST. Tool extra/qk_prefill_whole_synced.py. Prior `docs/prefill-per-role-transfer-attribution-result-20260623.md` (PER-ROLE: graph-GEMM within 2.5% of Tensile on concrete chunk; gap = small-N WG-starvation kv_proj 34%; bounded fix = per-shape config NOT search; whole multi-chunk axis TBD) + `docs/prefill-post-decode-parity-frontier-result-20260623.md` (PREFILL FRONTIER AUDIT: kernel at Tensile parity, MACHINE_SEARCH_NOT_READY; next lever = in-model integration penalty 66%->87% via synced per-role time-tax, NON-search; see `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`) + `docs/decode-campaign-final-synthesis-20260623.md` (⭐⭐DECODE CAMPAIGN COMPLETE: tinygrad decode 102-105% of llama.cpp default-on; whole-cache buffer-identity KV read; runtime-KV lane retired; POST_PARITY_REGRESSION_GUARD_PASS) + `docs/machine-code-translation-roadmap-result-20260623.md` (machine-code map, buffer-identity ABI rule in principles #12, search NOT_READY_FOR_8B_SPEED). Prior: `docs/owned-tile-buffer-identity-kv-read-result-20260623.md` (⭐SHIPPED default-off: buffer-identity whole-cache read = +13-19% BYTE-IDENTICAL, removes E_49152 slice materialization; tinygrad decode now 102-105% of llama.cpp; DEFAULT-ON 2026-06-23 owner-authorized; DECODE_ATTN_KV_IDENTITY=0 disables), `docs/runtime-kv-core-engine-result-v2-20260623.md` (MAJOR CORRECTION: runtime-KV correctness IS achievable via native-store+AFTER-read — 64-tok byte-identical; the callify hard-stop was the opaque-append only. The +11% materialization lever is the owned tile SLICING the cache -> fix = bounded buffer-identity whole-buffer read, NOT a core-engine project). SUPERSEDES `docs/runtime-kv-core-engine-result-20260623.md` (FINAL runtime-KV: RUNTIME_KV_CORE_CAPABILITY_BLOCKED by the callify/pure-function execution model — toy passes, one-layer NaN; fix = tinygrad-core Tensor-purity change, scope hard-stop; 8B decode complete at bounded layer ~88-89% of llama), `docs/three-lane-completion-result-20260623.md` (lanes COMPLETED: ISA wrapper + native-codegen experiment DONE; runtime-KV RUNTIME_KV_CORE_CAPABILITY_BLOCKED — bakes at 1 layer, needs CORE TinyJit/HCQ engine work; 8B bounded speed exhausted at model layer), `docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md` (lanes 2/3/6 scoped: ISA wrapper BUILT/guard-active; runtime-KV core-persistence DESIGN_A scoped — needs owner auth; native-codegen chartered), `docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md` (ROADMAP: NEXT = one small-ops fusion gate; runtime-KV deferred owner-decision; machine-search parked; attention+GEMV closed), `docs/post-default-runtime-kv-diagnostic-result-20260623.md` (8B bounded-exhaustion checkpoint: attention+GEMV CLOSED at llama parity; KV-materialization +11.8% but CORE-RUNTIME-BLOCKED; small-ops overlapped; machine-search NOT yet justified), `docs/post-owned-attention-default-audit-result-20260623.md` (FRESH gap map: weight-GEMV+attention at/near llama parity; residual = KV-copy + small-op fusion; tinygrad ~85-88% of llama), `docs/owned-attention-default-flip-result-20260623.md`,
> `docs/post-owned-attention-promotion-synthesis-20260623.md`,
> `docs/owned-tile-post-promotion-four-step-result-20260623.md`,
> `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`,
> `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`.

Date: 2026-06-21

Repo: `/home/ubuntu/tinygrad-arkey`

Branch: `qk-prefill-flag-leak-resolution`

This is the current handoff. Older shared-storage, flywheel, and early decode notes were removed from this file because
they are now superseded by the doc map and provenance index. Use:

- `docs/README.md`
- `docs/current-project-state-handoff-20260621.md`
- `docs/provenance-index-20260621.md`

## Current Baseline

Target machine and model:

```text
GPU: RX 7900 XTX / gfx1100
model: Qwen3-8B-Q4_K_M.gguf
repo: /home/ubuntu/tinygrad-arkey
python: .venv/bin/python
device: DEV=AMD
```

Canonical default decode curve:

| ctx | default decode |
|---:|---:|
| ctx≈0 | ~85-86 tok/s |
| ctx512 | ~68 tok/s |
| ctx1024 | ~66 tok/s |
| ctx4096 | ~61 tok/s |

Current default policy:

- `PREFILL_V2`: default off
- q8 FFN: opt-in only
- B4 AMDGCN decode-attention route: **DEFAULT-ON** (2026-06-23) for gfx1100/Qwen3-8B/B=1/T=1; `DECODE_ATTN_AMDGCN_TILE=0` disables
- decode default attention is the owned AMDGCN route (fp16 cache); +12.7/+15.4/+18.7/+22.4% @ctx512/1024/2048/4096 vs gqa

## Read First

Current authority docs:

- `docs/current-project-state-handoff-20260621.md`
- `docs/README.md`
- `docs/decode-attention-route-b-b3-owned-amdgcn-result-20260621.md`
- `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
- `docs/b4-split-kv-combine-tax-result-20260621.md`
- `docs/b4-split-kv-combine-tax-scope-20260621.md`

Core artifacts:

- `bench/qk-decode-attention-route-b-b3/latest.json`
- `bench/qk-decode-attention-route-b-b4/latest.json`
- `bench/qk-decode-attention-route-b-b4-combine-tax/latest.json`
- `bench/qk-decode-eval/candidates.json`

## Route B State

The live frontier is decode attention Route B: an owned AMDGCN/HSACO escape hatch for the llama-style decode-attention
primitive.

| phase | result | meaning |
|---|---|---|
| B1 | `PASS_ORACLE_LOCAL_AB` | vendored llama `flash_attn_tile` runs through tinygrad HCQ and wins on GPU-busy time |
| B2 | `B2_LOCAL_GRAPH_PASS` | bound HCQ queue recovers raw-dispatch overhead; graph-style launch integration works |
| B3 | `B3_LOCAL_PASS` | owned hand-AMDGCN tile for tinygrad native KV layout beats `gqa_coop_vec` locally |
| B4 graph-node | capability pass | external precompiled AMDGCN `.co` enters TinyJit as `Ops.PROGRAM` nodes |
| B4 W==D | `B4_WD_FAIL_INTEGRATION` | whole-decode economics do not clear promotion |
| B4 combine tax | `COMBINE_TAX_DOMINATES` | split-KV combine is the fixable latency-bound floor; Amdahl co-limits |
| split-KV economics audit | `SPLIT_KV_ECONOMICS_AUDIT_READY` | permanent audit layer: split-KV candidates must report tile/combine economics before W==D |

## B3 Summary

B3 produced the first owned, promotable hand-AMDGCN decode-attention tile:

- source: `extra/qk_owned_flash_decode.hip`
- runner: `extra/qk_owned_flash_decode_amdgcn_b3.py`
- candidate: `decode_attention_llama_flash_tile_owned_amdgcn`
- layout: tinygrad native K/V `[Hkv, MAXC, Hd]`, no repack
- comparator: `gqa_coop_vec`
- result @ctx1024:
  - `2.35x` GPU-busy faster
  - `1.70x` matched-sync wall faster
  - near-exact correctness
  - `v_dot2=2`, `56 VGPR`, `8 KB LDS`, `0 spills`

B3 answered “can we own the primitive?” with yes. Its blocker was that raw HCQ `.co` launches were not graph nodes.

## B4 Summary

B4 removed the B3 graph-node blocker.

Implementation:

- `extra/qk_owned_flash_decode_graph_node.py`
- `tinygrad/llm/model.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_policy_sweep.py`

Mechanism:

- specialize `extra/qk_owned_flash_decode.hip` into single-kernel ELFs:
  - `owned_flash_tile_gqa`
  - `owned_flash_combine`
- bake `S`, `scale`, and `MAXC`
- pass `start_pos` as the single symbolic scalar var
- inject a fully formed precompiled `Ops.PROGRAM` via `Tensor.custom_kernel` + `Ops.BINARY`
- route through `DECODE_ATTN_AMDGCN_TILE=1`
- context gate through `DECODE_ATTN_AMDGCN_MIN_CTX`
- fallback to `gqa_coop_vec`

Proof:

- standalone eager, TinyJit capture, and TinyJit replay pass
- replay with a different `start_pos` than capture is correct
- in-model route firing is visible in captured graph names:
  - `owned_flash_tile_gqa`
  - `owned_flash_combine`
- greedy tokens match
- default behavior unchanged

Measurement traps fixed during B4:

- `.item()` must be inside the timed region; otherwise timing only captures async dispatch.
- `should_use_flash_decode` can fire at ctx512 through auto-threshold; route firing must be recorded.
- use in-process/interleaved W==D comparisons where practical.
- do not use rocprofv3 for tinygrad HCQ visibility.

## B4 W==D Outcome

Best policy results from `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`:

| policy | ctx512 | ctx1024 | ctx4096 | route firing |
|---|---:|---:|---:|---|
| `ctx2048_only` | +0.08% | +0.18% | +5.44% | only ctx4096 fired in measured set |
| `ctx4096_only` | +0.11% | +0.40% | +5.56% | only ctx4096 fired |
| `adaptive` | +0.24% | -0.76% | +5.36% | ctx1024 and ctx4096 fired |

Promotion gate:

```text
no ctx512 / ctx1024 regression
AND (>= +5% @ctx1024 OR >= +7% @ctx4096)
```

No tested policy cleared this gate.

Verdict: `B4_WD_FAIL_INTEGRATION`.

Interpretation: graph-node integration works. The miss is whole-decode economics: attention is a limited share of the
token step, and split-KV combine gives back part of the tile win.

## Combine-Tax Result

The follow-on combine-tax analysis classified the next bottleneck as `COMBINE_TAX_DOMINATES`.

Standalone per-kernel timing:

| ctx | opt S | tile us | combine us | total us | combine % |
|---:|---:|---:|---:|---:|---:|
| 512 | 48 | 16.0 | 12.7 | 28.7 | 44% |
| 1024 | 48 | 23.4 | 12.6 | 36.0 | 35% |
| 2048 | 48 | 36.8 | 12.6 | 49.4 | 26% |
| 4096 | 64 | 56.5 | 16.2 | 72.7 | 22% |

Key findings:

- combine is a flat latency floor by context and scales mainly with `S`
- combine is not HBM-bandwidth-bound: about 64 GB/s, far below peak
- combine under-occupies the GPU: roughly `Hq=32` workgroups with 32 threads
- reducing `S` does not solve it because the tile becomes starved
- halving combine at ctx4096 is projected to move W==D from ~+5.6% to ~+7.4%
- a free/fused combine is projected around ~+9.2% ctx4096

Verdict: the next attention-specific lever is a cheaper combine, not another tile.

## Split-KV Economics Audit (permanent layer, 2026-06-21)

`SPLIT_KV_ECONOMICS_AUDIT_READY`. The B4 combine-tax lesson is now a **durable, reusable audit** so a future
split-KV candidate cannot pass a local A/B without exposing the combine tax.

- tool: `extra/qk_split_kv_economics_audit.py` (default read-only over the measured B4 artifacts; `--live`
  regenerates the attribution; general `--attribution/--wd/--candidate` for any future candidate)
- artifact: `bench/qk-split-kv-economics-audit/latest.json` (`split_kv_economics_audit_v1`, contract-stamped CONFORMS 13/13)
- binding requirement: `split_kv_economics_contract_v1` in `bench/qk-decode-eval/binding_templates.json`
- B4 classifies `COMBINE_TAX_DOMINATES` — combine latency-bound (~64 GB/s, 32 wg << 96 CUs); Amdahl projection
  ctx4096 +5.41% measured → +6.97% half-combine → +8.58% free-combine.

Run:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_split_kv_economics_audit.py
```

Every future split-KV decode-attention candidate must report tile/combine split, combine fraction, effective
bandwidth, workgroup count, and the Amdahl projection — and be classified by this audit — **before** W==D
promotion work. See `docs/split-kv-economics-audit-result-20260621.md`.

## Recommended Next Action (updated 2026-06-22)

Route B attention is CLOSED for default-promotion: B5-lite cheaper combine done (hw 2.4x) but W==D SATURATES ~+5.7%
@ctx4096 (combine overlaps in-graph) -> `B5_COMBINE_LOCAL_PASS_WD_FAIL`. The decode time-tax audit
(`docs/decode-time-tax-audit-result-20260622.md`, `NEXT_PRIMITIVE_Q4K_GEMV_SCHEDULER`) shows the FFN Q4_K weight GEMV is
the dominant tax (gate/up 24% + down 14% = ~38%); q8 (+6%) proves it transfers, attention does not.

The FFN-GEMV scheduler diagnostic (`docs/decode-ffn-gemv-scheduler-diagnostic-result-20260622.md`,
`FFN_GEMV_DIAGNOSTIC_BOUNDED_SCHEDULE_SCOPE_READY`, class `GEMV_SCHEDULE_BOUND`) named the gap: tinygrad's gate/up GEMV
is ~51% peak (1 thread/row, serial whole-row K, uncoalesced) vs llama MMVQ ~70% via **128 threads/row + K-block-parallel
+ in-kernel warp-shuffle reduce**. The dot4/extract are already matched; the missing piece is WORK DECOMPOSITION. The
int-dot path is REFUTED in-model (Q4K_VDOT +1.25%, eaten by the q8-activation lifecycle); the lossless lever is an FP
work-decomposition GEMV that pays no lifecycle tax.

```text
DONE 2026-06-22: q4k_gemv_warp IMPLEMENTED + W==D PASS -> Q4K_GEMV_WARP_WD_PASS (docs/decode-ffn-gemv-warp-result-20260622.md).
LOSSLESS FP work-decomposition GEMV (32 threads/row + K-block-parallel + in-kernel warp_reduce_sum/ds_bpermute, one
output). gate/up+down W==D: +9.78%@1024 / +8.71%@4096 / +9.83%@512, greedy BYTE-IDENTICAL (decode 66.7->73.9 @1024,
~67%->~73% of llama). Local A/B 1.31x gate/up / 1.37x down vs the opted default. The FIRST decode primitive to clear
the W==D gate since the attention arc. default_eligible=true (lossless) but DEFAULT-OFF (Q4K_GEMV_WARP /
Q4K_GEMV_WARP_DOWN) pending owner approval.

HARDENED 2026-06-22 -> Q4K_GEMV_WARP_READY_FOR_OWNER_DEFAULT_DECISION (docs/q4k-gemv-warp-promotion-hardening-result-20260622.md):
promoted route reproduced (~+9.6%@1024 / +8.5%@4096, spread ~0.4%), real-generation BYTE-IDENTICAL (0/64),
default_eligible=true / default_on=false, fallback-safe. Same-lever expansions TESTED + banked research-only (do NOT
help W==D): Q6_K down (1.09x local, already coop-served; flag Q6K_GEMV_WARP_DOWN) + attn q/o (1.32x local but
attention-OVERLAPPED, no transfer; flag Q4K_GEMV_WARP_PROJ). Transfer test again discriminates: FFN weight GEMV
transfers, attention-adjacent does not.

NEXT candidates: (1) OWNER: flip Q4K_GEMV_WARP + Q4K_GEMV_WARP_DOWN default-on (lossless, +9.6%@1024, byte-identical,
no regression). (2) generalize the route guards for 14B/32B (kernel is shape-general; bounded follow-on).
```

Non-goals: no q8 default (lossy), no int-dot/MMVQ reopen (null in-model), no coalescing-only (gate/up not coop-routed),
no attention work (closed), no deep backend before the bounded FP variant is W==D-measured.

## Working Tree Note

At this handoff, B4-related work may still be uncommitted. Preserve these unless explicitly asked to revert:

- `tinygrad/llm/model.py`
- `bench/qk-decode-attention-route-b-b3/latest.json`
- `bench/qk-decode-eval/candidates.json`
- `bench/qk-decode-runtime-overhead/result.json`
- `docs/decode-attention-route-b-b4-external-graph-node-scope-20260621.md`
- `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
- `docs/b4-split-kv-combine-tax-scope-20260621.md`
- `docs/b4-split-kv-combine-tax-result-20260621.md`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_policy_sweep.py`
- `extra/qk_b4_combine_tax.py`
- `extra/qk_split_kv_economics_audit.py`
- `bench/qk-split-kv-economics-audit/latest.json`
- `bench/qk-decode-eval/binding_templates.json`
- `bench/qk-decode-eval/candidates.json`
- `docs/split-kv-economics-audit-scope-20260621.md`
- `docs/split-kv-economics-audit-result-20260621.md`

Run this to inspect:

```sh
cd /home/ubuntu/tinygrad-arkey
git status --short
```

## Useful Commands

Verify B4 graph-node path:

```sh
cd /home/ubuntu/tinygrad-arkey
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_owned_flash_decode_graph_node.py 48
```

Run B4 W==D harness:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_b4_decode_eval.py
```

Run combine-tax attribution:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_b4_combine_tax.py
```

Use `docs/README.md` for older arcs; do not reintroduce old handoff material here unless it is again current.
