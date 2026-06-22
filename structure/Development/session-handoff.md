# Session Handoff

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
- B4 AMDGCN decode-attention route: opt-in only
- no decode default promotion from B4

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

## Recommended Next Action

If continuing this lane:

```text
Route B B5-lite: cheaper split-KV combine for the existing B4 route (justified: audit verdict COMBINE_TAX_DOMINATES).
```

Goal:

- reduce `owned_flash_combine` from ~12-16 us to <= ~8 us at useful S values, ideally ~5 us
- preserve existing B4 graph-node route and correctness
- rerun `extra/qk_b4_decode_eval.py`
- pass W==D only if:
  - ctx4096 >= +7% with no ctx512/ctx1024 regression, or
  - ctx1024 >= +5%

Allowed directions:

- more parallel combine over `Hq x Hd` or similar
- cooperative reduction shape that increases occupancy
- bounded fused/streamed merge that removes partial write/read or second-kernel latency

Non-goals:

- no new attention tile
- no Route-A native codegen
- no KV repack
- no default promotion without W==D gate

If not continuing combine work, current state is:

```text
B4 infrastructure win banked.
B4 W==D promotion failed.
Bounded attention lane rests.
```

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
