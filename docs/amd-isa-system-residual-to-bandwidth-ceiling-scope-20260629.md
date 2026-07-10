# AMD ISA System Residual To Bandwidth Ceiling Scope - 2026-06-29

## Purpose

Audit the remaining system-level gap after the Q4_K G3 route matched the hand-owned warp route.

The new question is no longer:

```text
Can generated Q4_K GEMV match owned?
```

It is:

```text
Why does the best full decode route sit around 95-104 tok/s when the measured streaming bandwidth ceiling implies
about 163 tok/s?
```

This is a system residual audit. Do not implement new kernels in this phase.

## Current Ground Truth

### Attention track

Closed:

```text
AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION
```

Attention KV-read is less than 1% of the weight-read floor at ctx4096. Further attention work is low leverage.

### Q4_K GEMV track

Closed for parity:

```text
AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT
```

Generated G3 LaneMap matches owned warp for the major Q4_K decode roles:

| ctx | owned | G3 | delta |
|---:|---:|---:|---:|
| 512 | 103.79 | 103.93 | -0.13% |
| 1024 | 101.98 | 102.04 | -0.06% |
| 2048 | 99.56 | 99.74 | -0.18% |
| 4096 | 94.83 | 94.44 | +0.41% |

Route attribution:

```text
Q4_K gate/up -> G3
Q4_K q/o     -> G3
Q4_K down    -> G3
owned warp leakage: 0
bridge leakage: 0
fallback leakage: 0
token_match: true
```

### Weight floor

Measured:

```text
peak HBM: 960 GB/s
measured streaming-copy achievable: 820 GB/s
Qwen3-8B-Q4_K model bytes: 5.03 GB
realistic bandwidth ceiling: about 163 tok/s
current best decode: about 95-104 tok/s
```

The implied best-route bandwidth is about:

```text
5.03 GB * 103.8 tok/s = about 522 GB/s
```

That is about 64% of measured streaming-copy achievable bandwidth.

## Central Hypothesis

The remaining gap is not a single Q4_K GEMV kernel issue. It is likely some mixture of:

| possible cause | why plausible |
|---|---|
| non-Q4_K roles | Q6_K/lm_head/unknown bucket still appear in the weight-path role breakdown |
| launch/graph fragmentation | many small kernels around GEMVs can reduce effective bandwidth vs streaming-copy |
| dequant arithmetic tax | Q4_K decode is not a pure byte stream; unpack/dequant/dot arithmetic may cap effective BW below memcpy |
| memory access pattern tax | even coalesced GEMV may not reach streaming-copy due to per-row/block layout and small working set rhythm |
| activation/state overhead | elementwise, silu/gate multiply, norm, residual/copy kernels consume wall and bandwidth |
| CPU/wall timing confound | short decode steps have auto-clock / ramp / wall spread artifacts; need GPU-time and interleaved W==D views |
| quant-policy overhead | Q6_K roles may be quality-preserving but bandwidth-expensive relative to Q4_K |
| unavoidable model work | the 5.03 GB streaming floor may undercount metadata, repeated reads, KV/cache, activations, and output work |

The audit must separate these into measured buckets.

## New Tool

Add:

```text
extra/amd_isa_system_residual_ceiling_audit.py
```

Artifacts:

```text
bench/amd-isa-backend-system-residual-ceiling/latest.json
bench/amd-isa-backend-system-residual-ceiling/summary.md
bench/amd-isa-backend-system-residual-ceiling/loss_stack.json
bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
bench/amd-isa-backend-system-residual-ceiling/probe_matrix.json
```

## Inputs

Use existing artifacts:

```text
bench/amd-isa-backend-g3-weight-promotion/latest.json
bench/amd-isa-backend-weight-path-ceiling/latest.json
bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
bench/amd-isa-backend-decode-attention-ceiling/latest.json
bench/amd-isa-backend-phase-n4/latest.json
bench/qk-proj-gemv-warp/decision.json
bench/qk-scheduler-gemv-vs-owned/decision.json
```

If an input is stale because G3 promotion superseded it, mark it stale and either re-run or derive a corrected row.

## Phase S0 - Full Decode Loss Stack

Build a loss stack from measured floor to current route:

```text
820 GB/s streaming ceiling
-> adjusted decode byte floor
-> Q4_K G3/owned GEMV effective bandwidth
-> non-Q4_K GEMV tax
-> non-GEMV GPU tax
-> launch/graph/wall tax
-> observed W==D
```

Required outputs:

| field | meaning |
|---|---|
| `streaming_copy_ceiling_tok_s` | 163-ish tok/s |
| `adjusted_decode_floor_tok_s` | after adding metadata/KV/activation/output bytes if measurable |
| `best_route_tok_s` | G3/owned W==D |
| `best_route_implied_bw` | model bytes * tok/s |
| `gap_to_streaming_ceiling_pct` | residual |
| `loss_buckets` | measured/estimated components |

Verdicts:

```text
AMD_ISA_SYSTEM_S0_PASS_LOSS_STACK_PINNED
AMD_ISA_SYSTEM_S0_INCONCLUSIVE_STALE_INPUTS
```

## Phase S1 - Kernel Taxonomy Refresh

Refresh full-decode per-kernel attribution with the promoted G3 route.

Add or reuse a route-aware N4-style profiler, but classify all kernels into:

| bucket | examples |
|---|---|
| `q4k_g3_gemv` | G3 gate/up, down, q/o |
| `q6k_gemv` | Q6_K coop/warp/fallback |
| `lm_head` | vocab projection |
| `norm_rope_elementwise` | RMSNorm, RoPE, silu, gate multiply |
| `attention_tile` | native/owned attention |
| `attention_reduce_combine` | gmax/combine |
| `copy_cast_sync` | copies/casts/materialization |
| `unknown` | must be split or justified |

Required metrics per bucket:

```text
gpu_time_per_token
calls_per_token
bytes_estimate
effective_bw_estimate
ctx512/1024/2048/4096
route_class
```

Verdicts:

```text
AMD_ISA_SYSTEM_S1_PASS_KERNEL_TAXONOMY_REFRESHED
AMD_ISA_SYSTEM_S1_BLOCKED_ROUTE_ATTRIBUTION
AMD_ISA_SYSTEM_S1_INCONCLUSIVE_UNKNOWN_BUCKET_TOO_LARGE
```

Unknown bucket threshold:

```text
unknown > 10% of GPU time => inconclusive unless explained.
```

## Phase S2 - Differential Probe Matrix

Create a probe matrix that tests where the 522 -> 820 GB/s gap lives.

Required probes:

| probe id | target | purpose |
|---|---|---|
| `SR0_BEST_ROUTE_REMEASURE` | W==D | stable best-route baseline with G3 promotion |
| `SR1_Q4K_ONLY_MICROSTEP` | Q4_K GEMV group | measure Q4_K G3 kernels in isolation against streaming-copy floor |
| `SR2_Q6K_OFF_OR_DEMOTE_ESTIMATE` | Q6_K/lm_head | estimate/measure whether Q6_K roles are a meaningful residual |
| `SR3_NON_GEMV_DISABLE_MASK` | non-GEMV GPU tax | semantic-masking or microstep to measure norm/elementwise/copy overhead |
| `SR4_LAUNCH_FUSION_BOUND` | launch/graph fragmentation | compare sum of GPU kernel times vs W==D wall; estimate launch/graph overhead |
| `SR5_DEQUANT_ARITH_TAX` | dequant lifecycle | microbenchmark Q4_K G3 load+dequant+dot vs raw packed-weight stream |
| `SR6_METADATA_BYTE_TAX` | scales/mins/metadata | compute actual bytes touched vs naive 5.03 GB floor |
| `SR7_CONTEXT_SLOPE` | ctx dependence | show which buckets change with ctx and which are weight-fixed |
| `SR8_CLOCK_NOISE_CONTROL` | measurement reliability | interleaved repeats / spread, GPU-time vs wall |

Each probe must report:

```text
probe_type: measurement-only | semantic-masking | microkernel | estimate
baseline
probe
delta
confidence
decision
```

Verdicts:

```text
AMD_ISA_SYSTEM_S2_PASS_PROBES_PINNED
AMD_ISA_SYSTEM_S2_INCONCLUSIVE_NOISY
AMD_ISA_SYSTEM_S2_BLOCKED_MISSING_PROBE
```

## Phase S3 - Next Target Selection

The audit must choose exactly one next target:

| condition | next target |
|---|---|
| Q6_K/lm_head is >=10% wall and has >=5% W==D ceiling | `q6k_lmhead_or_quant_policy_track` |
| non-GEMV elementwise/copy is >=10% wall | `decode_graph_fusion_or_elementwise_track` |
| launch/graph overhead is >=10% wall | `decode_graph_capture_or_kernel_fusion_track` |
| Q4_K G3 isolated BW is far below achievable while full route parity holds | `q4k_g3_microarchitecture_track` |
| metadata/dequant tax explains most gap and is unavoidable | `document_practical_ceiling_and_stop_weight_kernel_tuning` |
| all buckets below thresholds | `broaden_system_target_or_accept_practical_ceiling` |

Allowed final verdicts:

```text
AMD_ISA_SYSTEM_RESIDUAL_PASS_NEXT_TARGET_SELECTED
AMD_ISA_SYSTEM_RESIDUAL_PASS_PRACTICAL_CEILING_DOCUMENTED
AMD_ISA_SYSTEM_RESIDUAL_INCONCLUSIVE_NEEDS_BETTER_PROFILING
```

## Output Requirements

`latest.json` must include:

```json
{
  "verdict": "...",
  "current_best_route": {},
  "loss_stack": {},
  "kernel_taxonomy": {},
  "probe_matrix": {},
  "next_target": {
    "id": "...",
    "reason": "...",
    "expected_ceiling": "..."
  },
  "refuted_targets": [],
  "input_artifacts": {}
}
```

`summary.md` must include:

- current best W==D table;
- streaming/floor/implied-bandwidth table;
- kernel taxonomy table;
- loss stack table;
- probe matrix table;
- selected next target with reasoning;
- caveats on measurement noise.

## Non-Goals

- Do not change defaults.
- Do not remove owned kernels.
- Do not reopen Q4_K layout reshuffle unless a probe contradicts G3 parity.
- Do not reopen attention unless the taxonomy contradicts the attention ceiling.
- Do not implement new kernels.
- Do not edit `autogen/**`.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/amd-isa-system-residual-to-bandwidth-ceiling-scope-20260629.md

Context:
Attention is closed as low leverage. Q4_K G3 is promoted/hardened as speed-equivalent to owned:

  AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT

The remaining question is system-level:

  best decode route ~= 95-104 tok/s
  measured streaming-copy achievable ceiling ~= 163 tok/s
  implied best-route bw ~= 522 GB/s vs 820 GB/s achievable

Task:
Build an audit-only system residual ceiling tool:

  extra/amd_isa_system_residual_ceiling_audit.py

Artifacts:

  bench/amd-isa-backend-system-residual-ceiling/latest.json
  bench/amd-isa-backend-system-residual-ceiling/summary.md
  bench/amd-isa-backend-system-residual-ceiling/loss_stack.json
  bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json
  bench/amd-isa-backend-system-residual-ceiling/probe_matrix.json

Phases:

  S0: full decode loss stack from 820 GB/s streaming ceiling to observed W==D
  S1: refreshed route-aware kernel taxonomy with promoted G3
  S2: differential probe matrix for the 522 -> 820 GB/s residual
  S3: choose exactly one next target or document practical ceiling

Must distinguish:

  q4k_g3_gemv
  q6k_gemv
  lm_head
  norm_rope_elementwise
  attention_tile
  attention_reduce_combine
  copy_cast_sync
  unknown

Required probes:

  SR0_BEST_ROUTE_REMEASURE
  SR1_Q4K_ONLY_MICROSTEP
  SR2_Q6K_OFF_OR_DEMOTE_ESTIMATE
  SR3_NON_GEMV_DISABLE_MASK
  SR4_LAUNCH_FUSION_BOUND
  SR5_DEQUANT_ARITH_TAX
  SR6_METADATA_BYTE_TAX
  SR7_CONTEXT_SLOPE
  SR8_CLOCK_NOISE_CONTROL

Final verdicts:

  AMD_ISA_SYSTEM_RESIDUAL_PASS_NEXT_TARGET_SELECTED
  AMD_ISA_SYSTEM_RESIDUAL_PASS_PRACTICAL_CEILING_DOCUMENTED
  AMD_ISA_SYSTEM_RESIDUAL_INCONCLUSIVE_NEEDS_BETTER_PROFILING

Do not optimize kernels. Do not change defaults. Do not reopen Q4_K layout reshuffle unless the audit contradicts G3 parity. Do not edit autogen/**.
```

