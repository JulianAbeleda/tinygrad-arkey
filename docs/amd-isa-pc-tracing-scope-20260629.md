# AMD ISA PC-Level Speed Tracing Scope — 2026-06-29

## Purpose

Build the missing tracing layer between the existing speed attribution tools and implementation work:

1. `extra/amd_isa_phase_n4_whole_step_attribution.py` says which decode-step kernel owns the gap.
2. `extra/amd_isa_phase_n2b_pmc_attribution.py` says which counter category owns the tile gap.
3. The missing tool should say which exact instruction / PC / source lowering owns the hot category.

This is audit-only tooling. Do not optimize the kernel in this phase.

## Current Ground Truth

The repo already has a working two-level attribution stack:

| level | tool | artifact | current result |
|---|---|---|---|
| whole decode step | `extra/amd_isa_phase_n4_whole_step_attribution.py` | `bench/amd-isa-backend-phase-n4/latest.json` | native attention tile is the native-vs-owned GPU-compute delta; GEMV and generated reductions mostly cancel |
| tile hardware category | `extra/amd_isa_phase_n2b_pmc_attribution.py` | `bench/amd-isa-backend-phase-n2b/latest.json` | dynamic instruction volume / LDS+VALU work dominates; VMEM and LDS-wait are not primary |
| static ISA/resource mix | `extra/amd_isa_phase_n0_throughput_diff.py` | `bench/amd-isa-backend-phase-n0/latest.json` | useful static diff; N1A already fixed the major static VALU excess via hardware `v_exp_f32` |
| attempted per-PC tracing | `extra/amd_isa_phase_n2_dynamic_stall_attribution.py` | `bench/amd-isa-backend-phase-n2/latest.json` | degraded: SQTT/ATT path produces occupancy/wave timing but no reliable per-PC stall rows |

Current W==D package from N7:

| ctx | native | owned | native % |
|---:|---:|---:|---:|
| 512 | 67.09 | 103.15 | 65.0% |
| 1024 | 66.72 | 101.22 | 65.9% |
| 2048 | 64.46 | 98.92 | 65.2% |
| 4096 | 57.40 | 94.10 | 61.0% |

The next tracing tool should explain where the remaining native attention tile cost is spent, not rediscover that the
tile is the gap.

## Problem Statement

The current profiling stack can select the right area but cannot yet point at an exact codegen site:

- N4 attributes the decode-step delta to `native_block_tile`.
- N2B attributes the tile delta to dynamic work volume and LDS/VALU instruction counts.
- N5A found the likely structural fix, register accumulators, but it is blocked on regalloc semantics.
- The SQTT/ATT per-PC path is an infrastructure wall under the current HCQ setup.

We need a durable PC/source tracer that can answer:

```text
Which native_block_tile instruction families execute most dynamically?
Which PCs map to which generated Insts?
Which source UOp / lowering path emitted those Insts?
Which candidate implementation should we try next?
```

## Non-Goals

- Do not optimize decode attention in this phase.
- Do not re-open N1B scalar address math unless the new tracer proves it is live and hot.
- Do not revive N5A register accumulators as a code change in this phase.
- Do not require the broken ATT per-PC path for a pass.
- Do not make native attention the shipped default.
- Do not edit `autogen/**`.

## Required Tool

Add a new audit tool:

```text
extra/amd_isa_pc_source_trace.py
```

Write artifacts to:

```text
bench/amd-isa-backend-pc-source-trace/latest.json
bench/amd-isa-backend-pc-source-trace/summary.md
bench/amd-isa-backend-pc-source-trace/native_inst_stream.json
bench/amd-isa-backend-pc-source-trace/owned_disasm.json
```

The tool should compare `native_block_tile` against `owned_flash_tile_gqa_whole` where possible, but the primary goal is
native source attribution.

## Minimum Viable Implementation

### 1. Capture native Inst stream with PC addresses

Use the existing AMDISARenderer / `assemble_linear` path to capture the exact native `Inst` stream for `native_block_tile`.

The output must include, per instruction:

| field | meaning |
|---|---|
| `pc` | byte PC or byte offset in the kernel |
| `asm` | final rendered instruction text |
| `opcode` | normalized opcode, for example `ds_load_b32`, `v_fma_f32`, `v_dot2_f32_f16` |
| `category` | `VALU`, `SALU`, `LDS`, `VMEM`, `SMEM`, `BRANCH`, `WAIT`, `BARRIER`, `OTHER` |
| `src_uop` | best-effort UOp / tag / lowering origin |
| `lowering_site` | best-effort file/function marker, usually `tinygrad/renderer/isa/amd.py` lowering path |
| `loop_context` | best-effort loop / range id if available |
| `exec_predicated` | whether the instruction is inside an EXEC-masked region |

If exact source UOp mapping is not available yet, build the hook first:

- add optional debug metadata on AMDOps / Inst carriers in `tinygrad/renderer/isa/amd.py`;
- preserve it through post-regalloc / scheduler / waitcnt insertion;
- fail gracefully with `src_uop: null` only for instructions that are truly synthetic, such as inserted `s_waitcnt`.

### 2. Dynamic weighting without ATT

Because ATT/SQTT per-PC decode is not reliable here, create a conservative dynamic estimator from compiler-visible loop
structure:

```text
dynamic_weight = static_count * estimated_loop_trip_count * launch_workgroups * waves_per_workgroup
```

This does not need to be perfect. It must be good enough to rank families such as:

- LDS-backed accumulator load/store traffic
- ds_bpermute reductions
- v_dot2/fdot score path
- online-softmax exp/max/denominator path
- PV accumulation
- gated stores / EXEC regions
- index/address math

Use the active dynamic-S route, not fixed-S. For context-sensitive estimates, emit rows for at least:

```text
ctx512
ctx4096
```

The tool must clearly mark the weighting as `estimated_dynamic_weight`, not measured per-PC hardware stalls.

### 3. Merge measured category counters

Consume the latest N2B artifact if present:

```text
bench/amd-isa-backend-phase-n2b/latest.json
```

Use N2B to scale or sanity-check category totals:

- `SQ_INSTS_VALU`
- `SQ_INSTS_LDS`
- `SQ_WAVE_CYCLES`
- `SQ_WAIT_ANY`
- `SQ_WAIT_INST_LDS`
- `SQC_LDS_BANK_CONFLICT`
- `GL2C_HIT/MISS`

The new tracer should not claim hardware-measured per-PC stalls. It should say:

```text
category measured by PMC, PC/source rows estimated by static loop weighting
```

### 4. Produce ranked source rows

Emit `source_hot_rows` sorted by estimated dynamic weight:

| field | meaning |
|---|---|
| `rank` | sorted rank |
| `source_group` | accumulator, fdot, exp, reduce, PV, address, predicate, waitcnt, etc. |
| `category` | VALU/LDS/etc. |
| `static_insts` | static instruction count |
| `estimated_dynamic_insts_ctx512` | estimate |
| `estimated_dynamic_insts_ctx4096` | estimate |
| `pmc_category_ratio_native_over_owned` | if available |
| `example_pcs` | 3-8 representative PCs |
| `example_asm` | representative instruction text |
| `lowering_sites` | file/function names |
| `candidate_lever` | exact next possible optimization |
| `confidence` | high/medium/low |

The expected useful rows are likely:

- LDS-backed accumulator state
- half-LDS K/V staging
- online-softmax recurrence
- PV accumulation
- cross-lane reduce / `ds_bpermute`
- address/index math
- EXEC-gated stores

But do not hardcode the verdict. Let the rows rank themselves.

### 5. Optional SQTT retry, but not required

The tool may attempt SQTT/ATT per-PC decode behind an opt-in flag:

```text
AMD_ISA_PC_TRACE_TRY_SQTT=1
```

If it still hits the known wall, record:

```text
per_pc_hardware_trace: unavailable
reason: ATT_DECODER_REPAIR_BLOCKED / no profiled HSA AQL path / instructions_size==0
```

This must not make the main audit fail.

## Acceptance Criteria

The phase passes if:

1. `native_block_tile` route-binds, token-match remains true, and no hidden fallback is seen.
2. `native_inst_stream.json` contains PCs, opcodes, categories, and at least best-effort lowering metadata.
3. `latest.json` contains ranked `source_hot_rows`.
4. `summary.md` names the top 3 source groups and their candidate levers.
5. The tool reuses N4/N2B/N0 artifacts instead of replacing them.
6. It explicitly distinguishes measured PMC category data from estimated PC/source weighting.
7. It exits with a clear verdict:

```text
AMD_ISA_PC_SOURCE_TRACE_PASS_SOURCE_ROWS_PINNED
AMD_ISA_PC_SOURCE_TRACE_BLOCKED_ROUTE_ATTRIBUTION
AMD_ISA_PC_SOURCE_TRACE_BLOCKED_TOKEN_MATCH
AMD_ISA_PC_SOURCE_TRACE_BLOCKED_METADATA_LOSS
AMD_ISA_PC_SOURCE_TRACE_INCONCLUSIVE_DYNAMIC_WEIGHTING
```

## Expected Output Shape

`latest.json` should include:

```json
{
  "verdict": "AMD_ISA_PC_SOURCE_TRACE_PASS_SOURCE_ROWS_PINNED",
  "route": "native_dynamic_s",
  "token_match": true,
  "route_bound": true,
  "hidden_fallback": false,
  "hardware_per_pc_trace": {
    "available": false,
    "reason": "ATT/SQTT per-PC decode unavailable; PMC category counters used instead"
  },
  "pmc_category_reference": "bench/amd-isa-backend-phase-n2b/latest.json",
  "whole_step_reference": "bench/amd-isa-backend-phase-n4/latest.json",
  "source_hot_rows": []
}
```

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Task: implement the audit-only AMD ISA PC/source tracing scope in:

  docs/amd-isa-pc-tracing-scope-20260629.md

Goal:
Build the missing tracing bridge between the existing attribution tools and codegen optimization work. N4 already tells us the native attention tile owns the decode-step delta. N2B already tells us PMC category counters show dynamic LDS/VALU work, not VMEM/cross-lane wait, dominates. The missing tool must map the native tile's final instructions back to source/lowering groups so we can choose concrete places to speed up.

Required new tool:

  extra/amd_isa_pc_source_trace.py

Required artifacts:

  bench/amd-isa-backend-pc-source-trace/latest.json
  bench/amd-isa-backend-pc-source-trace/summary.md
  bench/amd-isa-backend-pc-source-trace/native_inst_stream.json
  bench/amd-isa-backend-pc-source-trace/owned_disasm.json

Use the current authoritative tools as inputs:

  extra/amd_isa_phase_n4_whole_step_attribution.py
  extra/amd_isa_phase_n2b_pmc_attribution.py
  extra/amd_isa_phase_n0_throughput_diff.py

Do not replace those tools. Reuse their artifacts when present.

Important constraints:

1. Audit only. Do not optimize decode attention in this phase.
2. Do not re-open N1B scalar address math unless the new trace proves it is hot and live.
3. Do not revive N5A register accumulators as a code change in this phase.
4. Do not require ATT/SQTT per-PC tracing for a pass. That path is known degraded under HCQ.
5. Do not edit autogen/**.
6. Preserve default AMD/HIP behavior.

Implementation target:

Capture the native_block_tile final Inst stream from AMDISARenderer / assemble_linear with best-effort PC/source metadata:

  pc
  asm
  opcode
  category
  src_uop
  lowering_site
  loop_context
  exec_predicated

If exact UOp metadata is missing, add a minimal optional metadata hook in tinygrad/renderer/isa/amd.py and preserve it through post-regalloc, scheduler, waitcnt insertion, and assembly. Synthetic instructions such as inserted s_waitcnt may have src_uop=null but should still have category/lowering_site.

Because ATT per-PC tracing is unavailable, add estimated dynamic weighting from visible loop/grid structure:

  estimated_dynamic_weight = static_count * estimated_loop_trip_count * launch_workgroups * waves_per_workgroup

Emit ctx512 and ctx4096 estimates. Clearly label this as estimated, not hardware-measured per-PC stalls.

Merge N2B PMC category data as measured category-level truth. The final report must say:

  "category measured by PMC, PC/source rows estimated by static loop weighting"

Output ranked source_hot_rows with:

  rank
  source_group
  category
  static_insts
  estimated_dynamic_insts_ctx512
  estimated_dynamic_insts_ctx4096
  pmc_category_ratio_native_over_owned
  example_pcs
  example_asm
  lowering_sites
  candidate_lever
  confidence

Acceptance:

Pass only if native route binds, token match remains true, no hidden fallback occurs, native_inst_stream.json has PCs/opcodes/categories/lowering metadata, latest.json has ranked source_hot_rows, and summary.md names the top 3 source groups plus concrete candidate levers.

Allowed verdicts:

  AMD_ISA_PC_SOURCE_TRACE_PASS_SOURCE_ROWS_PINNED
  AMD_ISA_PC_SOURCE_TRACE_BLOCKED_ROUTE_ATTRIBUTION
  AMD_ISA_PC_SOURCE_TRACE_BLOCKED_TOKEN_MATCH
  AMD_ISA_PC_SOURCE_TRACE_BLOCKED_METADATA_LOSS
  AMD_ISA_PC_SOURCE_TRACE_INCONCLUSIVE_DYNAMIC_WEIGHTING

Run the existing regression ladder that is relevant and cheap:

  DEV=AMD PYTHONPATH=. python3 extra/amd_isa_phase_n4_whole_step_attribution.py
  DEV=AMD PYTHONPATH=. python3 extra/amd_isa_phase_n2b_pmc_attribution.py
  DEV=AMD PYTHONPATH=. python3 extra/amd_isa_pc_source_trace.py

Stop at the first hard blocker. Do not claim a speedup. The deliverable is tooling that tells us where to look next.
```

