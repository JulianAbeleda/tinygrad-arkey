# AMD ISA Register Accumulator LDS Reclaim Scope - 2026-06-29

## Purpose

Finish the loop-carried register-accumulator feature by reclaiming the LDS that is no longer used for pinned
`DEFINE_REG` accumulator state when:

```text
AMD_ISA_REG_ACCUM=1
```

RA1-RA4 proved the backend feature works. The remaining issue is metadata/resource accounting: the renderer now keeps
selected accumulator elements in pinned VGPRs, but the ELF descriptor still reserves LDS for those same `DEFINE_REG`
buffers.

## Current Ground Truth

| phase | artifact | result |
|---|---|---|
| RA1 | `bench/amd-isa-backend-regalloc-accum/ra1_latest.json` | loop-carried pinned accumulator microgates pass |
| RA2 | `bench/amd-isa-backend-regalloc-accum/ra2_latest.json` | native tile `DEFINE_REG` accumulators wire to pinned VGPRs; DS load/store drops |
| RA3 | `bench/amd-isa-backend-regalloc-accum/ra3_latest.json` | high pins inflated VGPR descriptor and regressed ctx4096 |
| RA4 | `bench/amd-isa-backend-regalloc-accum/ra4_latest.json` | low pins fix VGPR inflation; ctx512 improves, ctx4096 neutral |

RA4 W==D:

| ctx | baseline dynamic-S | RA4 low pins | result |
|---:|---:|---:|---|
| 512 | 67.09 | 70.72 | +5.4% |
| 4096 | 57.40 | 56.73 | -1.2%, effectively neutral/noise |

RA4 also shows:

```text
VGPR: 248 -> 56
lds_accum_stage DS load/store: 31 -> 9
token_match: true
route_bound: true
deterministic: true
flag-off path: byte-identical / safe
```

## Problem

The AMD ELF packer still sizes LDS as if every `AddrSpace.REG` `DEFINE_REG` buffer is backed by LDS:

```text
tinygrad/renderer/amd/elf.py

elif u.op in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG):
  nbytes = u.ptrdtype.size * u.ptrdtype.base.itemsize
  if u.ptrdtype.addrspace == AddrSpace.REG: reg_bytes += nbytes
  else: lds_size += nbytes
...
lds_size += reg_bytes * n_threads
```

That is correct for the default LDS accumulator path. It is too conservative for the opt-in pinned accumulator path,
where selected compile-time-indexed `DEFINE_REG` accumulator elements are no longer accessed through LDS.

The renderer side already chooses the pinned path in `tinygrad/renderer/isa/amd.py`:

```text
if AMD_ISA_REG_ACCUM=1
and dreg.addrspace == REG
and idx is CONST
and a pin exists:
  INDEX -> NOOP(..., arg="accum")
  LOAD  -> ACCUM_READ
  STORE -> ACCUM_WRITE
```

The descriptor sizing code does not know which `DEFINE_REG` elements were successfully pinned. It still reserves
per-thread LDS for the whole `DEFINE_REG` buffer, so group segment size remains higher than necessary. That likely
explains why ctx4096 is neutral after RA4: DS traffic was removed, but LDS occupancy/resource pressure was not fully
reclaimed.

## Goal

Make ELF `group_segment_fixed_size` agree with the renderer's actual accumulator storage decision.

When `AMD_ISA_REG_ACCUM=1`, only `DEFINE_REG` bytes that still fall back to LDS should contribute to group segment
size. Pinned accumulator elements should not reserve LDS.

## Non-Goals

- Do not change the pinned accumulator regalloc representation.
- Do not change pin placement; RA4 low pins are the current good state.
- Do not apply pinned accumulators to `DEFINE_LOCAL` K/V staging.
- Do not make `AMD_ISA_REG_ACCUM=1` the shipped default.
- Do not edit `autogen/**`.
- Do not remove the default LDS accumulator fallback.
- Do not claim speedup without W==D.

## Phase RL0 - Sizing Audit

Add an audit-only tool:

```text
extra/amd_isa_reg_accum_lds_reclaim_audit.py
```

Artifacts:

```text
bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl0_latest.json
bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl0_summary.md
```

The audit must report, for the native tile with `AMD_ISA_REG_ACCUM=0` and `AMD_ISA_REG_ACCUM=1`:

| field | meaning |
|---|---|
| `group_segment_fixed_size` | ELF descriptor LDS size |
| `define_local_bytes` | shared LDS tile/staging bytes |
| `define_reg_total_bytes_per_thread` | current total REG accumulator bytes per thread |
| `define_reg_pinned_bytes_per_thread` | bytes proven to route through pinned VGPRs |
| `define_reg_lds_fallback_bytes_per_thread` | bytes still requiring LDS |
| `n_threads` | local thread count multiplier |
| `expected_reclaim_bytes` | `pinned_bytes_per_thread * n_threads` |
| `actual_reclaim_bytes` | off LDS size - on LDS size |

Allowed RL0 verdicts:

```text
AMD_ISA_REG_ACCUM_LDS_RL0_PASS_RECLAIM_TARGET_PINNED
AMD_ISA_REG_ACCUM_LDS_RL0_BLOCKED_NO_PIN_METADATA
AMD_ISA_REG_ACCUM_LDS_RL0_BLOCKED_NO_RECLAIMABLE_DEFINE_REG
```

Stop if RL0 cannot prove which bytes are pinned.

## Phase RL1 - Descriptor Sizing Fix

Implement the smallest fix so `assemble_linear` computes LDS size from actual storage:

```text
tinygrad/renderer/amd/elf.py
```

Recommended design:

1. Add a helper in the AMD ISA renderer or ELF packer that can answer whether a `DEFINE_REG` element is pinned under
   `AMD_ISA_REG_ACCUM=1`.
2. For `AddrSpace.REG` buffers, count only the elements that still need LDS fallback.
3. Keep `DEFINE_LOCAL` sizing unchanged.
4. Keep `AMD_ISA_REG_ACCUM=0` byte-identical to the old sizing.

Important subtlety:

The current pin path only applies to compile-time constant element indices and only until the pin pool is exhausted. Do
not subtract the whole `DEFINE_REG` buffer unless every element that can be accessed is actually pinned. If the analysis
cannot prove that safely, subtract only the proven pinned elements.

Potential approaches:

- **Preferred:** annotate/recover pinned element set from the same logic used by renderer `_accum_pin`.
- **Acceptable:** conservatively scan the sink for `INDEX(DEFINE_REG, CONST)` accesses and subtract only those pinned
  constant elements when `AMD_ISA_REG_ACCUM=1` and the element count is within `ACCUM_PIN_BASE..ACCUM_PIN_TOP`.
- **Not acceptable:** subtract all `AddrSpace.REG` bytes unconditionally.

RL1 gate:

```text
extra/amd_isa_reg_accum_lds_reclaim_gate.py
```

Required checks:

- `AMD_ISA_REG_ACCUM=0` group segment size unchanged.
- `AMD_ISA_REG_ACCUM=1` group segment size decreases by the expected pinned accumulator bytes.
- `lds_accum_stage` DS load/store remains reduced.
- token match holds.
- route-bound/no fallback holds.
- Phase G/H cheap correctness still passes with `AMD_ISA_REG_ACCUM=1`.

Allowed RL1 verdicts:

```text
AMD_ISA_REG_ACCUM_LDS_RL1_PASS_DESCRIPTOR_RECLAIM
AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_FLAG_OFF_CHANGED
AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_UNSAFE_SIZE_ANALYSIS
AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_TOKEN_MATCH
AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_ROUTE_ATTRIBUTION
```

## Phase RL2 - W==D Measurement

Only after RL1 passes, measure W==D with:

```text
AMD_ISA_REG_ACCUM=1
```

Required measurements:

| context | required |
|---|---|
| ctx512 | native vs owned W==D |
| ctx4096 | native vs owned W==D |
| route | native tile route-bound; no hidden fallback |
| correctness | token match; deterministic repeated runs |
| resources | VGPR, group segment size, LDS occupancy estimate |
| attribution | PC/source trace and N4 whole-step attribution before/after |

Expected outcome:

- ctx512 may stay near RA4's +5.4%.
- ctx4096 should be the primary target. If LDS group segment size was the reason ctx4096 stayed neutral, ctx4096 should
  improve after reclaim.

Allowed RL2 verdicts:

```text
AMD_ISA_REG_ACCUM_LDS_RL2_PASS_CTX4096_MOVEMENT
AMD_ISA_REG_ACCUM_LDS_RL2_PASS_RESOURCE_RECLAIM_NO_WD_MOVEMENT
AMD_ISA_REG_ACCUM_LDS_RL2_BLOCKED_TOKEN_MATCH
AMD_ISA_REG_ACCUM_LDS_RL2_BLOCKED_NONDETERMINISM
AMD_ISA_REG_ACCUM_LDS_RL2_BLOCKED_ROUTE_ATTRIBUTION
```

## Required Regression Ladder

Minimum:

```text
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_reg_accum_lds_reclaim_audit.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_reg_accum_lds_reclaim_gate.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_pc_source_trace.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_phase_n4_whole_step_attribution.py
```

Also run the existing RA microgate to ensure the feature itself still works:

```text
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_regalloc_accum_microgate.py
```

Run the cheap route/correctness gates used by RA2/RA4 if present.

## Success Criteria

The scope succeeds if:

1. flag-off descriptor sizing is unchanged;
2. flag-on descriptor sizing reclaims only proven pinned accumulator LDS;
3. token match, route-bound, no fallback, and determinism hold;
4. `lds_accum_stage` DS load/store remains reduced;
5. ctx512/ctx4096 W==D is measured;
6. the final artifact honestly reports whether ctx4096 moved.

## Stop Conditions

Stop and record a blocker if:

- the tool cannot prove which `DEFINE_REG` bytes are pinned;
- subtracting LDS breaks token match;
- subtracting LDS changes flag-off behavior;
- the analysis accidentally removes LDS required for `DEFINE_LOCAL` K/V staging;
- the pin pool falls back to LDS but descriptor sizing assumes it did not;
- W==D cannot be measured because route attribution fails.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/amd-isa-reg-accum-lds-reclaim-scope-20260629.md

Context:
RA1-RA4 completed the loop-carried register accumulator feature. The feature is correct and net-positive:

  ctx512: 67.09 -> 70.72 tok/s (+5.4%)
  ctx4096: 57.40 -> 56.73 tok/s (~neutral/noise)

RA4 fixed the VGPR inflation by moving pins low: VGPR 248 -> 56. The remaining issue is that elf.py still reserves LDS for DEFINE_REG accumulator buffers even when AMD_ISA_REG_ACCUM=1 routes those accumulator elements to pinned VGPRs. DS load/store dropped 31 -> 9, but group_segment_fixed_size still includes now-unused accumulator LDS. This likely explains why ctx4096 did not improve.

Task:
Implement the LDS reclaim scope in phases.

Phase RL0:
Add audit-only:

  extra/amd_isa_reg_accum_lds_reclaim_audit.py

Artifacts:

  bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl0_latest.json
  bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl0_summary.md

RL0 must report group_segment_fixed_size, DEFINE_LOCAL bytes, DEFINE_REG total bytes, DEFINE_REG pinned bytes, DEFINE_REG LDS fallback bytes, n_threads, expected_reclaim_bytes, and actual_reclaim_bytes for AMD_ISA_REG_ACCUM=0 and =1.

Stop if RL0 cannot prove which DEFINE_REG bytes are pinned.

Phase RL1:
Only if RL0 passes, fix descriptor sizing in:

  tinygrad/renderer/amd/elf.py

Goal:
When AMD_ISA_REG_ACCUM=1, pinned DEFINE_REG accumulator elements must not reserve LDS. DEFINE_LOCAL K/V staging must remain unchanged. AMD_ISA_REG_ACCUM=0 must be byte-identical / descriptor-identical to the old path.

Do not subtract the whole AddrSpace.REG buffer unless every element is proven pinned. If uncertain, subtract only proven pinned constant-index elements.

Add gate:

  extra/amd_isa_reg_accum_lds_reclaim_gate.py

Required proof:

  - flag-off group segment unchanged
  - flag-on group segment decreases by expected pinned accumulator bytes
  - lds_accum_stage DS load/store remains reduced
  - token_match true
  - route_bound true
  - no hidden fallback
  - Phase G/H cheap correctness with AMD_ISA_REG_ACCUM=1 still passes

Phase RL2:
Only if RL1 passes, measure W==D at ctx512 and ctx4096 with AMD_ISA_REG_ACCUM=1. Re-run PC/source trace and N4 whole-step attribution before/after. The primary target is ctx4096.

Constraints:

  - Do not edit autogen/**
  - Do not change the pinned register regalloc feature
  - Do not move pins again
  - Do not apply reclaim to DEFINE_LOCAL
  - Do not make AMD_ISA_REG_ACCUM=1 the shipped default
  - Do not claim speedup without W==D
  - Stop at the first hard blocker

Final report must include:

  - RL0 sizing audit verdict
  - RL1 descriptor reclaim verdict
  - before/after group_segment_fixed_size
  - before/after VGPR
  - before/after lds_accum_stage DS load/store
  - W==D ctx512/ctx4096
  - token_match / route_bound / no fallback / determinism
  - exact files changed
  - final verdict
```

