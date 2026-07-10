# 14B MMQ R4 ASM Cracking Scope

Purpose: test the R4 theories by coding the smallest bounded AMD low-level
probes. These are research-only probes. They must not change production
dispatch, default route policy, or promotion gates.

## Definition of ASM for This Pass

In this repo, the existing MMQ atoms use tinygrad custom UOps plus AMD
intrinsics such as `__builtin_amdgcn_sudot4`. This pass should use that same
lowest available surface unless a true raw-ASM path is already present.

Allowed:

```text
extra/qk/mmq_q4k_q8_atom.py research helpers
UOp.special for gidx/lidx
UOp.placeholder REG/LOCAL
Ops.CUSTOMI AMD intrinsics
custom_kernel test runners
source/hash evidence
```

Not allowed:

```text
production dispatch changes
route binding
default_route changes
claims of promotion eligibility
silent direct_packed fallback while claiming MMQ
```

## Objective

Crack R4 by proving or refuting the minimum representation problem:

```text
Can tinygrad's AMD custom-kernel surface represent llama's cooperative 8-wave
writeback ownership with no duplicate/missing output stores?
```

If yes, proceed to accumulator and staging probes. If no, record the exact
missing primitive and stop before building numeric kernels on a false base.

## Work Packages

### A. Store-Only Owner Trace

Theory tested:

```text
Theory A - final-store ownership is the primary blocker.
```

Implement:

```text
research-only store marker kernel
input/output shape: bounded MxN marker matrix
intended owner law: wave_id owns 16 output rows, j fragment owns 16 columns
write marker=1 exactly once per output
return actual_store_coverage
compare to llama_mma_writeback_coverage
```

Start shapes:

```text
16x16
32x16
32x32
128x128 if compile/resource safe
```

Pass:

```text
duplicate_store_count == 0
missing_store_count == 0
actual_owner_hash == oracle_owner_hash
production_dispatch_changed == false
```

Fail/block:

```text
cannot map lidx/gidx to wave_id/lane_id
cannot emit multi-wave store-only kernel without illegal duplicate stores
custom UOp surface cannot address output by owner fragment
```

### B. Sum-Slot / Accumulator Probe

Theory tested:

```text
Theory B - per-thread sum[] accumulator placement is the primary blocker.
```

Implement:

```text
research-only accumulator-slot map
no Q4/Q8 math initially
write each output as a deterministic function of intended sum slot
report sum_slots_per_thread
report VGPR/scratch if resource extraction is available
```

Pass:

```text
sum slot -> output map equals oracle fragment map
scratch_bytes == 0 if measured
VGPR is below an explicit threshold or recorded as unknown
```

Fail/block:

```text
REG placeholders cannot model multiple per-thread sum slots
lowering spills or loses slot ownership
```

### C. LDS Lifecycle Probe

Theory tested:

```text
Theory C - LDS staging/reuse lifecycle is the primary performance lever.
```

Implement after A or B:

```text
stage Q8_1 tile_y panel bytes into LOCAL
barrier
read staged bytes to output marker/checksum
repeat two-panel lifecycle
record barrier/local load/store counters
```

Pass:

```text
panel bytes/checksum match reference
barrier lifecycle matches expected order
no production route changes
```

Fail/block:

```text
cannot express panelized LOCAL layout
barrier ordering cannot be tied to subsequent loads
```

### D. Q4_K Tile-X Probe

Theory tested:

```text
Theory F - Q4_K loader/scale decode is the hidden bottleneck.
```

Implement after A:

```text
stage or compute Q4_K tile_x fields for a bounded row/k slice
compare x_qs/x_dm/scales/mins against a Python oracle derived from vendored mmq.cuh behavior
```

Pass:

```text
tile_x bytes/fields match expected layout for multiple rows and k offsets
```

Fail/block:

```text
lane mapping cannot reproduce wave64 loader mapping
scale/min unpack differs from reference
```

## Required Evidence Row

Every probe should emit or expose:

```text
schema
candidate_id
backend_atom_id
probe_kind
shape
production_dispatch_changed=false
default_route=direct_packed
owner_fragment_count
covered_output_count
expected_output_count
duplicate_store_count
missing_store_count
expected_owner_hash
actual_owner_hash
resources if available: vgpr, sgpr, lds_bytes, scratch_bytes
source_hash
status: PASS | FAIL | BLOCKED
blocker if not PASS
```

## Integration Points

Likely files:

```text
extra/qk/mmq_q4k_q8_atom.py
extra/qk/mmq_llama_oracle.py
test/unit/test_mmq_q4k_q8_atom.py
test/unit/test_mmq_llama_oracle.py
```

Do not wire into:

```text
tinygrad/llm/prefill_routes.py
tinygrad/llm/route_policy.py
extra/qk/mmq_machine_search.py promotion rows
```

Machine-search rows can be updated only after a probe is stable and bounded.

## Stop Conditions

Stop and report blocked if:

```text
store-only ownership cannot be represented
owner coverage cannot be measured
all store-only attempts produce duplicate/missing stores
all feasible accumulator mappings spill or cannot be represented
```

The correct outcome can be "blocked." A blocked result must name the missing
tinygrad primitive/API, not just "ASM is hard."
