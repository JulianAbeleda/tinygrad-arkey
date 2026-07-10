# 14B MMQ Renderer ASM Proof Introspection Scope

Purpose: add the smallest renderer-level proof surface needed to turn the R4
static ownership probes into runtime ASM evidence.

## Problem

The R4 store and sum-slot probes proved that llama-style ownership can be
represented statically. The remaining blocker was lower-level:

```text
For each emitted accumulator read/write and output store, can we recover the
logical source identity and the physical register/store instruction after AMD
ISA lowering?
```

Without this, a generated kernel can look structurally correct while the final
instruction stream has lost the `sum[]` slot identity, reused the wrong VGPR, or
stored a value whose source cannot be tied back to the intended owner.

## Implemented Surface

`tinygrad.renderer.isa.amd` now exposes an opt-in proof manifest:

```text
AMD_ISA_PROOF_MANIFEST=1
reset_amd_isa_proof_manifest()
amd_isa_proof_manifest()
```

Default behavior is unchanged. The manifest records rows only when the env flag
is enabled.

Recorded instruction classes:

```text
ACCUM_READ
ACCUM_WRITE
V_WMMA
GLOBAL_STORE
```

Each row includes:

```text
schema
kind
logical_op
emitted
dest_reg
source_regs
```

Accumulator rows additionally preserve:

```text
carrier_kind
define_reg_id
element
pin_vgpr
source_pin_vgpr / dest_pin_vgpr
```

WMMA rows additionally preserve:

```text
a_vgpr_range
b_vgpr_range
c_vgpr_range
accumulator_in_place
```

Global store rows additionally preserve:

```text
store_lane
itemsize
byte_offset
addr_vgpr
data_vgpr
saddr_sgpr_pair
```

## What This Proves

This is enough to prove that a generated R4 candidate's post-regalloc AMD ISA
lowering still carries:

```text
sum slot / DEFINE_REG element identity -> pinned physical VGPR
WMMA C fragment range -> in-place accumulator output
global store lane -> physical data VGPR and emitted store instruction
```

That closes the previous "blocked_missing_physical_slot_introspection" gap for
bounded ASM proof probes.

## What It Does Not Prove

This does not prove MMQ numeric correctness or performance by itself.

Still required for R4:

```text
1. Build a bounded generated candidate that uses the llama ownership law.
2. Enable AMD_ISA_PROOF_MANIFEST=1 during compile.
3. Join manifest rows against llama_mma_sum_slot_mapping and the store owner map.
4. Prove no missing/duplicate stores and matching sum-slot-to-store identity.
5. Only then run numeric Q4_K x Q8_1 bounded comparisons.
```

## Stop Rule

If a candidate cannot produce manifest rows for all expected accumulator/store
events, it is blocked before performance testing. Silent fallback is not a
valid R4 result.
