# Precontract target generalization scope

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to `dev`/`master`.

## 1. Why

M1 (`metal-prefill-M-theories-scope-20260730.md`) is blocked: the packed-weight precontract path -- the mechanism
that fuses Q4_K dequant into the WMMA operand and is the only route to tensor cores that fits Metal's memory
budget -- refuses to lower on Metal with `KernelOptError: RDNA3 WMMA descriptor dims drifted`.

**This is not a missing Metal path.** `validate_rdna3_wmma_descriptor` (`tinygrad/codegen/opt/kernel_lds.py:23-36`)
never inspects the device. It asserts the tensor-core descriptor *equals* RDNA3's numbers, so it rejects every
non-RDNA3 backend by shape.

Critically, the values it compares against are a **frozen snapshot of a generic computation**. `tc.py` already
declares per-target descriptors and `lane_map.remaps()` already derives remaps from them. Verified:

    METAL  dims=(8,8,8)    ept=(2,2,2)    remaps={'l0':'r1','l1':'l1','l2':'l2','l3':'r2',...}
    AMD    dims=(16,16,16) ept=(16,16,8)  remaps={'l0':'l4','l1':'u0','l2':'u1','l3':'u2',...}

AMD's remaps are exactly `_RDNA3_REMAPS`. **Metal already has its own, computed, today.** Nobody hand-wrote AMD's
lane map; a general mechanism produced it and someone froze the output as constants.

Probe control (`scratchpad/m1_packed_wmma_metal_feasibility_probe.py`): the same view-chain and forced TC opt,
*without* the precontract machinery, **does** lower on Metal and emits 2 `__WMMA` + 1
`simdgroup_multiply_accumulate`. So generic TC selection and `packed_half_carrier` are already portable. Only the
precontract path is locked.

## 2. Two kinds of coupling, sized

**Kind 1 -- frozen snapshots (generalizable).** `_RDNA3_DIMS/_ELEMENTS/_OPTS/_SWIZZLE/_REMAPS` (12-18), the
validator (23-36), `_RDNA3_ELEMENTS` as a tc-is-None default (295), a second shape check and vec widths (323-327).
Call sites: `kernel_lds.py:226`, `kernel_lds.py:294`, `gemm_consumer.py:124`.

**Kind 2 -- real hardware structure (parameterizable, not generalizable).** `cooperative_store_octet_rows` (414-422)
hardcodes 8 lanes per bank cycle (`8 % vectors_per_row`), and the conflict analysis (426-460) assumes RDNA3's
"32 x 4 B banks" and `32//V` aligned dword quads. Apple's threadgroup memory banks differently.

~60 of 581 lines are materially affected.

## 3. Boundaries

Work goes through **vocabulary -> scheduler -> codegen**. Specifically:

| layer | change |
| --- | --- |
| vocabulary | declare LDS/threadgroup bank facts as target facts, in the TG2 capability shape |
| scheduler | none -- TC selection already reads `self.ren.tensor_cores` and the probe proved it portable |
| codegen | derive descriptor facts from `tc`; parameterize the bank math on the declared facts |

**Do not add a Metal branch, a Metal-specific validator, or a second constant block.** A third frozen snapshot is
the failure this scope exists to remove. The end state is one mechanism driven by declared facts, which makes the
next target free rather than adding a fourth block.

## 4. Evidence contract

1. **AMD behaviour byte-identical.** No AMD hardware here, so prove it by rendered-source equality for a kernel
   that exercises the precontract path, exactly as TG1 did. State the limitation.
2. The generalized validator must still **reject** a descriptor that is genuinely unsupported. Demonstrate a
   rejection; a validator that accepts everything is not a generalization, it is a deletion.
3. Metal must reach the precontract path and emit `__WMMA`. **Search `__WMMA` / `simdgroup_multiply_accumulate`,
   never `simdgroup_matrix`** -- `MetalRenderer` never emits that string and it has already produced one wrong
   conclusion in this campaign.
4. Failing-test-id **sets** diffed, not counts (~114 pre-existing failures).

## 5. Work packages

### PG0 -- Generalize the descriptor validator (Kind 1)

- Replace equality-against-constants with derivation from `tc`: `dims`, `threads`, `elements_per_thread`,
  `lane_map.remaps()`. The guard becomes "self-consistent and supported", not "is RDNA3".
- Fix the three call sites and the tc-is-None default at 295 and the second check at 323-327.
- Acceptance: AMD rendered source byte-identical; an unsupported descriptor still raises; Metal gets past this
  validator (it may still fail later on Kind 2 -- that is PG1, and reaching a *different* error is progress).

### PG1 -- Declare and parameterize the bank facts (Kind 2)

Prerequisite: PG0.

- Declare threadgroup/LDS bank count and access granularity as target facts. AMD: 32 banks x 4 B, 8 lanes per
  b128 bank cycle. Metal: derive from Apple's documented threadgroup memory behaviour; if it cannot be
  established, say so and stop rather than guessing -- a wrong bank model produces silent conflicts, not errors.
- Parameterize `cooperative_store_octet_rows` and the conflict analysis on those facts.

### PG2 -- Prove M1 unblocked

Prerequisite: PG1. Rerun the M1 feasibility probe. GREEN means M1 proceeds as a qualification campaign.

## 6. Non-goals

- Metal-specific precontract code, validators, or constant blocks.
- Qualifying `PACKED_WMMA_ROUTES` rows (that is M1, downstream).
- Promotion to `dev`/`master`.
