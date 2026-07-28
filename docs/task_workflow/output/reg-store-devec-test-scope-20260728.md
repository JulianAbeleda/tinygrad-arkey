# Register-store devectorizer test scope

Date: 2026-07-28
Status: required evidence before promoting or consolidating `extra/qk/reg_store_devec.py`

## Objective

Prove the exact behavior of the extra matcher before changing ownership. The matcher rewrites:

```text
STORE(STACK(LOAD(INDEX(REG, ...)), ...), vector_value)
```

into scalar register stores. The existing core matcher `pm_distinct_reg_store_devec` performs a related rewrite but
rejects duplicate REG pointers. The tests must establish whether the extra matcher owns a genuinely distinct residual
shape or is redundant, without assuming either answer from the filenames.

## Unit-test matrix

The direct matcher tests should call `_devec_reg_store` and `pm_reg_store_devec` with hand-built UOps and inspect the
rewritten graph. They must remain CPU-only and must not require an AMD device.

| ID | Input shape | Expected result |
|---|---|---|
| U1 | Distinct `REG.index(const)` targets wrapped as `LOAD`, vector value with matching lane count | Rewrite to one scalar `STORE` per target; target addresses preserve source order and values use matching `value.gep(i)` lanes |
| U2 | Mixed direct `REG.index` and `LOAD(REG.index)` target children | Rewrite all valid children; wrapper normalization must not change lane order |
| U3 | Duplicate REG pointer targets, including duplicate indices | Capture current extra-pass behavior explicitly; require one scalar store per input lane and document whether duplicate writes are intentional, rejected, or consolidated |
| U4 | Non-REG target (`GLOBAL`, `LOCAL`, or other address space) | Return no rewrite; original graph identity/shape remains unchanged |
| U5 | Non-`INDEX` target, malformed `LOAD`, or mixed valid/invalid target children | Return no rewrite; no partial rewrite is allowed |
| U6 | Lane/value ordering and vector widths 1, 2, 4, and 8 | Scalar stores retain exact positional mapping; width mismatches must fail closed rather than create invalid `GEP`s |
| U7 | Empty stack and single-lane stack | Empty/malformed input declines; valid single-lane input has the documented scalar result |

U3 is the key discriminator. Its expected behavior must be written as an intentional contract, not merely copied from
the current implementation. If duplicate pointers are required by a shipped route, retain the behavior or consolidate
it with an equivalent rule plus a regression. If they are not required, bank the evidence before removing it.

## Core-pass interaction tests

Run the same fixtures through both passes in the actual order used by `tinygrad/codegen/__init__.py`:

1. `pm_distinct_reg_store_devec`
2. `pm_reg_store_devec`

Assert that the extra rule receives only the residual shapes left by the core rule. Include a duplicate-pointer case
where the core pass declines, then prove whether the extra pass rewrites it. Include a distinct-pointer case where the
core pass already rewrites and assert the extra pass does not double-rewrite or alter the scalar stores.

## Pipeline-dispatch tests

Use an AMD ISA renderer as a compile-only oracle; no GPU execution is needed.

| Gate/device state | Expected matcher dispatch |
|---|---|
| AMD, `COALESCED_LOAD_LOWERING` unset/off, no kernel metadata flag | Extra matcher not called |
| AMD, `COALESCED_LOAD_LOWERING=1` | Extra matcher called once after core matcher |
| AMD, env gate off but `sink.arg.coalesced_loads=True` | Extra matcher called once |
| CPU, env gate on | Extra matcher not called |
| Non-AMD renderer, metadata flag true | Extra matcher not called |
| AMD, both env and metadata enablement true | Still one invocation, no duplicate rewrite |

Use a dispatch spy around `graph_rewrite` or the matcher entry point. Restore the cached `getenv` state after each
case. The test must prove the gate and target checks, not only that the module imports.

## Compile/render smoke

- Lower a minimal AMD-ISA AST containing the positive U1 shape and verify the final graph has no multi-lane REG target
  stores.
- Render the lowered graph with the HIP/AMD cstyle renderer and assert no `make_floatN(...)` constructor remains on a
  store left-hand side.
- Keep this compile-only; no AMD hardware, GPU lock, or benchmark authority is required for the unit scope.

## Required regression set

Run these existing tests alongside the new focused file:

```text
test/unit/test_coalesced_load_lowering.py
test/unit/test_devectorizer_reconstruction.py
test/unit/test_logits_only_reg_store.py
test/unit/test_gate_inventory.py
test/unit/test_lowering_baseline.py
test/unit/test_lowering_fingerprint.py
test/unit/test_codebase_organization_audit.py
```

The lowering baseline and fingerprint must be regenerated at the exact clean candidate commit. Historical artifact
records containing `COALESCED_LOAD_LOWERING: null` are evidence snapshots and must not be rewritten by this test work.

## Acceptance gates

Promotion/consolidation is allowed only when:

- U1-U7 pass with an explicit duplicate-pointer decision.
- Core/extra pass ordering is pinned and no double rewrite occurs.
- AMD/coalesced-load dispatch is proven and CPU/non-AMD paths remain unchanged.
- Compile/render smoke produces no invalid multi-lane REG store target.
- The existing regression set and regenerated lowering authorities pass.
- The organization manifest, boundary test, docs, and recovery record agree on the final owner.

Until these gates pass, retain `extra/qk/reg_store_devec.py` and its forwarding shim; do not move, delete, or alias it to
`pm_distinct_reg_store_devec`.
