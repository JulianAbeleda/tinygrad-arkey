# AMD dtype-orthogonality validation gate — pending (2026-07-29)

## Status

- Code migration on EXP: **complete for the custom AMD attention bridge**.
- CPU and Metal regression: **pass**.
- Physical AMD validation: **pending**.
- Promotion beyond EXP: **not complete until this checklist passes**.

The migration is recoverable from the preceding EXP checkpoints. Do not restore vector dtypes merely
because this gate is pending; validate the descriptor-owned scalar-plus-shape representation first.

## Why AMD must be checked

The changed custom UOps lower before final instruction selection, so their final WMMA and register
carriers should be identical. macOS can prove construction, SPEC=3 contracts, rewrite completion, and
compile-oriented structure, but it cannot prove gfx1100 runtime behavior, VGPR allocation, LDS ordering,
or final ISA identity.

The final generic `Ops.WMMA` carrier is intentionally still represented as `float.vec(8)`. An EXP
scalarization trial exposed a real dependency: associative symbolic rewrites currently use dtype identity
to keep a complete `(8,)` WMMA value separate from one scalar projected lane. Removing that identity before
those rewrites become shape-aware produced a `(8,)` versus `()` graph mismatch. This is the next code
migration boundary; it is not an AMD runtime failure and must not be papered over with broadcasting.

## Required host

- The known gfx1100 AMD system used for the Qwen hot-path evidence.
- The same tinygrad EXP commit and model artifact used for comparison.
- LLVM inspection tools available (`llvm-objdump` and `llvm-readelf`, or the repository-supported ROCm
  equivalents).

Record device identity, kernel/driver version, commit SHA, model checksum, and command environment with
the result.

## Gate A — focused contracts

Run the focused descriptor, lowering, register, and generated-program tests:

```bash
SPEC=3 python -m pytest -q \
  test/unit/test_amd_wave_lds_fence.py \
  test/unit/test_amd_attention_kv_tile_oob_guard.py \
  test/unit/test_precontract_int8_lds_contract.py \
  test/unit/test_attention_residency_contract.py \
  test/unit/test_shared_attention_compiler_capture.py
```

Acceptance: all applicable tests pass. Any known unrelated test must be identified by exact test id and
reproduced on the checkpoint commit before it may be excluded.

## Gate B — final-program equivalence

For representative QK/PV and packed Q4 roles, compare the final generated program against the preceding
checkpoint, ignoring only source-level scalar/vector spelling metadata.

Acceptance:

- Same WMMA instruction family and count.
- Same launch geometry and LDS allocation.
- Same load/store widths and address expressions.
- No new barriers or waits.
- No spills.
- VGPR/SGPR counts do not regress.

Store the comparison as a content-addressed evidence artifact; do not rely on console output alone.

## Gate C — numerical execution

Run the native attention and Q4 packed-weight numerical fixtures on AMD, including aligned and ragged KV
geometry.

Acceptance: identical pass/fail classification and existing numerical tolerances. A compiler-only result
does not satisfy this gate.

## Gate D — model hot path

Run the standard Qwen 8B AMD smoke and benchmark workflow used by the repository's current README/handoff,
with machine-search route evidence enabled.

Acceptance:

- Model loads and produces correct tokens.
- The expected machine-search-selected hot path fires.
- No fallback or hand-authored kernel is silently selected.
- Prefill/decode classification and memory residency match the checkpoint.
- Record performance, but treat normal run-to-run noise separately from a structural regression.

## Completion record

When A-D pass, change this document's status to **complete**, add the evidence artifact paths and commit
SHA, and update the parent migration document. Only then may the dtype-orthogonality change be described
as AMD-validated or promoted beyond EXP.
