# Decode fused-MMVQ integration B2 runtime/cache result - 2026-06-19

Purpose: execute PCG-1 / FMI-4 B2 from `primitive-coverage-gap-scope-20260619.md`: determine whether the remaining
decode MMVQ transfer loss is caused by an in-model runtime/cache/program identity mismatch.

No kernels were built. No model route or default changed.

Artifacts:

- `extra/qk_decode_runtime_cache_identity.py`
- `bench/qk-decode-fused-mmvq-integration/runtime_cache_identity.json`
- `bench/qk-decode-fused-mmvq-integration/runtime_cache_identity_summary.md`

## Verdict

`B2_CLOSED_NO_RUNTIME_CACHE_MISMATCH`.

Representative high-share MMVQ roles reuse the same program/cache/launch identity in-model and in direct same-process
role calls. The remaining decode gap is **not** a bounded runtime-cache wiring issue.

| role | identity match | in-model variants | direct-call variants |
|---|---:|---:|---:|
| `attn_q/o` | `true` | `1` | `1` |
| `ffn_gate/up` | `true` | `1` | `1` |
| `ffn_down` | `true` | `2` | `2` |
| `lm_head` | `true` | `1` | `1` |
| `attn_k/v` | `true` | `1` | `1` |

`ffn_down` legitimately has two variants in this model state, and both are covered:

- Q4_K partial: `q4k_gemv_partial_4096_12288_4`, launch `[4,128,1] x [32,1,1]`;
- Q6_K coop: `q6k_coop_partial_4096_12288`, launch `[1024,1,1] x [4,16,1]`.

## Method

The probe installs two temporary hooks:

- `tinygrad.engine.realize.get_runtime`: records runtime cache key, program name, object id, launch metadata, lib hash,
  and whether the runtime existed before lookup.
- `tinygrad.runtime.support.hcq.HCQProgram.__call__`: records actual HCQ dispatch program name, global size, local size,
  and values length.

It then compares:

1. a warm in-model decode step through `model.logits`;
2. direct calls to representative installed Q4_K/Q6_K linears in the same process.

Identity key:

```text
program_name + global_size + local_size
```

The artifact also stores ast/cache metadata for audit, but the gate intentionally ignores ast-key differences when the
same program object/launch contract is reproduced by the direct role call. This avoids false mismatches from graph
construction provenance while still catching actual launch/program changes.

## Decision

Close B2.

Do not spend more time on runtime cache, graph identity, or existing launch-shape knobs as the base-decode unlock.
FMI-4 B1 already killed env knobs; B2 now kills runtime/cache identity as the hidden wiring bug.

The remaining large decode path is one of:

- renderer/scheduler work to natively preserve the low-VGPR, high-grid MMVQ contract;
- decode MMVQ artifact/import discovery if a mature Q4_K/Q6_K family exists;
- keep q8 artifact route as the small research-pass route (`~5-6%`, lossy/default-off).
