# AMD Broad Backend BB-5a.10 P1 Layout Spec

Date: 2026-06-19

Generator:

- `extra/qk_amd_bb5a10_p1_layout_spec.py`

Artifact:

- `bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json`

## Verdict

`PASS_BB5A10_P1_LAYOUT_SPEC_READY`.

P1 derives the first non-bitexact Tinygrad LDS layout spec from the selected Tensile audit. It keeps the selected
rocBLAS evidence as the target and avoids overfitting to the aggregate Tensile corpus.

## Spec

The candidate layout has two logical LDS regions:

| region | base | observed role |
|---|---:|---|
| `operand_A_or_low_region` | `0` | low-offset selected-kernel LDS stores and `ds_load_b128` reads |
| `operand_B_or_high_stage_region` | `16384` | high-offset selected-kernel LDS stores and `ds_load_b128` reads |

Required lowering features:

- nonzero LDS allocation in generated ELF/source;
- selected-kernel-compatible LDS stores, with `ds_store_b64` accepted for the first rocBLAS authority candidate;
- `ds_load_b128` LDS reads;
- WMMA source operands overlap `ds_load_b128` destination VGPRs;
- dependency metadata for `vmcnt`/`lgkmcnt` waits and barriers;
- scratch/private spill rejection before timing.

Not required:

- bit-identical Tensile LDS byte layout;
- `ds_store_b128` for the first selected rocBLAS authority candidate;
- q8 transfer before pure tinygrad prefill reaches `>=60 TFLOPS`.

Next: run P2/P3/P4/P5 as one batch.
