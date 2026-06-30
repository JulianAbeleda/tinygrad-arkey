# TG5 Cross-Target Feature Model -- verdict: **TG5_PASS_TARGET_FEATURE_MODEL_READY**

Target features as DATA (wave/subgroup, vector-dot/matrix-core, LDS, barrier, registers, occupancy, native-ISA backend, compiler ownership, profiling). The candidate author gates on them.

## Targets

| target | wave | subgroup | vector_dot | matrix_core | native ISA | profiling | backend_validated |
|---|---:|---|---|---|:--:|:--:|:--:|
| amd_gfx1100 | 32 | wave32 | v_dot4_i32_i8,v_dot4_u32_u8,v_dot2_f32_f16 | WMMA_rdna3 | True | True | True |
| nvidia_sm89 | 32 | subgroup32 | dp4a,mma_m16n8k16 | TensorCore_ada | False | False | False |
| apple_metal_m3 | 32 | simdgroup | simd_shuffle | simdgroup_matrix | False | False | False |

## Acceptance

- **gfx1100 reproduces route permissions**: lane_extent==32 True; all 20 author candidates TARGET_OK True; families covered True
- **wave64 / subgroup_simdgroup pruned on gfx1100**: True / True (candidate needs wave64 (wave_size=64) but target amd_gfx1100 has wave_size=32)
- **NVIDIA/Metal candidates -> TARGET_BACKEND_INCOMPLETE (never silently OK)**: True

The author can now say 'algorithmically plausible but target lowering is missing' instead of pretending portability:

- **nvidia_sm89**: ['TARGET_BACKEND_INCOMPLETE'] -- algorithmically plausible on nvidia_sm89 (subgroup32) but target lowering/profiling missing: ['native_isa_backend', 'W==D/whole-prefill profiling']
- **apple_metal_m3**: ['TARGET_BACKEND_INCOMPLETE'] -- algorithmically plausible on apple_metal_m3 (simdgroup) but target lowering/profiling missing: ['native_isa_backend', 'W==D/whole-prefill profiling']
