# Decode DNR-4 T2 Dot-Body Scope - 2026-06-20

Verdict: `PASS_DNR4_T2_DOTBODY_SCOPE_READY`

DNR4-T1 proved that the high reduction/tail band is not the main decode gap. T2 moves upstream into S2/S3: q4/q8 vector load, nibble select, dot4, scale/min, and live-range packing.

## Target

Prior DNR-3C2 closed the global-load budget but used `v80-v95` for the preloaded q4/q8 words. That proved the load shape, but it is the wrong resource shape. T2 keeps the b128 preload primitive and packs it into low registers that are dead after scale/min setup.

Candidate map:

| role | registers |
| --- | --- |
| q4 lanes 0-7 | `v12-v19` |
| q8 lanes 0-4 | `v25-v29` |
| q8 lanes 5-7 | `v33-v35` |
| accumulators | `v4-v5` |
| scale/min payload | `v30-v32`, `v36-v37` |
| scratch | `v10-v11` |

Then combine it with DNR4-T1 reduction reuse so the tail does not reintroduce `v50-v54`.

## Gates

- no `v80-v95` vector band;
- static max VGPR index `<=41`;
- 16 `dot4`;
- grouped global loads `<=11`;
- DNR4-T1 low reduction/tail reuse;
- synthetic correctness;
- real GGUF correctness;
- same-harness timing movement before any promotion.

Promotion remains timing-gated: `>=30us` vs native, `>=15us` vs best static, or `>=10us` vs C7C.

Probe: `extra/qk_decode_dnr4_t2_dotbody_scope.py`

