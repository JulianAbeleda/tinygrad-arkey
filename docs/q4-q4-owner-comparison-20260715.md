# Q4 owner comparison: fused `sudot4` versus iu8 WMMA

## Decision

`sudot4` is a useful bounded stepping stone for Q4 arithmetic and packed-ABI
validation, but it is not a WMMA replacement. WMMA integration remains
mandatory for a viable Q4 production path.

This conclusion is intentionally limited to the bounded `(M,N,K)=(16,16,256)`
tile and the current AMD/gfx1100 checkout. No emitter or route selector was
changed for this comparison.

## Apples-to-apples evidence

Both candidates use the same deterministic Q4_K words, the same Q8_1 input,
the same dequantized CPU oracle, and the same output tile.

| property | fused Q4 `sudot4` | existing iu8 WMMA substrate |
|---|---:|---:|
| numeric result | PASS, relative RMSE `1.17e-7` | PASS in `test_q4k_wmma_value.py` |
| generated instruction proof | 31 `sudot4` sites in the emitted kernel source | `v_wmma_i32_16x16x16_iu8`, signed flags asserted by `test_amd_isa_wmma.py` |
| measured kernel count | 2 (one Q8 pack plus one DS4 dot4x4) | 8 for the vectorized bounded graph |
| measured device time | 0.0225 ms | 0.0840 ms |
| compile time | 5.2 ms | not used as a promotion metric |
| LDS/resource contract | no shared-memory staging; logical geometry is 32 threads/wave and 1,536 declared LDS-window bytes | no comparable physical resource snapshot was available from this run |

The fused result is faster in this tiny tile, but the comparison is not a
whole-linear win: its two-kernel count includes a separate activation packing
kernel, and the candidate has no cooperative tile reuse. The WMMA count is
inflated by the current bounded vectorized graph’s prerequisite kernels; its
instruction evidence is nevertheless the only evidence here for tensor-core
execution.

The fused source contains the exact intrinsic spelling
`__builtin_amdgcn_sudot4(true, ..., true, ..., ..., false)`. The WMMA source
contains the exact `v_wmma_i32_16x16x16_iu8` form, including signed A/B flags.
These are generated-source observations, not inferred labels.

## Resource interpretation

The fused atom is register/reduction based and explicitly reports
`shared_memory_staging=False`; its logical LDS windows are a contract
placeholder, not proof of allocated LDS. The WMMA path has no physical LDS,
VGPR, SGPR, spill, or occupancy row in this bounded comparison artifact.
Consequently neither candidate gets a resource promotion claim from missing
compiler metadata. A future owner must capture code-object resources for both
paths under the same launch geometry before making a production claim.

## Ownership consequence

Keep `sudot4` as the Q4 stepping-stone gate: it proves nibble packing, Q8
metadata/sum handling, signedness, and a runnable packed dot. Do not promote it
as the Q4 route. The production candidate still needs a cooperative
Q4/Q8-tile lifecycle that reaches legal iu8 WMMA (or an equivalent tensor-core
path), with resource-backed evidence at role shapes.

Reproduce the fused gate with:

```sh
python3 extra/qk/q4k_fused_q4_correctness_gate.py
python3 -m pytest -q test/unit/test_q4_q4_owner_comparison.py
```
