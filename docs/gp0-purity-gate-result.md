# GP0 — Purity Gate Result

Date: 2026-07-01

## Verdict: PASS

The block tile kernel (`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`,
`extra/qk_flash_decode.py:954`) is a **generated UOp kernel** — it builds a UOp
computation graph parameterized over `(Hd, Hq, Hkv, MAXC, L, S, Tc)` and executes
via `Tensor.custom_kernel`. No handwritten RDNA3/HIP assembly. The GP0 forbidden
pattern does not apply.

## Evidence

The kernel uses:
- `UOp.range(...)` for loop structure (GLOBAL/LOCAL/REDUCE axis types)
- `UOp.placeholder(..., addrspace=AddrSpace.LOCAL)` for LDS buffers (ksh, vsh, acc, den, mx)
- `UOp.barrier(UOp.group(...))` for synchronization
- `UOp(Ops.CUSTOMI, ...)` for `__builtin_amdgcn_fdot2` dot product
- All compute expressed as UOp graph nodes, lowered by tinygrad's ISA backend

Parameterization: `G = Hq // Hkv; WARPS = G` — the kernel already supports any G.
For G=5 (14B: Hq=40, Hkv=8), WARPS=5, THREADS=160. No G=4 hardcode.

## Separate handwritten kernel (not in scope)

`extra/qk_owned_flash_decode.hip` is a handwritten RDNA3 kernel used by the
default-on `DECODE_ATTN_AMDGCN_TILE=1` route (8B shape only, Hq=32). This is a
separate route and is NOT the kernel being extended in the GP track. The GP track
extends `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` only.

## AMDISARenderer

`tinygrad/renderer/isa/amd.py:614` — a generic ISA renderer that lowers UOps to
RDNA3 instructions. Phase H adds `DEFINE_VAR` (runtime scalar) support. The block
tile kernel uses `custom_kernel` (via `Ops.PROGRAM` injection), not `AMDISARenderer`
directly, but both paths are generated — neither is handwritten assembly.
