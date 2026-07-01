# G=5 Block Tile Pathology Result

Date: 2026-07-01  
Candidate: `decode_flash_block_tile_g5_native_context`  
Artifact: `bench/g5-block-tile/compiler_pathology_v1.json`

## Measurement

| metric | G=5 block tile | baseline gqa_coop_vec |
|--------|---------------|----------------------|
| time_us/workgroup | 2090 | 27 |
| raw slowdown | 77.4× | — |
| positions per WG | 128 (FLASH_L=128) | 32 (FLASH_L=32) |
| work-adjusted slowdown | **~19.4×** (2090 / 27×4) | — |
| barrier_count | 8 (1 per NB=8 tiles) | 32 (1 per j=32 positions) |
| lds_bytes | 8192 (K+V) | 256 (K only) |
| scratch_bytes | null | null |
| vgpr | null | null |

## What static analysis ruled out

- **BARRIER_FLOOD:** G=5 has 8 barriers vs baseline's 32. Barriers are NOT the cause.
- **LAYOUT_MISMATCH:** WARPS=G=5 is intentional. The grid structure is correct.
- **REGISTER_SPILL:** Unconfirmable without VGPR/scratch counts; LDS allocation (8KB) is small.

## BoltBeam diagnose output

Classification: **`UNKNOWN`**  
Reason: `time_ratio=77.4×, scratch=None, barriers=8, static_instr=None`  
Next action: manual disassembly review  

## Root hypothesis (unconfirmed)

The 19.4× work-adjusted slowdown likely comes from one or both of:

1. **LDS bandwidth**: G=5 stages BOTH K and V into LDS (8KB/tile) vs baseline K-only (256B).
   At FLASH_L=128 → NB=8 tiles → 8×8KB=64KB of LDS traffic per WG vs baseline 8×256B.
   32× more LDS traffic per WG.

2. **VGPR pressure**: Online softmax state across G=5 heads simultaneously requires more live
   registers. If VGPR count exceeds the file (256 VGPRs per wavefront on gfx1100), spill to
   scratch adds global memory traffic.

## What is needed to confirm

- `vgpr_count` from ISA disasm (`.amdhsa_next_free_vgpr`)
- `scratch_bytes` from ISA metadata (`.amdhsa_private_segment_fixed_size`)  
- `static_counts.lds` from ISA (count of DS_LOAD/DS_STORE instructions)
- Source: rocprof kernel metadata, or `llvm-objdump --amdgpu-decode-metadata` on compiled ELF,
  or tinygrad ISA native backend (Phase H, commit a860a6178)

## Reopen condition

`decode_flash_block_tile_g5_native_context` reopens when:
1. Dynamic artifact provides VGPR + scratch counts — if VGPR spill confirmed, fix by tiling smaller
2. Or: native ISA implementation of G=5 block tile that avoids the LDS double-bandwidth cost
   (K staging only, V read direct from L2 as in baseline gqa_coop_vec)
