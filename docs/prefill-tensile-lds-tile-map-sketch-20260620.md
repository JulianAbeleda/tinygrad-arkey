# Prefill Tensile LDS Tile-Map Sketch

Date: 2026-06-20

## Verdict

`PASS_LDS_TILE_MAP_SKETCH_NON_BITEXACT`

We now have a source-backed LDS region sketch for the selected rocBLAS ffn_gate/up Tensile kernel. It is enough to
design a non-bitexact tinygrad schedule object, but not enough to claim a byte-for-byte clone of Tensile's generated
layout.

The main finding is that the selected kernel uses a lower A/B operand region plus a second prefetch buffer whose base
is rounded up to a power-of-two element offset. That explains both:

- the observed low LDS offsets around `0..2352`; and
- the observed high LDS offsets around `16384..18736` bytes.

## Source Anchors

| source | relevant rule |
|---|---|
| `/home/ubuntu/rocm-libraries-tensile-sparse/shared/tensile/Tensile/SolutionStructs.py` `getLdsNumElements` | tensor LDS elements = `MacroTile * _DepthULds + padding` |
| `SolutionStructs.py` LDS offset assignment | with `PrefetchGlobalRead`, B follows aligned A; second buffer base is power-of-two rounded from `A+B` |
| `LraTileAssignment.py` `LraTileAssignmentMFMA` | local read address uses `UnrollMajorLDS`, `_DepthULds`, `MIInputPerThread`, `LocalReadVectorWidth`, wave/tile/block offsets |
| `LocalRead.py` `LocalReadMFMA` | local read instruction offsets are derived from vector/tile/read indices and `LocalReadVectorWidth`; `ds_load_b128` comes from `LRVW16` on fp16 |
| `KernelWriterAssembly.py` `lwaFirstOffset` / `lraFinalOffset` | local write/read addresses apply LDS offsets, padding, and B-region base |

## Selected Parameters Used For The Sketch

From `bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json` and the kernel name:

| field | value |
|---|---|
| `MacroTile` | `[128, 128, 1]` |
| `DepthU` / `_DepthULds` | `16` |
| `ThreadTile` | `[4, 64]` |
| `WorkGroup` | `[32, 4, 1]` |
| `PrefetchGlobalRead` | `1` |
| `LocalReadVectorWidth` | `16` |
| `GlobalLoadVectorWidthA/B` | `4 / 4` |
| `UnrollMajorLDSA/B` | `0 / 1` |
| `LdsPadA/B` | `0 / 8` from `LPA0/LPB8` |
| `LdsBlockSizePerPadA/B` | `0 / 128 bytes` from `LBSPPA0/LBSPPB128` |
| element size | fp16 = `2` bytes |

## Region Size Calculation

Tensile computes LDS size in elements, then metadata reports bytes.

For A:

```text
A elements = MacroTileA * DepthU = 128 * 16 = 2048 elements
A bytes    = 4096
```

For B, `LdsBlockSizePerPadB=128 bytes = 64 fp16 elements`, `LdsPadB=8` elements:

```text
B raw elements    = MacroTileB * DepthU = 128 * 16 = 2048
B padding blocks  = 2048 / 64 = 32
B pad elements    = 32 * 8 = 256
B total elements  = 2304
B bytes           = 4608
```

Tensile aligns A to its LDS alignment. Here A is already aligned:

```text
LdsOffsetA      = 0 elements      = 0 bytes
LdsOffsetB      = 2048 elements   = 4096 bytes
```

Because `PrefetchGlobalRead=1`, Tensile allocates a second LDS buffer. The second-buffer base is rounded up to the
next power of two after `A + aligned B`:

```text
lower A+B elements       = 2048 + 2304 = 4352
second-buffer base elems = pow2_ceil(4352) = 8192
LdsOffsetA_Blk           = 8192 elements  = 16384 bytes
LdsOffsetB_Blk           = 10240 elements = 20480 bytes
total LDS elements       = LdsOffsetB_Blk + B elements = 10240 + 2304 = 12544
total LDS bytes          = 12544 * 2 = 25088
```

This exactly matches the selected kernel metadata: `group_segment_fixed_size=25088`.

## Region Map

| logical region | element base | byte base | element span | byte span | role |
|---|---:|---:|---:|---:|---|
| A lower buffer | `0` | `0` | `2048` | `4096` | current or prefetched A tile |
| B lower buffer | `2048` | `4096` | `2304` | `4608` | current or prefetched B tile with padding |
| unused/alignment gap | `4352..8191` | `8704..16383` | `3840` | `7680` | power-of-two second-buffer alignment gap |
| A second buffer | `8192` | `16384` | `2048` | `4096` | alternate A tile for `PGR1` |
| B second buffer | `10240` | `20480` | `2304` | `4608` | alternate B tile for `PGR1` |

## Observed Offset Reconciliation

From `bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json`:

| observed family | interpretation |
|---|---|
| `ds_store_b64` low offsets `0,256,288,512,576,768,864` | low-buffer stores within the active operand region; B base may be carried in address VGPR rather than immediate |
| `ds_store_b64` high offsets `16384,16640,16672,16896,16960,17152,17248` | second-buffer stores relative to `LdsOffsetA_Blk=16384 bytes` |
| `ds_load_b128` low offsets `0,16,32,48,2304,2320,2336,2352` | lower-buffer local-read instruction immediates for WMMA fragments |
| `ds_load_b128` high offsets `16384,16400,16416,16432,18688,18704,18720,18736` | second-buffer local reads with the same low-offset pattern plus `16384` byte buffer base |

Important caveat: disassembly immediates are not the full logical address. Tensile often carries A/B base offsets in
address VGPRs and adds immediates per local-read/local-write instruction. Therefore these offset families identify
regions and stride structure, not a complete per-element coordinate map.

## Layout Rules For A Tinygrad Non-Bitexact Candidate

The first native schedule object should encode these rules rather than copy raw immediates:

| rule | candidate requirement |
|---|---|
| two operand regions | allocate logical A and B regions in LDS |
| B padding | model B padding as `8` fp16 elements per `128` byte block, or choose an explicit non-bitexact padding policy with structural gates |
| PGR double buffer | allocate second A/B region at a power-of-two aligned base; selected authority uses `16384` byte second-buffer base |
| vectorized LDS stores | produce visible `ds_store_b64` or an intentionally chosen wider variant, then gate it structurally |
| vectorized LDS reads | produce `ds_load_b128` feeding WMMA operand VGPRs |
| staged pipeline | separate lower/current buffer reads from next-buffer writes with dependency-aware waits/barriers |
| resource envelope | stay scratch/private-free and close to `LDS=25088`, `VGPR=256`, `SGPR=58` |

## What Is Still Not Known

| unknown | why it remains |
|---|---|
| exact A/B logical coordinate for every `ds_load_b128` | source formulas depend on generated `tP`, wave/tile indices, and address VGPR state not fully reconstructed here |
| exact loop-carried buffer swap points | current segmentation is first-WMMA/last-WMMA heuristic, not labeled Tensile loop structure |
| exact bank-conflict rationale beyond B pad | source shows padding policy, but we have not replayed the generator with all resolved kernel fields |

## Build Readiness

This is enough to scope the tinygrad AMD GEMM schedule object:

```text
lds_layout:
  A0 base 0 bytes, span 4096
  B0 base 4096 bytes, span 4608, padded
  A1 base 16384 bytes, span 4096
  B1 base 20480 bytes, span 4608, padded
  total LDS 25088 bytes

pipeline:
  global_load -> ds_store_b64 into one buffer
  barrier/wait
  ds_load_b128 from the other/current buffer
  v_wmma consume
  swap buffers across K slices
```

It is not enough to claim a bit-identical Tensile clone or to run machine search. Search becomes meaningful after this
layout is represented as a schedule object with resource-gated lowering.

