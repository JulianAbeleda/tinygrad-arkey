# Prefill Tensile Schedule Template Extraction Result

Date: 2026-06-20

## Verdict

`PASS_SCHEDULE_TEMPLATE_EXTRACTED_NOT_SEARCH_READY`

The selected rocBLAS ffn_gate/up Tensile solution row is now decoded from the installed `.dat`, and the selected
function disassembly is segmented into a conservative schedule-template artifact.

This closes the immediate "do we know the actual selected solution row?" gap. It does **not** make BEAM/search ready,
because the source-level logical LDS tile map and exact steady-state loop schedule are still not reconstructed.

## New Artifacts

| artifact | purpose |
|---|---|
| `extra/qk_tensile_schedule_template_extract.py` | decodes the selected Tensile `.dat` row and segments selected disassembly |
| `bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json` | machine-readable selected solution row plus schedule-region counts |

Command:

```bash
python3 extra/qk_tensile_schedule_template_extract.py
```

## Selected `.dat` Solution Row

The installed rocBLAS `.dat` is msgpack:

`/opt/rocm-7.2.4/lib/rocblas/library/TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1100.dat`

The selected symbol matched exactly:

| field | value |
|---|---|
| `.dat` solution array index | `52 / 290` |
| Tensile solution index | `1140853605` |
| operation | `Contraction_l_Ailk_Bljk_Cijk_Dijk` |
| types | A/B/C/D `Half`; high-precision accumulate `true` |
| strided batched | `true` |
| beta | `true` |

Size mapping:

| field | value |
|---|---|
| `macroTile` | `[128, 128, 1]` |
| `threadTile` | `[4, 64]` |
| `workGroup` | `[32, 4, 1]` |
| `depthU` | `16` |
| `globalSplitU` | `1` |
| `streamK` / `streamKAtomic` | `0 / 0` |
| `globalAccumulation` | `0` |
| `workspaceSizePerElemC` | `0` |
| `workGroupMapping` | `8` |
| `staggerU` / `staggerStrideShift` | `0 / 0` |
| `magicDivAlg` | `2` |
| `preloadKernargs` | `0` |

This confirms the selected ffn_gate/up route is a workspace-free, non-StreamK, non-GSU, shape-specialized GEMM row
with the same macro/thread/workgroup/depth fields inferred from the kernel name.

## Disassembly Segmentation

Input disassembly:

`/tmp/td_all.txt`

Selected function range:

| field | value |
|---|---|
| function lines | `282071..289317` |
| function line count | `7247` |
| first `v_wmma` | line `282555` |
| last `v_wmma` | line `283735` |
| `v_wmma` count | `80` |

Segmentation heuristic:

`prologue = before first v_wmma; steady = first through last v_wmma; epilogue = after last v_wmma`

This is conservative. It is good enough to separate operand staging and compute-bearing regions, but it is not yet a
symbolic loop reconstruction.

| region | line range | key counts |
|---|---:|---|
| prologue before first WMMA | `282071..282554` | `buffer_load_b64=16`, `ds_store_b64=16`, `ds_load_b128=8`, `s_waitcnt=12`, `s_barrier=1` |
| steady first-to-last WMMA | `282555..283735` | `v_wmma=80`, `buffer_load_b64=8`, `buffer_load_d16_b16=16`, `buffer_load_d16_hi_b16=16`, `ds_store_b64=16`, `ds_load_b128=32`, `s_waitcnt=17`, `s_barrier=5` |
| post-WMMA tail | `283736..289317` | `buffer_load_d16_b16=256`, `s_waitcnt=516` |

LDS offsets by region:

| region | `ds_store_b64` offsets | `ds_load_b128` offsets |
|---|---|---|
| prologue | `0, 256, 288, 512, 576, 768, 864, 16384, 16640, 16672, 16896, 16960, 17152, 17248` | `0, 16, 32, 48, 2304, 2320, 2336, 2352` |
| steady | `0, 256, 288, 512, 576, 768, 864` | `0, 16, 32, 48, 2304, 2320, 2336, 2352, 16384, 16400, 16416, 16432, 18688, 18704, 18720, 18736` |
| post-WMMA tail | none | none |

## What This Adds To The Transfer Table

Before this pass, the transfer table relied on kernel-name tokens and launch metadata for the solution shape. Now the
actual `.dat` row confirms:

- `macroTile=[128,128,1]`
- `threadTile=[4,64]`
- `workGroup=[32,4,1]`
- `depthU=16`
- `workGroupMapping=8`
- no StreamK, no global split, no workspace

The disassembly segmentation also shows that both prologue and compute-bearing regions contain the LDS staging path:

`buffer_load_b64 -> ds_store_b64 -> ds_load_b128 -> v_wmma`

This strengthens the conclusion that the missing primitive is a scheduled GEMM pipeline, not a standalone LDS store
opcode.

## Remaining Blockers

| blocker | status | why it matters |
|---|---|---|
| source-level LDS tile map | non-bitexact sketch now exists in `prefill-tensile-lds-tile-map-sketch-20260620.md` | enough for schedule-object design, not enough for bit-identical clone |
| exact steady-state loop reconstruction | partial/heuristic | first-WMMA/last-WMMA is not enough to encode a faithful K-loop scheduler |
| bank/padding rationale | still missing | avoids repeating the slow native LDS macro family |
| search surface | missing | BEAM/search cannot help until these rows are represented as schedule dimensions |

## Next Required Artifact

Build the LDS tile-map sketch from Tensile source plus selected offsets:

1. Read `LraTileAssignment.py` and `LocalRead.py` for the selected `UMLDSA0/UMLDSB1/LRVW16/TLDS1` behavior.
2. Map the observed offset families into A/B regions:
   - low offsets around `0..2352`
   - high offsets around `16384..18736`
3. Produce a non-bitexact native layout spec:
   - A/B logical regions;
   - global-load vector shape;
   - LDS store vector shape;
   - LDS read vector shape;
   - WMMA operand VGPR handoff contract;
   - required waits/barriers.

Only after that should we build the tinygrad AMD GEMM schedule object.
