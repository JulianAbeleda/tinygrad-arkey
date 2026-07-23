# LDS-rotating PV accumulator primitive scope

## Goal

Keep one QK/online-softmax pass per KV tile while limiting live PV C state to
two Hd16 blocks (`16 fp32 VGPRs`) per wave. Full Hd128 accumulator state is
explicitly staged in wave-private LDS; this is a residency experiment, not a
production default.

## LDS ABI

One wave owns 16 query rows.  Each of eight Hd16 blocks has eight WMMA C
lanes, so the full accumulator is `8 * 8 * 4 = 256 bytes` per workgroup.

```text
lds acc[output_block][lane][element] : fp32
offset = ((output_block * 8 + lane) * 8 + element) * 4
```

The first two-block window uses blocks `0,1`; subsequent windows are `2,3`,
`4,5`, and `6,7`. LDS is wave-private because local size remains 32: no
workgroup barrier is legal or required. Publication/reload uses the existing
wave LDS wait contract.

## Pseudocode

```text
initialize m=-inf, l=0 in VGPR; initialize lds.acc[:] = 0
wave_wait_after_lds_init()
for kv_tile:
  score = QK_WMMA(q, k[kv_tile])                 # 8 chained QK WMMA
  p, new_m, new_l, alpha = online_softmax(score, m, l)
  m, l = new_m, new_l
  for output_block_base in (0, 2, 4, 6):
    c0, c1 = lds_load(acc[base], acc[base+1])
    wave_wait_after_lds_load()
    c0 = PV_WMMA(p, v[kv_tile, base], alpha*c0)
    c1 = PV_WMMA(p, v[kv_tile, base+1], alpha*c1)
    lds_store(acc[base], c0); lds_store(acc[base+1], c1)
    wave_wait_before_next_window()
finally:
  for output_block_base in (0,2,4,6):
    c0,c1 = lds_load(...); wave_wait(); drain normalized c0,c1 to output
```

`alpha` is applied to the prior accumulator **before** each PV WMMA C input.
`p` remains unnormalized tile probability relative to `new_m`; final divide
by `l` occurs only in drain. No score/probability buffer is materialized.

## Register ownership

```text
PV C window: v8..v23              # two vec8 C fragments
m/l:         existing v72..v87
QK C:        existing v88..v95
fragment A/B and address ABI: unchanged
```

The experiment must prove physical allocation falls below the current
two-block direct candidate, not merely that virtual C aliases are compact.

## Affected ABI points

- New `AMDAttentionRotatingPVSpec(acc_blocks=2, total_blocks=8, lds_bytes=256)`.
- Loop-state lowering distinguishes register `m/l` from LDS-backed `acc`.
- Typed wave-private LDS load/store and wait markers; reject multi-wave grids.
- Slice-aware PV fragment construction reuses prebiased-V offset support.
- Final drain accepts sequential two-block LDS reloads, not eight simultaneous
  C sources.
- Capture records 256-byte LDS, explicit waits, eight PV roles, and no spill
  or private allocation.

## Fail-closed gates

- only gfx1100, wave32, local size 32, Hd128, and exactly eight output blocks;
- reject multi-wave, missing/duplicated wave waits, workgroup barriers, or
  any implicit scratch/private memory;
- exact full-output numeric comparison against fp32 reference;
- no score/probability allocation; eight QK and eight PV role records;
- require lower VGPR allocation and higher calculated residency before replay.
