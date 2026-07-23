# LDS-rotating PV accumulator primitive scope

## Goal

Keep one QK/online-softmax pass per KV tile while limiting live PV C state to
one Hd16 block (`8 fp32 VGPRs`) per lane. Full Hd128 accumulator state is
explicitly staged in wave-private LDS; this is a residency experiment, not a
production default.

## LDS ABI

One wave owns 16 query rows. Each of eight Hd16 blocks has eight fp32 WMMA C
values **per lane**, so the full accumulator is
`8 blocks * 8 values * 32 lanes * 4 bytes = 8192 bytes` per workgroup. The
previous 256-byte figure omitted the wave32 lane dimension. With the existing
512-byte probability slab, the kernel requests 8704 bytes total LDS.

```text
lds acc[output_block][lane][element] : fp32
offset = ((output_block * 32 + lane) * 8 + element) * 4
```

The first diagnostic uses one block at a time in order `0..7`. LDS is
wave-private because local size remains 32: no
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
  for output_block in 0..7:
    c = lds_load(acc[output_block])
    wave_wait_after_lds_load()
    c = PV_WMMA(p, v[kv_tile, output_block], alpha*c)
    lds_store(acc[output_block], c)
    wave_wait_before_next_window()
finally:
  for output_block in 0..7:
    c = lds_load(...); wave_wait(); drain normalized c to output
```

`alpha` is applied to the prior accumulator **before** each PV WMMA C input.
`p` remains unnormalized tile probability relative to `new_m`; final divide
by `l` occurs only in drain. No score/probability buffer is materialized.

## Register ownership

```text
PV C window: v8..v15              # one vec8 C fragment
m/l:         existing v72..v87
QK C:        existing v88..v95
fragment A/B and address ABI: unchanged
```

The experiment must prove physical allocation falls below the current
two-block direct candidate, not merely that virtual C aliases are compact.

## Affected ABI points

- New `AMDAttentionRotatingPVSpec(acc_blocks=1, total_blocks=8, acc_lds_bytes=8192)`;
  final compiled LDS must be 8704 bytes including the existing P slab.
- Loop-state lowering distinguishes register `m/l` from LDS-backed `acc`.
- Typed wave-private LDS load/store and wait markers; reject multi-wave grids.
- Slice-aware PV fragment construction reuses prebiased-V offset support.
- Final drain accepts sequential two-block LDS reloads, not eight simultaneous
  C sources.
- Capture records 8192-byte accumulator LDS and 8704-byte total LDS, explicit waits, eight PV roles, and no spill
  or private allocation.

## Fail-closed gates

- only gfx1100, wave32, local size 32, Hd128, and exactly eight output blocks;
- reject multi-wave, missing/duplicated wave waits, workgroup barriers, or
  any implicit scratch/private memory;
- exact full-output numeric comparison against fp32 reference;
- no score/probability allocation; eight QK and eight PV role records;
- require `vgpr <= 192`; the arithmetic-only one-block estimate is 198, so a
  candidate must also shorten at least six non-PV live values before replay.

The static cost and admission model is recorded in
`docs/shared-attention-lds-pv-demotion-cost-model-20260724.md`.
