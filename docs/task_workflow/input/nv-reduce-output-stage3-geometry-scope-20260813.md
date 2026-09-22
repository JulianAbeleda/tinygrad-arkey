# NV reduce-output stage 3 scope: llama-geometry norm kernels (corrected ledger)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `882ce66a5`, P2 promotion booked)
Status: **implementation/test scope. Corrects the reduce-output row's premise
with llama's own nsys shape census and source: llama does NOT absorb the
q/k or block norms in-kernel. It runs standalone `rms_norm_f32` kernels at
~1.3 us (q/k) / ~2.88 us (4096). Our fused bodies cost ~3.4 us (q/k) and
~7.97 us (1_4096) for the same work, so the real gap is per-kernel launch
geometry, not kernel presence. Stage 3 target: match llama's per-kernel
geometry while keeping our bitwise serial-chain association (logits sha
`9e6664fd...` must not move). No GEMV absorption: the M1 fold into
`w1w3fused` is closed NO-GO (+81.92 us, cost gate contradicted) and llama
does not do it either.**

## 1. Why this scope exists (corrected premise)

The P2 record (`nv-reduce-output-site-absorption-p2-promotion-record-20260812.md`)
booked the fp32 q/k + FFN-down admission at +52 us wall. The scope doc's
arithmetic priced the row at up to ~201-208 tok/s on the claim that "llama
pays 0 for reduce_output; its mmvq absorbs the output reduce in-kernel".
That claim is FALSE against llama's own evidence:

- llama nsys shape census (`nv-llama-d512-node-ledger-20260812.json`,
  `shape_census`): the q norms render as `rms_norm_f32<256>` grid [32,1,1],
  36 launches x 1.312 us = 47.2 us; k norms as grid [8,1,1], 36 x 1.28 =
  46.1 us; the 4096-dim block norms as `rms_norm_f32<1024>` grid [1,1,1],
  73 x 2.88 = 210.2 us. Total norm family 303.5 us.
- llama source (`ggml/src/ggml-cuda/mmvq.cu`): `has_fusion` is the GLU
  gate/x_bias epilogue, not a norm. The q/k norms come from
  `build_norm(..., LLM_NORM_RMS)` -> `ggml_rms_norm` -> `rms_norm_f32`
  (`ggml/src/ggml-cuda/norm.cu`): one block per row, block_reduce over the
  row, scale + multiply.
- The old ledger double-counted: it charged our 392 us against llama's 0 in
  the matmul row while ALSO crediting our 49.6 us rmsnorm class against
  llama's 307.6 us norms class. The honest comparison is llama 303.5 us vs
  tinygrad 441.6 us (49.6 rmsnorm + 392.0 reduce_output) = **+138.1 us**,
  and the q/k share of that is 240.5 - 93.3 = **+147.2 us** (the 4096 side
  already over-earns: 201.1 vs 210.2).

So stage 3 is not "remove the kernels llama doesn't have"; it is "make the
kernels we both have cost what llama's cost". The bodies are 2.6x (q/k) and
2.8x (1_4096) more expensive per launch than llama's standalone kernels,
almost certainly because of launch geometry: ours is one 32-warp block with
a block barrier and redundant idle lanes; llama's is grid-per-row small
blocks with a tree block_reduce.

## 2. Audit (this session)

| shape | llama kernel | llama launch | llama us/launch | our body | our us/launch |
| --- | --- | --- | ---: | --- | ---: |
| q norm 32x128 | `rms_norm_f32<256>` grid 32 | 36 | 1.312 | `reduce_output_rmsnorm_32_128` | 3.70 |
| k norm 8x128 | `rms_norm_f32<256>` grid 8 | 36 | 1.28 | `reduce_output_rmsnorm_8_128` | 3.17 |
| block/FFN norm 1x4096 | `rms_norm_f32<1024>` grid 1 | 73 | 2.88 | `reduce_output_rmsnorm_1_4096` (19) + decode_norm (17) | 7.97 / 2.92 |

Our body emitter (`tinygrad/codegen/late/reduce_output.py`) emits one block:
multi-row shapes use `warps == rows` local warps, one `UOp.barrier`, and the
NV ordinary partial-chain association (`_NV_MULTI_ROW_ASSOC`). The ordinary
reduce programs the bodies replace (`r_2_8_4_4_16`, `r_8_16_8`) used
per-row 8-lane x 16-serial / 16-lane x 8-serial chains. The single-row
1_4096 body mirrors `r_16_256`: 16 threads x 256 contiguous serial with a
serial partial chain, currently widened to 512 threads with idle lanes.

llama's geometry per launch: grid = rows (independent blocks), block = 256
(q/k) or 1024 (4096) threads, tree `block_reduce` in shared memory.

Constraint: our logits are pinned bitwise to the ORDINARY reduce association
(that is what makes full-logit SHA-256 identical). A per-row block geometry
preserves the per-row serial chain exactly (the association is per-row); a
tree reduce (llama's) would NOT be bitwise-equal and is out of scope.

## 3. Arithmetic (corrected)

Baseline 5.2031 ms/token = 192.19 tok/s production (P2 candidate measured
5.1955 ms = 192.47 on the reverse bracket). Rule: ~25 us/token ~= +1 tok/s.

| site | ours (launch x us) | llama (launch x us) | recoverable at 1:1 |
| --- | ---: | ---: | ---: |
| q/k 128 (32_128 + 8_128) | 240.5 (70 x 3.44 avg) | 93.3 (72 x 1.30 avg) | **+147.2 us** |
| 4096 block norms | 201.1 (19 x 7.97 + 17 x 2.92) | 210.2 (73 x 2.88) | already at parity (-9.1) |
| total norm family | 441.6 | 303.5 | +138.1 us |

If the q/k bodies reach llama's 1.30 us/launch: 240.5 -> 93.3 us, saving
~147 us/token -> **~198.0 tok/s (+5.9)** at 1:1 (pure geometry, no body
work added). If the 19 x 1_4096 bodies additionally reach llama's 2.88 us:
another ~97 us -> **~202 tok/s (+9.7)**. Neither needs the +50 us bar
waived if it lands; the q/k site alone is above the bar, the package is far
above it.

Do NOT relitigate: M1 fold (norm into `w1w3fused` GEMV) NO-GO +81.92;
phase6 single-fused-program NO-GO 18.5 us slower; Q4 FFN-down load pattern
NO-GO. This scope changes ONLY the reduce_output body launch geometry and
leaves every production default untouched until the A/B books.

## 4. Implement plan

### P1: per-row grid geometry for the multi-row bodies (q/k site)

1. Rework `emit_reduce_output` multi-row path: one block per row
   (`gidx0 = row`), block = 32 lanes, the NV partial-chain association
   unchanged (8 lanes x 16 serial stride-8 for 32x128; 16 lanes x 8 serial
   for 8x128), partial publish + barrier + serial combine within the row
   block, epilogue writes the row's `dim` elements (4 per lane for 128).
   The reduction association must be byte-identical UOps per row so the
   full-logit sha cannot move.
2. Keep the kernel names (`reduce_output_rmsnorm_32_128`,
   `reduce_output_rmsnorm_8_128`) so the 08-05 body pin and the P2 census
   contract (91 bodies) stay valid; only the launch shape changes.
3. Hermetic: extend `test/unit/test_generic_reduce_output.py` with a
   geometry pin (grid = rows, block lanes, barrier count) plus the existing
   bitwise digest pins; CPU census must still show the same body counts.

### P2: lean single-row launch for 1_4096 (FFN-down side)

1. Reduce the 1_4096 body from 16 warps x idle lanes to the minimum width
   that preserves the `r_16_256` serial chain (16 active threads + combine),
   keeping the name and the association.
2. Hermetic + CPU census as above (body count stays 19).

### P3: real-token A/B (GPU, lock-held)

1. Single-site reverse control/candidate/control bracket at d512, exact
   full-logit sha `9e6664fd...`, census contract (bodies count identical to
   P2: 91), +50 us bar against both controls (the `nv_reduce_output_fp32_qk_ab.py`
   discipline, self-managed lock).
2. Book the q/k site, the 1_4096 site, or the package per the measured
   bracket; promotion record only on pass.

## 5. Gates (hard stop)

1. Bitwise association per row is preserved (hermetic digest pins + exact
   full-logit sha on the A/B).
2. Census shows the same 91 bodies with zero weight materializations (the
   geometry change must not alter the program-count contract booked in P2).
3. Reverse wall bracket clears +50 us against BOTH controls (or principal
   waiver with the exact numbers).
4. No phase6 / M1 / Q4-down relitigation; no production default change
   before the bracket books.

## 6. Evidence

- llama shape census: `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
  (`shape_census` rows for `rms_norm_f32`)
- llama source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/norm.cu`,
  `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`
- our bodies: `tinygrad/codegen/late/reduce_output.py`;
  P2 evidence `docs/task_workflow/evidence/nv-reduce-output-site-absorption-p2-ab-20260812.json`
