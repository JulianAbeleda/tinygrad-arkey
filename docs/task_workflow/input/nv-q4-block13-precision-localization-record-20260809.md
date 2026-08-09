# NV Q4 block-13 precision localization (CPU-only, post-census-fix HEAD)

Status: **localization record; no lease expansion authorized.**

Date: 2026-08-09
HEAD: `dbfb4ab8296b62efcf2be2ab0fdc1a1dbd0fc994` (branch
`nvidia-bringup-20260731`).

Section 5.3 of `nv-gemv-substrate-landing-scope-20260808.md` authorizes a
CPU-only localization of the cumulative llama-Q8 error across the g12->g18
boundary using per-block signed logit-delta vectors.  It does not authorize
any lease expansion: block 13 and blocks 19-35 remain NO-GO, and any future
candidate must return through a fresh semantic + settled wall gate.

## Method

CPU-only (`DEV=CPU`, no GPU time, no `/tmp/gpu-bench.lock`).  The small-
maxcontext Qwen3-8B-Q4_K_M fixture loads with the unit-test DeviceFacts
pattern (`Transformer.from_gguf` with MAXC=256, fake facts via
`tgm.scan_device_facts`).  Every primitive's admission is forced off
(ordinary fallback path) so CPU can run without the CUDA-only kernels, while
the packed Q4_K/Q6_K words stay installed.  The shared-Q8 lease call is
replaced by a faithful numpy emulation of the CUDA kernels
(`/tmp/q8_lease_emulation.py`: Q8_1 provider, Q4 cooperative and Q6
consumers, fp16 rounding, XOR-tree reductions).  Each config runs as its own
process with the same 192-token prelude, a KV snapshot restore, and one eager
full-logit capture at `start_pos=193` (`--depth 192 --count 1`); a repeated
in-process capture collapses to all-zero buffers on this emulated path, so
one-config-per-process is the stable harness shape.

Configs: `baseline` (no lease), `g12` (blocks 1-12), and `g12+k` for
k in 13..18.  Per-block marginal delta = `logits(g12+k) - logits(g12)`.

## Engagement verification

The emulated call is confirmed to fire for every block with the correct
`(1,1,4096)` shape; blocks 1-13 carry `SharedQ8AttentionAdmission` in the
g12+13 config, and block 13 takes the fused RMSNorm provider path
(`_reduce_output_rmsnorm_marker` present, fp16 norm weight) exactly like
blocks 1-12.  The block-13 delta is therefore a real numerical result, not a
silent no-op.

## Per-block signed logit-delta vectors (vs g12, position 193)

| block | V type | rel L2 | mean | mean abs | max abs | p99 abs | nonzero frac |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | Q4K | `2.723858e-4` | `-9.862e-6` | `6.451e-4` | `5.397e-3` | `2.170e-3` | `0.9999` |
| 14 | Q4K | `1.066739e+0` | `+1.446e+0` | `2.455e+0` | `2.016e+1` | `9.010e+0` | `1.0000` |
| 15 | Q6K | `5.857933e-1` | `+2.052e-1` | `1.378e+0` | `1.160e+1` | `4.672e+0` | `1.0000` |
| 16 | Q4K | `1.264688e+0` | `+3.264e+0` | `3.382e+0` | `1.790e+1` | `7.662e+0` | `1.0000` |
| 17 | Q4K | `1.255917e+0` | `-1.404e+0` | `3.005e+0` | `2.112e+1` | `9.865e+0` | `1.0000` |
| 18 | Q6K | `1.027340e+0` | `+7.888e-1` | `2.405e+0` | `1.806e+1` | `8.517e+0` | `1.0000` |

Token streams (argmax of the captured logit): baseline `4710`, g12 `13876`,
g12+13 `13876`, g12+14 `1782`, g12+15 `3974`, g12+16 `279`, g12+17 `13`,
g12+18 `13`.  All eight captures finite; logit SHA-256s in
`docs/task_workflow/output/nv-q4-block13-precision-localization-20260809.json`.
The g12+13 arm re-ran bit-identical (`1066949bba6e`), confirming determinism.

## Findings

1. Block 13 is numerically inert at this position: its marginal delta is
   `2.72e-4` relative L2 with `6.45e-4` mean absolute logit movement and an
   unchanged token stream.  It is the only block in 13..18 below `1e-3`.
2. Blocks 14-18 each carry `0.59-1.26` relative L2 and change the token;
   the cumulative perturbation across the g12->g18 boundary is carried by
   that set, not by block 13.
3. Absolute rel L2 here is a CPU-emulation instrument (emulated CUDA kernels
   vs ordinary CPU fallback), not a GPU authority number; the localization
   signal is the block-relative pattern.  It is consistent with the 08-05
   record's non-monotonic precision budget and does not revise any GPU row.

## Verdict

Localization only.  Block 13 remains NO-GO for lease expansion, blocks 19-35
remain NO-GO, and no admission is changed.  The block-13 boundary in the
ledger stands; any future candidate (including block 13 after an arithmetic
correction) must return through a fresh semantic + settled wall gate per
section 5.3.  No code changed; no promotion record touched.

