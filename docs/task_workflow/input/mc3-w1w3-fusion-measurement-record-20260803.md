# MC3 w1+w3 fusion measurement record - gate/up fused-shape viability (compile + standalone timing)

Date: 2026-08-03
Status: measurement record. Authorized by
`decode-gemv-instruction-bandwidth-scope-20260803.md` section 4.3 (MC3 probe) and the
section 10 amendment (per-item HARD STOP lifted; stop only on a blocker with no named
next step, or on the MC3 coordination handoff). Diagnostic probe and measurement only:
no route admission, no emitter change in the runtime. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `e71f2ef17` (docs-only ahead of the scope boundary
`56ecbd62d`; tracked tree has only user-owned doc modifications and untracked artifacts,
all untouched).

## 1. Why this record exists

MC3 asks whether llama's w1+w3 fusion is expressible and worth landing on our side. llama
fuses gate and up into ONE 12288-row kernel with silu folded in (campaign scope section
14.2: 37.9 us, no separate silu kernel); we run gate and up as two 12288x4096
`q4k_g3_lanemap_gemv_12288_4096` kernels (20.69 us each in-loop at the like-for-like
census, 41.4 us combined, 72 kernels/token). The kernel-count difference (252 vs 216 =
exactly 36) is the dominant class-level term per the INFERRED arithmetic of scope section
2.3. This probe sizes the fused kernel itself: render the fused shape through the existing
q4k emitter machinery and CUDARenderer, verify it compiles with 0 spills, inspect SASS,
and time it standalone against the pair. No route changes; the additive design is recorded
for parent coordination (MC1/MC2 own the same files and run in parallel).

## 2. Protocol

- Probe: `extra/llm_research/microbench/w1w3_fused_render_probe.py` (new file, [test]
  commit). Renders three sources through the real emitter machinery + CUDARenderer:
  `q4k_g3_lanemap_gemv_12288_4096` (installed kernel, both pair arms) and the probe-only
  fused `q4k_g3_lanemap_gemv_w1w3fused_12288_4096`. The fused kernel is composed from the
  SAME lowering pieces the installed emitter uses (`_q4k_block_dot_packed_load`,
  `LanePartition`, `_silu_uop`, `_warp_reduce_sum_staged`) with the two-accumulator
  loop-carried pattern proven in production by the flash decode kernel
  (`flash_decode_attention.py` acc/den/mx registers). Compiles the emitted sources
  verbatim with nvcc 13.2 `-arch=sm_120 -O3 -std=c++17 --ptxas-options=-v`; the 0-spill
  gate is enforced before any timing. SASS via cuobjdump + nvdisasm (triton package);
  instruction census per kernel. Timing at the emitted geometry grid (12288,1) x 32
  threads, one launch per full projection/pair/fused pass, host-looped best-of-N passes
  under one CUDA event pair per arm, rounds interleaved. Numerics: fused[r] vs
  silu(gate[r]) * up[r] from the pair outputs, finite-weight fill.
- Session: same RTX 5090 / sm_120 box as the wall authority, driver 595.84, CUDA 13.2.
  All GPU runs serialized with `flock /tmp/nv_gpu.lock` and confirmed 0% util at lock
  acquisition. Fused prefill attention disabled
  (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`) in every decode run.
- Config: Qwen3-8B-Q4_K_M. Fused shape 12288 rows x 4096 k: per the GGUF, `ffn_gate` and
  `ffn_up` are each [4096, 12288] (n_ff = 12288 per projection), so the fused node computes
  z = silu(w1 h) * (w3 h) over 2 x 28.3 MB of Q4_K words.
- Evidence classes: OBSERVED = measured in this session under the lock (or reused
  artifacts re-derived this session); INFERRED = arithmetic, flagged inline. Lifecycle
  vocabulary only.

## 3. The fused shape and what the installed emitter can and cannot express

llama's fused node, re-derived this session from the reused CUDA-era artifact
`/tmp/llama_tg10_node.sqlite` (the like-for-like record's node basis): `mul_mat_vec_q`
launches grid (12288,1) with block 32, n = 614, avg 37.98 us, median 37.92 us - one kernel
computing both projections with silu folded in, no separate silu node. The probe's fused
kernel has exactly that shape: 12288 rows, one output per row
`out[r] = silu(dot(gate_row_r, x)) * dot(up_row_r, x)`, two weight buffers (gate words, up
words) and the shared fp16 x prelude.

The installed `q4k_g3_lanemap_gemv_kernel` cannot emit this by itself: it is a
single-accumulator, single-weights-buffer per-row emitter, and its epilogue kinds
(`residual_add`, `ffn_down_fused`, `fp16_cast`) all operate on pre-computed vectors rather
than a second weight dot. The fused shape needs two loop-carried fp32 accumulators in one
reduce loop. The UOp CFG rewrite (`CFGContext`, `linearizer.py:182`) forbids two ENDs on
one RANGE and two-store AFTER chains in a loop body, so two sequential loops or a
body-chain do not lower. The production-proven shape is the flash decode kernel's
multi-register loop (`flash_decode_attention.py:236-240`): one init chain, the first
update chained via `.after()` WITHOUT `.end`, the last update ends the range. The probe
reproduces that pattern with distinct REG slots (20/21) and distinct shfl staging bases
(90-94, 95-99); the second lane reduce must not reuse the default slot base 90.

**Expressibility verdict: the fused shape is expressible through the existing lowering
machinery** (all pieces already exist; the composition pattern is already in production),
and the minimal additive emitter change is a new kernel builder - e.g.
`q4k_g3_lanemap_gemv_w1w3_kernel(rows, k)` with signature `(out, gate_words, up_words, x)`
using the two-accumulator loop - not a modification of the installed kernel. That change
is NOT made here (probe only).

## 4. Purity - ptxas and SASS table

`--ptxas-options=-v` (both runs, identical):

| kernel | registers | spill stores | spill loads |
| --- | ---: | ---: | ---: |
| q4k_g3_lanemap_gemv_12288_4096 | 61 | 0 | 0 |
| q4k_g3_lanemap_gemv_w1w3fused_12288_4096 | 74 | 0 | 0 |

SASS instruction census (cuobjdump --dump-sass, nvdisasm from the triton package; no
LDL/STL anywhere in either kernel):

| op | pair | fused |
| --- | ---: | ---: |
| LDG.E (weights u32) | 8 | 16 |
| LDG.E.U16 (x halves) | 32 | 32 |
| STG.E | 1 | 1 |
| FFMA | 64 | 131 |
| FMUL | 16 | 35 |
| FADD | 5 | 11 |
| HADD2.F32 / HFMA2 | 34 / 1 | 36 / 3 |
| LOP3.LUT / SHF.R.U32.HI / I2FP.F32.U32 | 55 / 48 / 48 | 116 / 99 / 96 |
| SHFL.BFLY (lane reduce) | 5 | 10 |
| MUFU.EX2 / MUFU.RCP (silu) | 0 / 0 | 1 / 3 |
| total | 368 | 674 |

Hot-loop purity: in both kernels the reduce loop contains only weight/x LDG plus bit math
(LOP3/SHF/I2FP) plus the FMA family; SHFL (post-loop lane reduce), MUFU (silu epilogue)
and the single STG sit outside the loop. 0 spills at both the ptxas and the SASS level.
The fused kernel shares the x loads with the pair (same 32 U16 loads per iteration,
hash-consed indexing) and adds the up projection's 8 weight words to the gate's 8.

## 5. Results - standalone timing and numerics (OBSERVED, same session, same harness/data)

All arms timed at the emitted geometry grid (12288,1) x 32 threads, host-looped passes
under one event pair per arm, 0% util at lock acquisition:

| arm | run A (32 passes x 3 rounds), us | run B (128 passes x 5 rounds), us |
| --- | ---: | ---: |
| gate only | 17.18-17.21 | 17.12-17.14 |
| up only | 17.15-17.17 | 17.08-17.12 |
| pair (gate+up back-to-back) | 34.19-34.27 | 34.11-34.24 |
| fused | 23.17-23.18 | 23.27-23.36 |

Numerics (finite Q4_K fill): max abs diff vs silu(gate)*up 0.000002, max rel 1.97e-07,
nonfinite rows 0, bit mismatches 20/12288 (host exp2f vs device MUFU.EX2 rounding),
max abs value 4.98e8. The fused kernel reproduces the composed pair output to the
approximation accuracy of the in-kernel exp2.

Sizing against the floor:

| path | us |
| --- | ---: |
| fused standalone (this probe) | 23.3 (min 23.17-23.27 across runs) |
| pair standalone (this probe) | 34.2 (34.11-34.27) |
| pair in-loop census at HEAD (2 x 20.77, n=72; INFERRED addition; scope cites 2 x 20.69 = 41.4) | 41.5 |
| llama fused node (reused artifact, med 37.92) | 37.9 |

The fused kernel is 1.47x faster than the pair in the same harness and sits ~14.6 us
below llama's fused node time standalone (INFERRED subtraction). L2-residency note: the
two projections (56.6 MB) fit the 128 MB L2, so the standalone loop is L2-resident steady
state for BOTH arms - the fused-vs-pair comparison is apples-to-apples, and neither number
is a wall prediction. The isolated same-session d512 wall (scope 5.3) is the
landing-scope ranking step and is not run here (no route change exists yet).

## 6. Viability verdict and the exact additive design (coordination handoff)

**VIABLE.** The fused shape compiles clean through the existing machinery and
CUDARenderer (0 spills, 74 regs), validates numerically, and times below the pair in the
same harness (23.3 vs 34.2 us) and below llama's fused node time standalone (23.3 vs 37.9
us). Per the MC3 coordination handoff, the route admission is NOT implemented: MC1/MC2
run in parallel and own the same files (`decode_routes.py` / `decode_kernels.py`);
concurrent edits are not allowed. This record carries the exact additive design:

1. Emitter: new builder in `decode_kernels.py`, e.g. `q4k_g3_lanemap_gemv_w1w3_kernel(rows,
   k)`, signature `(out, gate_words, up_words, x)`, two-accumulator loop (section 3),
   silu folded in-kernel via `_silu_uop` (bit-identical to the Tensor.silu lowering, so
   the decode sha cannot move from rounding deltas), one f32 per row. New kernel name
   (`q4k_g3_lanemap_gemv_w1w3fused_12288_4096` follows the house suffix convention);
   legacy kernel hashes untouched.
2. Route admission: new candidate in `decode_routes.py` bound to the ffn_gate_up role pair
   (`self.ffn_gate` + `self.ffn_up`), admitted when both linears are Q4K decode-enabled on
   the target and the new gate is open; one program produces z = silu(gate)*up (12288,).
   The ffn_down route then consumes z through the existing `residual_add` epilogue
   (`out[row] = total + normed_h[row]`) instead of `ffn_down_fused` - the gate_out/up_out
   inputs and the in-kernel silu*mul prelude are gone from the down kernel. Legacy gate/up
   rows untouched; 72 -> 36 kernels/token (the 36 launch-count removal).
3. pg3 arms: the new builder is renderer-neutral UOps (it renders through HIPRenderer and
   MetalRenderer as well); any AMD/Metal admission must move both arms identically with
   its own pg3 hash, and the 10 legacy rows + `add50a7aa43f` stay byte-identical (control
   below). Initial NV admission is CUDA-only.
4. Landing gates: pg3 re-derive, NV pins 3/3, isolated same-session d512 wall vs llama,
   like-for-like cap. Node-sum upper bound 36 x (41.4 - 37.9) us ~ 0.13 ms (INFERRED
   arithmetic, cited from scope section 3; under the 0.597 ms cap; no composed forecast,
   no wall claim).

## 7. Controls

- pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU, no lock): all 10
  legacy hashes byte-identical to the pinned table (312422c73a49 / 27857cb8ca03 /
  851760e2053c / 39ddb717ddd4 / cc38fbb3db92 / 5795e66a7292 / 344e1c388eeb /
  c708302aa2d2 / 66d4c4da3108 / c78e4651ad35), and the M2 promoted fused row
  `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f` holds.
- Fixed-depth decode pin (nmeas=20, reps=3, d512, fused prefill disabled): token sha256
  `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3, first token
  `151936` 3/3, 1020 kernels/token, 175.18 tok/s median.
- Decode sha256 + census (`model_e2e_bench.py`, d512 prefill, 96 decode tokens):
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, first token ids
  start `50994, 82, 31109, 3508, ...`, census row
  `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f` holds.
- Fresh GEMV-class census at HEAD (`gemv_class_census_nv.py`, d512): 
  `q4k_g3_lanemap_gemv_12288_4096` median 20.77 us (n=72, min 20.10 max 29.47), GEMV-class
  252 kernels / 3.849 ms, token sha 3/3, first token 3/3, 174.19 tok/s median.

## 8. Deviations

- `nvdisasm` is not installed with the CUDA toolkit; cuobjdump needs it on PATH, and it
  ships in the triton package (same note as the mma_peak README and the L4 record). The
  probe adds the triton bin dir to PATH for the cuobjdump subprocess.
- The probe's first 128x5 run had a harness stack-layout bug (round arrays sized 4 for 5
  rounds) that corrupted the pair min (17.14, less than the gate min - impossible). The
  bug was found by that impossibility, fixed (arrays sized 8), and the run redone; the
  buggy row is excluded from this record and the fixed file is the committed one.
- The initial random-raw-word fill dequantized to non-finite values, which made the
  equality check vacuous (NaN comparisons cannot fail). The harness now constructs valid
  finite Q4_K blocks (fp16 d/dmin, bounded scale bytes, random nibbles), giving
  nonfinite=0 and a meaningful max_rel. Timing is unchanged by the fill change.
- `to_program` compiles the rendered source once with its own nvcc invocation during
  render (CPU-side, cached); the committed probe compiles the harness verbatim with the
  mandated flags and times only that binary.
- Standalone vs in-loop deltas are expected: the same gate/up kernel measures 17.1 us
  standalone vs 20.77 us in the DEBUG=2 trace (the L2 record's own standalone/in-loop
  deltas are larger, e.g. partial 12.92 vs 17.15). Both pair and fused standalone arms run
  in the same harness, so the comparison is unbiased.
- Worktree at record time: tracked tree clean except user-owned doc modifications
  (`docs/README.md`, `docs/beating-llama-first-principles-20260731.md`,
  `docs/what-makes-inference-fast.md`); untracked user-owned artifacts
  (`dp4a_peak_cuda*`, `flash_score_tile_peak_cuda`, `l2_q6k_partial_sweep`,
  `q6k_vocab_coop_ceiling_cuda`, `scratchpad/t6_metal_admission_probe.py`) untouched.

## 9. Closeout

The MC3 diagnostic is complete: the fused shape is expressible, compiles clean, and times
below the pair and below llama's fused node standalone. Verdict: **viable - implementation
pending parent coordination**. No implementation code, no promotion, no push; parent
review happens on the commits.

## 10. References

- `decode-gemv-instruction-bandwidth-scope-20260803.md` sections 3, 4.3, 5, 10
- `nv-performance-campaign-scope-20260801.md` section 14.2 (llama fused node shape/time)
- `like-for-like-cap-settling-record-20260803.md` (class census, 0.597 ms delta, cap)
- `l4-vocab-substrate-fusion-measurement-record-20260803.md` (record style, probe pattern)
- Reused artifact: `/tmp/llama_tg10_node.sqlite` (CUDA-era nsys node trace)
- Probe: `extra/llm_research/microbench/w1w3_fused_render_probe.py`
