# QK Ansor-direction baseline artifacts

Date: 2026-06-12

Purpose: freeze the known-good Q4_K/Q6_K v1 primitive runtime before adding the
semantic descriptor and generated candidate harness. These logs are baselines for
the search/generation experiment, not a new production policy.

## Environment

- Repo: `tinygrad-arkey`
- Baseline commit: `e42fbce5c`
- Device: `AMD::gfx1100`
- Flags: `DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 PYTHONPATH=.`
- Models:
  - `~/models/Qwen3-8B-Q4_K_M.gguf`
  - `~/models/Qwen3-14B-Q4_K_M.gguf`

## Commands

```bash
DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128

DEV=AMD Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-14B-Q4_K_M.gguf --warmup --benchmark 128
```

## Results

| model | log | tokens | avg tok/s | avg reported GB/s |
|---|---|---:|---:|---:|
| Qwen3-8B-Q4_K_M | `8b-q4q6-baseline-benchmark128.log` | 128 | 58.08 | 279.73 |
| Qwen3-14B-Q4_K_M | `14b-q4q6-baseline-benchmark128.log` | 128 | 28.26 | 247.21 |

The current runtime wrappers in `tinygrad/llm/model.py` are unchanged by this
artifact. Generated-policy integration must reproduce this target before it can
replace the explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` path.

## Generated Candidate Artifacts

Files:

- `8b-descriptors.json`: representative descriptor snapshot.
- `8b-level0-search-full.json`: generated level-0 candidate search for Q4/Q6
  shape coverage.
- `8b-level0-policy-full.json`: generated shape/format policy cache.
- `8b-generated-policy-smoke.log`: runtime install smoke test using
  `QK_GENERATED_POLICY`.
- `8b-generated-policy-benchmark128*.log`: full generated-policy decode runs.
- `8b-q4q6-baseline-benchmark128-rerun.log`: same-session explicit-flag
  baseline comparison.
- `8b-level2-q8-sketch.json`: level-2 run showing the generated q8_1 sketch is
  present but rejected as `not-implemented`.
- `policy-parity-8b.{json,md}`: explicit primitive policy vs generated-policy
  behavior comparison for every real Q4_K/Q6_K weight tensor.
- `8b-level2-q8-real.json`: level-2 run with the first runnable Q4_K x q8_1
  activation candidate.
- `8b-level2-q8-intdot.json`: level-2 run with the Q4_K x q8_1 integer-dot
  candidate.
- `8b-level2-q8-vdot-parallel.json`: level-2 run with the scheduled
  `v_dot4_u32_u8` q8_1 candidate.

Level-0 generated search, Qwen3-8B:

| tensor | format | shape | fused GB/s | generated winner | winner GB/s |
|---|---|---:|---:|---|---:|
| `blk.0.ffn_gate.weight` | Q4_K | 12288x4096 | 81.20 | `v1_q4_packed` | 417.94 |
| `blk.4.ffn_down.weight` | Q4_K | 4096x12288 | 15.67 | `v1_q4_packed` | 265.90 |
| `blk.0.attn_q.weight` | Q4_K | 4096x4096 | 15.44 | `v1_q4_packed` | 183.60 |
| `blk.0.attn_k.weight` | Q4_K | 1024x4096 | 100.22 | `fused_graph` | 100.22 |
| `blk.0.ffn_down.weight` | Q6_K | 4096x12288 | 21.18 | `v1_q6_packed` | 128.83 |

Generated-policy runtime:

| mode | log | tokens | avg tok/s | avg reported GB/s |
|---|---|---:|---:|---:|
| explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` rerun | `8b-q4q6-baseline-benchmark128-rerun.log` | 128 | 58.00 | 279.39 |
| `QK_GENERATED_POLICY=8b-level0-policy-full.json` | `8b-generated-policy-benchmark128.log` | 128 | 54.77 | 263.73 |
| `QK_GENERATED_POLICY=8b-level0-policy-full.json` rerun | `8b-generated-policy-benchmark128-rerun.log` | 128 | 56.07 | 270.10 |

The generated policy installed the same class of wrappers as the explicit path:
162 Q4_K linears and 18 Q6_K linears, with the small Q4_K KV shape explicitly
falling back through `policy_fused`. The remaining 3-5% runtime gap is unresolved
run variance or a subtle path difference, so `QK_GENERATED_POLICY` remains
opt-in and does not replace the explicit primitive flags.

Policy parity check:

| check | value |
|---|---:|
| total Q4_K/Q6_K weight tensors | 254 |
| effective mismatches | 0 |
| explicit installed wrappers | 180 |
| generated installed wrappers | 180 |
| generated unsupported winners | 0 |

The parity report rules out a generated-policy coverage bug as the cause of the
56.07 vs 58.00 tok/s rerun difference. The raw differences are fallback-reason
differences only: explicit policy uses `policy_fallback`, while the generated
cache records either `policy_fused` for measured small Q4_K shapes or
`policy_missing` for unsearched fallback shapes.

## Runnable q8_1 Level-2 Candidate

The first real structural q8_1 candidate was generated and timed for Q4_K. It
packs the activation into 32-wide int8 blocks inside the candidate path, then
runs a Q4_K x q8_1 custom kernel. Correctness compares against the centralized
Q4_K reference and dequantized q8_1 activation reference.

| tensor | shape | fused GB/s | v1 packed GB/s | q8_1 packed GB/s | winner |
|---|---:|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 12288x4096 | 81.17 | 416.59 | 170.92 | `v1_q4_packed` |
| `blk.4.ffn_down.weight` | 4096x12288 | 15.62 | 269.20 | 150.17 | `v1_q4_packed` |
| `blk.0.attn_k.weight` | 1024x4096 | 111.71 | 51.55 | 36.44 | `fused_graph` |

All q8_1 runs passed the GEMV correctness gate (`max_abs <= 0.001233` on the
listed shapes), but the first q8_1 lowering did not win any tested shape.

The next candidate changed the inner math from per-element float dequant to a
grouped integer-dot identity:

```text
sum((d*sc*q4 - dmin*mn) * (xscale*q8))
  = xscale * (d*sc*sum(q4*q8) - dmin*mn*sum(q8))
```

That was a real improvement over the first q8_1 lowering, but still not enough
to beat v1:

| tensor | shape | v1 packed GB/s | q8_1 float GB/s | q8_1 intdot GB/s | winner |
|---|---:|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 12288x4096 | 420.80 | 173.75 | 216.20 | `v1_q4_packed` |
| `blk.4.ffn_down.weight` | 4096x12288 | 262.82 | 148.74 | 262.50 | `v1_q4_packed` |
| `blk.0.attn_k.weight` | 1024x4096 | 53.07 | 35.16 | 37.40 | `fused_graph` |

Verdict: int-dot validates the diagnosis that q8_1 needs a better inner dot, but
the current UOp/register-reduction lowering still loses. The ffn_down result is
a near-tie, not an acceptance margin, and gate remains far behind v1. It is
rejected by the generated policy and is not wired into `model.py`.

## DEBUG=4 q8_1 Int-Dot Codegen Inspection

Artifacts:

- `q8-intdot-ffn-gate-debug4.log`
- `q4-v1-ffn-gate-debug4.log`

The q8_1 int-dot candidate does not emit a packed dot instruction. The generated
kernel is `q4k_q8_1_intdot_partial_12288_4096_1` and uses scalar nested loops for
the hot dot:

- line 983: kernel entry;
- lines 1008-1013: `uint32` Q4 word load, scalar nibble extraction, scalar
  `signed char` Q8 load, scalar integer multiply/add;
- lines 1016-1019: separate scalar `sum(q8)` loop for the min correction;
- line 1145: timed kernel line, about `150.64us` in this DEBUG=4 run;
- line 1190: clean candidate summary, about `209.63 Q4-GB/s` by device time.

Search for dot-related names found no `v_dot`/`dot4` style intrinsic in the
hot q8_1 kernel. The only AMD builtins in these logs are barriers/fences, not
packed integer dot operations.

The v1 Q4_K packed candidate is `q4k_gemv_partial_12288_4096_1`. It also does
not emit a packed integer dot instruction, but its fp16-activation path is much
simpler: packed Q4 words are loaded once and multiplied directly against fp16
activation values. In the same DEBUG=4 artifact, the clean timed lines for this
kernel are about `67us`, roughly half the q8_1 int-dot kernel time for the same
FFN gate shape.

Conclusion: the q8_1 experiments have isolated the wall to packed-dot lowering,
not q8_1 representation or algebra. Another q8_1 candidate built from the same
generic scalar loops is not justified. The next q8_1 experiment must first prove
that tinygrad's AMD path can emit a real RDNA3 packed-dot operation and pack the
Q4/Q8 lanes in the form that operation expects.

## AMD vdot Smoke and Generated Candidate

Files:

- `extra/amd_vdot_smoke.py`: standalone compiler/on-device smoke.
- `8b-level2-q8-vdot-gate.json`: generated level-2 run including the first
  `q8_1_q4_vdot` candidate.
- `q8-vdot-gate-debug4.log`: generated-code dump for the vdot candidate.

The instruction smoke passed on `gfx1100`:

```text
instruction_smoke: PASS arch=gfx1100 line=v_dot4_u32_u8 ...
group_correctness: PASS seed=1337 dot=2844 q8_sum=-58 q4_sum=219
```

Important signedness result: the usable RDNA3 path here is unsigned byte dot.
The q8_1 signed activation is handled by biasing q8 by `+128` and correcting:

```text
sum(q4 * q8) = udot(q4, q8 + 128) - 128 * sum(q4)
sum(q8)      = sum(q8 + 128) - 128 * 32
```

That proves the instruction path and the q8_1 algebra are compatible.

The generated-code dump confirms the full candidate path contains inline
`v_dot4_u32_u8` calls in `q4k_q8_1_vdot_partial_64_4096_1`.

The first full generated candidate, `q8_1_q4_vdot`, was then added to
`extra/qk_ansor.py` level 2 and run once on the 8B FFN gate shape. This was a
gate run, not a policy benchmark. It passed the same correctness gate as the
scalar int-dot candidate (`max_abs=0.00122976`) but lost badly on performance:

| tensor | candidate | correctness | Q4-GB/s | verdict |
|---|---|---:|---:|---|
| `blk.0.ffn_gate.weight` | `q8_1_q4_vdot` | `0.00122976` | `21.37` | reject |

The reason is structural, not instruction correctness. This first vdot candidate
uses an inline custom C statement with one work item per output row and loops
through K inside that work item. It emits packed dot but throws away the
parallel row/local schedule that made v1 fast. So it is a valid generated
candidate and a useful rejection, not a speed candidate.

Next implication: packed-dot has to be exposed inside a parallel schedule or
renderer lowering. A serial custom-C loop around `v_dot4` is the wrong shape.

## Parallel vdot Candidate

The follow-up candidate exposes `v_dot4_u32_u8` inside the existing parallel
UOp schedule instead of wrapping the whole K loop in a serial custom-C statement.
`extra/qk_ansor.py` level 2 now emits:

- `q8_1_q4_vdot_parallel_p1`: parts=1, `LOCAL:0:64`;
- `q8_1_q4_vdot_parallel_p2`: parts=2, `LOCAL:0:32`;
- `q8_1_q4_vdot_parallel_p4`: parts=4, `LOCAL:0:32`.

`q8-vdot-parallel-gate-debug4.log` confirms the scheduled kernel shape:

- kernel: `q4k_q8_1_vdot_parallel_partial_64_4096_1`;
- workgroup: `amdgpu_flat_work_group_size(1, 64)`;
- hot loop: inline `v_dot4_u32_u8` calls inside the generated scheduled kernel;
- correctness: `max_abs=0.00058949` on the 64-row gate.

Generated level-2 search on the two large Q4_K FFN shapes rejected every
parallel-vdot variant:

| tensor | existing winner | winner GB/s | best parallel vdot | vdot GB/s | vdot correctness |
|---|---|---:|---|---:|---:|
| `blk.0.ffn_gate.weight` | `q4_local32_p2` | 391.88 | `q8_1_q4_vdot_parallel_p1` | 335.01 | 0.00122976 |
| `blk.4.ffn_down.weight` | `v1_q4_packed` | 408.47 | `q8_1_q4_vdot_parallel_p4` | 242.44 | 0.00114174 |

Verdict: packed-dot instruction emission now works inside a parallel scheduled
kernel, so the earlier serial-integration failure is resolved. It still does
not beat the existing v1/intdot candidates through `Ops.CUSTOMI`. The likely
causes are the inline-asm statement-expression overhead, extra q8 bias-pack
kernel, and lost compiler visibility around the fused dot/correction arithmetic.
No more `extra/` q8 arithmetic variants are justified from this point. If q8_1
continues, the next step is renderer/core lowering for the semantic packed-dot
pattern, followed by the same generated-search gate.
