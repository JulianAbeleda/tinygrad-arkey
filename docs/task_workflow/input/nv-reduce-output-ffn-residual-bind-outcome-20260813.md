# NV reduce-output FFN-norm residual bind outcome (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `6c224e7c2`)
Status: **CPU + GPU measured. GPU wall bracket NOT_PROMOTED; no production
policy change.**
Closes the residual-identity blocker for the FFN-norm (`1_4096`) reduce-output
site and accounts for the full `ffn-before -> promoted` census delta, including
the extra `-34` that a strict site scope would otherwise flag as unexplained.

## What changed

The ffn-norm marker input is the decode residual `h = x + attn_out`. The first
admission (commit `920a94ec1`) proved `ADD(x, attn_out)` as a bounded
residual-sum identity but then re-materialized a fresh ADD buffer at lowering.
That turned the 2->1 norm win into a net-zero swap: the shared `h` was already
consumed by the ffn_down residual slot, so the fresh ADD was a duplicate.

Commit `6c224e7c2` changes lowering from
`_reduce_residual_sum_materialized_view` to `_reduce_residual_sum_view`: it
re-proves the ADD with `_bounded_residual_sum_identity` and returns the exact
marker input, so the fused body reads the same `h` buffer the ordinary chain
consumes. A self-add (`x + x`) fail-closed guard and a hermetic test were
added.

## Census result (DEV=CPU, depth 256, typed-semantic-producer on)

| arm | programs | `reduce_output_rmsnorm_1_4096` | weight stores |
| --- | ---: | ---: | ---: |
| ffn-before (q/k on, ffn off) | 734 | 36 | 0 |
| promoted (q/k on, ffn on) | 666 | 70 | 0 |

Net program delta is **-68**. The FFN-norm site itself is the expected -34:
34 `r_1024_4` reduces and 34 `E_1024_4` epilogues collapse into 34 fused
`reduce_output_rmsnorm_1_4096` bodies (36 -> 70).

## The extra -34 is an explained CPU GEMV fusion, not a q/k change

The remaining -34 comes from one family:

```text
r_64_16_4_16_4_2_4_8      34 -> 0     role=attn_qo, blk.N.attn_output (o-proj), m=1 n=4096 k=4096
r_64_16_4_48_2_2_2_4_8    16 -> 0     role=ffn_down, k=12288
r_64_16_4_48_4_2_4_8      18 -> 0     role=ffn_down, k=12288
r_64_16_4_16_4_2_32_48_*   0 -> 34    role=ffn_down (renamed), k=12288
```

The labels come from graph-admission `semantic_identities`
(`role`/`tensor_name`/`module_path`/`logical_m-n-k`) captured on the same
machine. Binding the shared `h` removes the duplicate-ADD materialization
barrier, so the CPU scheduler fuses the attention-output (o-proj) GEMV reduce
into the ffn_down GEMV reduce: both write a 4096-wide output over the same
grid geometry (`64*16*4 = 4096`), and the fused kernel name grows the o-proj
reduce ranges (`16_4_2_32`). The graph-admission role stays `ffn_down` (the
primary output), which is why the semantic label does not show `attn_qo` on
the fused kernel.

This is a correctness-preserving CPU-scheduler multi-output fusion. The q/k
families are unchanged between the arms:

```text
r_2_32_16_4_16_4_2_4_8  34 -> 34
r_8_2_16_4_16_4_2_4_8   36 -> 36
E_2_128_4               36 -> 36
E_8_128_4               34 -> 34
```

It is CPU-only: on the GPU the o-proj and ffn_down are custom Q4K/Q6K GEMV
kernels and do not fuse this way, so the GPU census will show the ffn-site
-34 without the CPU GEMV shift.

## GPU A/B result (RTX 5090, d512)

The GPU admission was blocked by the rangeify m4 input-view proof: the
absorbed epi-resadd ffn spelling is `MEMORY_SEMANTIC(RESHAPE(AFTER(PARAM,
CALL)))`, and RESHAPE carries its shape descriptor as a second source.  The
proof rejected any two-source RESHAPE leg as an arity mismatch, so the
absorbed blocks never admitted (`input_proof_missing`).  Commit `9ed605f46`
walks the RESHAPE leg by `src[0]` and re-proves the bounded opaque AFTER.

| arm | fused bodies | C6 | q | k | kernels | norms kernels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 0 | 0 | 0 | 0 | 757 | 328 |
| candidate | 127 | 55 | 36 | 36 | 558 | 2 |

Net program delta is **-199**; the norms population drops 328 -> 2 (only the
output norm stays ordinary).  The 55 C6 bodies are 36 ffn + 18 attn + the
final norm.  Full fp32 logits SHA-256 is bitwise identical to control
(`4ff1b9eaf5308a4e0d44d8ab24b9ca23bc2831c430654b0f6f0724cc0ff84e69`).

Reverse wall bracket (control / candidate / control, 5 reps x 32 tokens):

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control A | 5.2469 | 190.57 |
| candidate | 5.2502 | 190.47 |
| control B | 5.2749 | 189.58 |
| control bracket median | 5.2609 | 190.08 |

Candidate is **+10.68 us/token faster than the bracket median but -3.30
us/token slower than control A**, well inside the 50 us/token promotion
threshold and within run-to-run noise.  Verdict **NO-GO / NOT_PROMOTED**: the
route is correctness-clean and removes 199 kernels, but at d512 the fused
bodies carry real per-launch cost that offsets the removed cheap ordinary norm
kernels, so wall-clock is flat.

## Gates remaining

None at this stage. The per-site GPU gate above completed; the route stays
closed by default because the wall bracket did not promote.

## Evidence

- `docs/task_workflow/evidence/nv-reduce-output-ffn-bindfix-census-20260813.json`
  (ffn-before/promoted census, `ffn_diff`, `ffn_contract.explained_gemv_shift`)
- `docs/task_workflow/evidence/nv-reduce-output-ffn-merged-records-20260813.json`
  (graph-admission records with semantic identities for the merged
  `r_64_16_4_16_4_2_32_48_*` kernels)
- `scratchpad/nv_reduce_output_rmsnorm_census.py`
  (`_gemv_shift_explanation`)
