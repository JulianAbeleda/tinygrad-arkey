# NV reduce-output FFN-norm residual bind outcome (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `6c224e7c2`)
Status: **CPU measurement record. No GPU arm, no production policy change.**
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

## Gates remaining

The GPU gate in the per-site absorption scope
(`nv-reduce-output-site-absorption-scope-20260812.md` section 5) is the next
authority: single-layer real-token A/B at d512 under the GPU bench lock, exact
logits/token SHA, census contract, reverse wall bracket, then full A/B. The
census contract now records `explained_gemv_shift` so the o-proj/ffn_down
fusion is explicit rather than a silent unrelated-count change.

## Evidence

- `docs/task_workflow/evidence/nv-reduce-output-ffn-bindfix-census-20260813.json`
  (ffn-before/promoted census, `ffn_diff`, `ffn_contract.explained_gemv_shift`)
- `docs/task_workflow/evidence/nv-reduce-output-ffn-merged-records-20260813.json`
  (graph-admission records with semantic identities for the merged
  `r_64_16_4_16_4_2_32_48_*` kernels)
- `scratchpad/nv_reduce_output_rmsnorm_census.py`
  (`_gemv_shift_explanation`)
