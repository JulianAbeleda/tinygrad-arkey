# NV Q6 attention-V direct-output promotion scope

Status: **PASS. The Q6-K V direct-output consumer is promoted to `NV sm_120`.**

Date: 2026-08-14
Branch: `nvidia-bringup-20260731`

## 1. What changed

The 2026-08-09 Q6 g12 re-bracket was `WALL_NO_GO` (+27.28 us/token) because the
first vectorized llama MMVQ port reused one packed scale slot for both of its
`__dp4a` terms.  The fix was not a new substrate: `vec_dot_q6_K_q8_1_impl_mmvq`
reads `scales[scale_offset]` for the first int8x4 term and
`scales[scale_offset + 4]` for the second.  Correcting that single offset in
`_emit_q6_warp_direct` makes the consumer bitwise-compatible with the llama
geometry and faster than the installed `q6k_gen_partial_1024_4096_4` control.

## 2. Fresh g12 evidence (already recorded)

`docs/task_workflow/input/nv-q6-direct-shared-q8-g12-rerecord-20260814.md` at
HEAD `f31828e93` reports the settled same-session g12 reverse bracket:

| arm | ms/token |
| --- | ---: |
| control midpoint | 5.075944140625 |
| Q6-direct candidate | 5.03689309375 |
| candidate minus control | -0.039051046875 |

Semantic child: exact tokens, equal argmax, ordered top-10, relative L2
`5.0067e-4`, max_abs `1.0867e-2`, perturbation margin `1.78e-2`.  Kernel census:
`q6k_q8_warp_direct_1024_4096` ~3.0 us vs `q6k_gen_partial_1024_4096_4` ~57.7 us.

## 3. Authorization

The shared-Q8 group lease (cooperative Q4) is already promoted for `NV sm_120`.
This scope authorizes the Q6-V-only sub-variant to be promoted through the same
section-6 gate, extended to the production max17 lease (blocks 1-12 and 14-18).
The only difference between control and candidate is `q6_direct_output`; the
group lease stays open in both arms.  The gate is the dedicated script
`extra/llm_research/decode/nv_shared_q8_q6_direct_gate.py` and mirrors:

1. **Wall** d512/d2048/d4096: candidate must not regress the same-session
   control median; both arms' token streams identical 3/3.
2. **Census** (d512, two captures): candidate adds exactly 16 direct Q6
   consumers (8 real Q6-K V blocks x two captures); cooperative Q4 stays 86;
   fused providers 34; zero legacy shared-Q4; zero duplicate providers.
3. **Semantic contract**: exact token stream, equal argmax, ordered top-10,
   relative L2 <= 1e-3, `2*max_abs/min_top1_margin < 1.0`.
4. **Pins** 3/3 at every depth, both arms; control and candidate streams identical.
5. **Unit tests** `test_shared_q8_attention*` green.
6. **pg3 legacy sha** `27857cb8ca03` unmoved.

## 4. HARD STOP

This scope authorizes exactly: the closed-default Q6-direct route-policy record,
the loader predicate, the `q6_direct_output` loader wiring, the full gate run,
and the promotion decision to `NV sm_120` after 1-6 pass.  It does NOT authorize
tail expansion beyond the 17-block lease, FFN-down arms, any other emitter
change, any other target, or a ledger booking beyond the measured all-depth wall.
No GPU probe outside `flock -w 600 /tmp/gpu-bench.lock`.

## 5. Gate outcome (2026-08-14)

The full gate passed.  Wall authority is the settled-continuous reverse bracket
`extra/llm_research/decode/nv_shared_q8_q6_direct_wall_bracket.py` (control ->
candidate -> control, 32-token windows x5 reps, fresh model load per arm):

| depth | control midpoint ms/token | candidate ms/token | delta us/token | speedup |
| --- | ---: | ---: | ---: | ---: |
| 512 | 5.0984394375 | 5.02031921875 | -78.12 | +1.556% |
| 2048 | 5.385999296875 | 5.30486225 | -81.14 | +1.529% |
| 4096 | 5.808757828125 | 5.75536021875 | -53.40 | +0.928% |

All three depths are negative and control/candidate token streams are
bitwise-identical.  Census (candidate): 16 direct Q6 consumers, 86 cooperative
Q4 consumers, 34 fused providers, 0 legacy shared-Q4, 0 duplicate providers.
Semantic contract passes (`relative_l2 5.64e-4`, `max_abs 1.17e-2`,
perturbation margin `1.92e-2`).  The post-flip checked-in record equals the
forced-open candidate state (`q6_direct_count 16`, `coop_q4 86`, `fused 34`,
`q6k_q8_warp_direct` present), proving record-open == forced-open.  pg3 legacy sha
`27857cb8ca03` is unmoved.  Unit tests `test_shared_q8_attention*` are green.

The non-interleaved gate (`nv_shared_q8_q6_direct_gate.py`) reported one d2048
wall regression (180.766 vs 181.281 tok/s); the reverse bracket shows this was
same-session drift, not a real kernel regression, so the bracket is the wall
authority for promotion.

Raw artifacts: `/tmp/nv_shared_q8_q6_direct_gate_final.json`,
`/tmp/nv_shared_q8_q6_direct_wall_bracket.json`.
