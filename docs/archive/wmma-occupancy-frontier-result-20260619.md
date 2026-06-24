# RESULT — occupancy frontier CLOSED; GROUP+TC guard is justified (answers tinygrad's TODO)

Executed the occupancy-frontier scope. Decisive negative, with the precise reason.

## Config sweep (codegen, 12288x4096x512 fp16, this session's clock)
| config | TFLOPS | note |
|---|---:|---|
| PROD baseline (TC+UPCAST0:2,1:4+UNROLL8) | 38.6 | the ceiling |
| +LOCAL1:2 / +LOCAL1:4 (more waves) | 32.7 / 33.2 | REGRESS (POWN reproduced) |
| hi-intensity UPCAST0:4,1:4 (more acc) | 24.5 | REGRESS (VGPR spill) |
| TC+GROUP / GROUPTOP (LDS reduce-stage) | INVALID | blocked: "no grouping with tensor cores" |

## The blocker, found and TESTED
`tinygrad/codegen/opt/postrange.py:173`: `check(... no grouping with tensor cores) # TODO: why is this wrong?`
-- a guard the tinygrad devs themselves don't understand. Gated it behind ALLOW_GROUP_TC and tested:
- GROUP+TC now COMPILES + RUNS (local_size shows cooperative threads / LDS), BUT:
  - **PERF: no gain** -- TC+GROUP0:2/0:4 = 34.7, TC+GROUPTOP0:4 = 38.6 (= baseline), GROUP+LOCAL = 33.8. Best ties baseline.
  - **CORRECTNESS: WRONG** -- rel_err **1.3938** vs baseline 0.0002 (clean 512x4096x512 check). GROUP+TC produces
    garbage. **This ANSWERS the dev TODO: GROUP (LDS reduce-staging) + tensor-core fragment accumulation do NOT
    compose correctly in the current lowering.** The guard is justified; reverted (kept unconditional).

## Why this closes the frontier
1. Every EXPRESSIBLE occupancy lever fails: more waves (LOCAL) regress (intensity loss, POWN reproduced), more acc
   (UPCAST) regress (spill), LDS-reduce (GROUP+TC) is BROKEN (wrong) AND gives no speedup even when forced.
2. CRUCIAL distinction: tinygrad's GROUP = REDUCE-staging (split K across threads, LDS-reduce partials). Tensile's
   66 uses OPERAND-staging (A/B tiles in LDS for reuse across the output tile) -- a DIFFERENT mechanism tinygrad has
   NO opt for. So even fixing GROUP+TC wouldn't give Tensile's path; and GROUP+TC gave no speedup anyway.
3. So "LDS refuted on RDNA3" is UPHELD, now with the precise reason: the only LDS+TC mechanism tinygrad can express
   (GROUP) is a broken lowering with no perf upside; the mechanism that WOULD help (operand-staging) is not an opt.

## Verdict (final on the WMMA ceiling)
**~38-42 TFLOPS is the confirmed pure-tinygrad RDNA3 WMMA ceiling.** Reaching 66 needs operand-LDS-staging, which
requires: (a) a NEW operand-staging opt/lowering (project-level, doesn't exist), AND (b) the GROUP+TC correctness
bug fixed first if reusing that path -- and the perf evidence (GROUP+TC no-gain) suggests even then it's uncertain.
Both WMMA levers (A addressing-refuted, B pipelining-moot) and now the occupancy/LDS frontier are exhausted.
Prefill rests at PREFILL_V2 ~47% llama + concrete-KV 1.24x (shippable). Path past 42 = external tuned kernel
(doesn't transfer in-model) or a multi-week operand-staging codegen capability.

## Upstream note
The `postrange.py:173` TODO can be answered in tinygrad: GROUP/GROUPTOP + TC produces incorrect results
(rel_err ~1.4 on a WMMA matmul) -- the guard should stay; the lowering bug is the open item if anyone wants GROUP+TC.

## Files
postrange.py:173 (guard, reverted). Sweep inline. Prior: POWN, wmma-both-levers-conclusion, wmma-occupancy-frontier-scope.
