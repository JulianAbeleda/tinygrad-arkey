# NVIDIA pp512 ranked test-invest campaign closure

Date: 2026-08-29

## Outcome

The low-agent campaign covered every row in the ranked plan. It did not turn
profile percentages into speculative wall recovery. Each row was either
measured through its applicable gate, closed as a negative, retained as an
existing win, or stopped at a precise missing substrate boundary.

The current-tree matched wall authority is now the composed unroll-4 + Q4-V route:

| runtime | R9 minimum | settled median |
|---|---:|---:|
| tinygrad compiler-packed gate/up + K + Q/O + unroll4 + Q4-V | 67.153915 ms | 67.235719 ms |
| llama.cpp | 34.680367 ms | 35.019399 ms |
| difference | 32.473548 ms | 32.216320 ms |

This is a current-tree matched result. The composed candidate is exact and
retained default-off. Against the matched unroll-4/no-Q4-V control at
69.165843 ms minimum / 69.315714 ms median, Q4-V recovers 2.011928 ms minimum
and 2.079995 ms median. The standalone unroll-4 result and the older
69.378154-ms authority are precursor results with different generated-program
inventories, not controls for the composed result.

The traced device-union difference is also service dominated: tinygrad is
73.879808 ms versus llama at 32.683341 ms. Tinygrad has only 0.105790 ms more
device idle. The remaining gap is executed kernel/lifecycle service, not an
idle-GPU gap that generic overlap can erase.

## Exhaustive decision ledger

| rank | lane | result | investment decision | exact next prerequisite |
|---:|---|---|---|---|
| 1 | final-layer row prune | correctness PASS, captured performance FAIL | STOP | none; a packed M=1 kernel is not justified by the graph-law result |
| 2 | vocabulary | existing full-logit and top-1 assets FAIL wall | STOP current assets | a new many-row full-logit packed-Q6 service design |
| 3 | gate/up | matched physical counters PASS; unroll4 matched whole-model win | RETAIN default-off | further scheduling only with the same full lifecycle gates |
| 4 | V, Q6 | primitive PASS, model V-only gain marginal and combined lifecycle loses | STOP | a materially different lifecycle, not another geometry sweep |
| 4 | V, Q4 | 18-role model gate and composed unroll4 gate PASS | RETAIN default-off | only revisit with a new lifecycle mechanism |
| 5 | Q/O | existing default-off route PASS with safe dependency cut | RETAIN | production-generic dependency-identity queue policy before promotion |
| 6 | down, Q6 | primitive PASS, model down-only and combined wall FAIL | STOP current spelling | different producer/K16/epilogue lifecycle |
| 6 | down, Q4 | matched A/B correctness and performance FAIL | STOP | new Q4 down lifecycle/substrate |
| 7 | Flash | live capture/oracle PASS; S6 candidate service FAIL | STOP current spellings | new vectorized topology |
| 8 | K | existing Q4 route PASS and integrated | CLOSE | reopen only with matched cold producer-to-main/cache evidence |
| 9 | support | all 829 launches mapped, zero unknown | MAPPING COMPLETE | rank exact transport families only after dense blockers are resolved |

## Gate results

### Final-layer row prune: closed

The numeric row comparison passes with `max_abs=0.0059030056`,
`mean_abs=0.00099487335`, and `relL2=0.00030978036`; argmax and top-10 are
exact. The production captured bracket is decisive:

| arm | R9 minimum | median | dense census |
|---|---:|---:|---|
| unpruned | 68.992404 ms | 69.177272 ms | gate/up 72, K 36, Q/O 72 |
| final-row pruned | 69.269884 ms | 69.343492 ms | gate/up 70, K 36, Q/O 72 |
| prune delta | +0.277480 ms | +0.166220 ms | two final gate/up calls removed |

The candidate is correct but slower. Do not build packed M=1 for this path.

### Vocabulary: current assets closed

The existing full-logit Q6/Q8 candidate is `1368.125 us` median versus the
installed full-logit Q6/FP16 kernel at `324.636 us`, a recorded debt of
`1043.235 us`. The quality-passing four-accumulator spelling also regresses the
model bracket by `5.876 us`, and the greedy top-1 route does not preserve the
full-logit API while also losing wall. Decode M=1 assets are therefore not a
prefill solution. A new design must directly target the many-row vocabulary
projection and retain full logits.

### Gate/up: arithmetic solved, service diagnosis incomplete

The real `(512,12288,4096)` compiler-owned K64 primitive passes the complete
output oracle (`max_abs=2.136e-4`) but measures `483.920 us` median versus the
qualified v4 chain at `464.352 us`, a `4.22%` loss. K32 did not supply a valid
replacement. The matched counter bridge is now complete. Tinygrad and llama
both execute `6,291,456` IMMA instructions; L2 requests are `412.892` versus
`386.984 MB`. Tinygrad nevertheless takes `409.312` versus `219.200 us`,
executes 24.4% more total instructions, reaches only `14.65%` versus `31.71%`
tensor duty, has `0.554` versus `0.807` eligible warps per cycle, and incurs
`0.535` versus `0.324` long-scoreboard stalls per issue-active cycle. The
selected test is a register-safe K-step/fragment load-to-use scheduling
change. Additional bandwidth, cp.async, TMA, or overlap work is not authorized
by these counters.

### V: format split

Q6 V is arithmetically valid, but the model-lifecycle result is not strong
enough to book: V-only moved `70.332612` to `70.208615 ms`, while Q6 down and
the combined arm regress decisively. The current Q6 spelling remains closed.

Q4 V is now qualified for the exact 18 type-12 projections. The producer ABI
symbol was corrected, the first real Q4 weight (`blk.4.attn_v.weight`) passes
finite full-output checks, and the non-vacuous deep20 graph replay is exact for
V records/outputs, KV, logits, and token. A matched current-tree R9 gives
`72.116495 ms` versus `74.002698 ms` without Q4 V: `-1.886203 ms` minimum,
`-1.903406 ms` median, and `+2.62%` throughput. The 18 type-14 Q6 V
projections remain FP16. See `nv-prefill-q4v-result-20260829.md`.

### Q/O: retain the measured route

The isolated role proxy and full model route pass correctness. In the combined
gate/up+K+Q/O graph, the graph-derived Flash dependency cut gives `69.378 ms`
versus the `70.173 ms` gate/up+K control. Default ready placement is faster in
some samples but fails recurrent replay; primary-only and one-queue policies
are correct but slower. Retain the default-off safe-cut arm. Further work is a
production-generic dependency-identity policy, not another Q/O kernel sweep.

### Down: Q4 and Q6 require different decisions

The real blk.4/type12 K=12288 compact-Q8 producer passes the saved-z gate
(`q_mismatch=0`, scale and raw-sum mismatches zero), and the Q4 main passes its
full-output oracle (`max_abs=0.00146484375`, unwritten sentinels 0). Those
primitive results do not survive the matched 18-role lifecycle. The corrected
device-side A/B fails numerical tolerance (`max_abs=2.695646`) and is slower:
367.76 versus 365.56 ms minimum and 368.97 versus 367.22 ms median. Stop this
spelling; the earlier zero-output/debug and host-readback comparisons are
superseded and are not authority.

Q6 down already has an exact compiler primitive, but the matched model arm
regresses by about `1.61 ms` down-only and `1.84-1.95 ms` combined. Stop the
current spelling; arithmetic validity is not the missing piece.

### Flash: exact route known, fixture binding missing

The installed NVIDIA route is the finalized fused program
`nv_sm120_q16_grid_hd128_loop_attention`, 36 calls at global 1024/local 32,
with FP16 Q/K/V/output buffers. The live capture/oracle gate passes: one
selected call, 26 graph calls, fresh replay, full coverage, and
`max_abs=0.00930290096` with the recorded allclose criterion. The retained
trace assigns `3.328096 ms` to tinygrad Flash versus `1.657447 ms` to llama.
The installed-population comparison is complete; current S6 spellings are
closed because the cooperative candidate is far slower. A new vectorized
topology would be a separate substrate project.

### K and support: deprioritized, not unknown

K already uses the canonical packed weight, compact Q8 record, signed IMMA,
and a 256-CTA route; its whole-model integration recovered `4.004704 ms` in
its matched bracket. The residual K debt is `0.959419 ms`, so it is closed
behind V, down, gate/up, and Flash.

The post-unroll exact trace accounts for 1,449 launches with zero unknowns;
its support split is 937 residual/RoPE/KV/support launches (`501106.784 us`),
253 norm/activation-conversion launches (`1376.896 us`), and 180 compact-Q8
producer launches (`525.088 us`). This is a `PROFILE=1` trace, so it is
instrumentation-perturbed and not unprofiled wall authority. The earlier exact
support census of 829 launches / `3118.624 us` remains attribution only.

## Next execution order

The next campaign should remain test -> test -> invest:

1. Build a low-perturbation HCQ-native timestamp census for the exact composed
   graph. The current `PROFILE=1` trace changes the wall too much, so it cannot
   rank the remaining 32.473548-ms minimum gap by subtraction.
2. Continue gate/up scheduling only if a new register-safe discriminator beats
   the retained unroll4 route under the same full lifecycle gates.
3. Keep Flash S6 closed; reopen only with a genuinely new vectorized topology.
4. Design a genuinely many-row full-logit Q6 vocabulary kernel. Do not reuse
   the rejected decode M=1 or top-1 routes.
5. Reopen Q4/Q6 down only with a materially new producer/correction/epilogue
   lifecycle; geometry tweaks to the rejected spellings are closed.
6. Only after those lanes are resolved, rank the mapped support transport
   families and revisit the residual K service.

Every investment remains default-off until it passes primitive correctness,
population lifecycle, full logits/token and recurrent replay, and matched R9
wall in that order.

## Execution-status addendum (2026-08-29)

This addendum supersedes the stale row labels above. Q4-V is **RETAIN,
default-off**: its 18-role model gate, deep20 replay, and matched wall result
are recorded in `nv-prefill-q4v-result-20260829.md`. Q4-down is **STOP** for
the current spelling: the authoritative matched A/B fails the declared
allclose gate (`max_abs=2.695646`) and is slower; see
`evidence/nv-q4down-matched-ab-20260829/result.json`. Flash S6 current
spellings are **STOP/not integrated**: the corrected standalone probes pass
oracle checks, but the installed-population comparison rejects the candidate
on service time.

The composed candidate passes exact deep20 replay, full logits/token, and the
198-role census. It improves its matched unroll4/no-Q4-V control (69.165843 /
69.315714 ms) by 2.011928 ms minimum and 2.079995 ms median (+2.996%). The
standalone Q4-V bracket remains precursor evidence only.
