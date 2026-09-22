# Native NV d512 predispatch A/B record

Date: 2026-08-05. Route: `DEV=NV`; Qwen3-8B-Q4_K_M; depth 512;
RTX 5090 / driver 595.84. Status: **both measured predispatch costs move
whole-token wall; neither candidate is promoted.**

## Result

Seven sequential arms ran under `/tmp/gpu-bench.lock`, with 30 steady tokens
per arm after three settling tokens. Each candidate had an immediately adjacent
control on both sides. All seven arms produced the exact same 30-token stream
(SHA-256 `4c0986e6f829832e9133df0d47007396473dcaaa0a1d91e42aa394583c265b9d`).

| candidate | control midpoint us/token | candidate us/token | delta us | delta |
| --- | ---: | ---: | ---: | ---: |
| A: cached structural descriptor | 5616.477 | 5550.942 | -65.536 | -1.167% |
| B: reusable private feedback shadow | 5607.171 | 5578.799 | -28.372 | -0.506% |
| A+B | 5599.054 | 5529.888 | -69.166 | -1.235% |

The combined result is smaller than the sum of the individual point estimates.
These paths remove overlapping host latency and/or expose ordinary bracket
drift; their improvements must not be added independently.

## A: settled structural descriptor

The diagnostic cache is deliberately identity-strict. It caches only the
expensive substitute/rewrite/unbind structural result for the immutable input
UOps. Every invocation still performs tensor discovery and realization,
collects the current concrete buffer UOps, rejects duplicate inputs, extracts
fresh bound-variable values, and runs `TinyJit`'s existing name and expected
descriptor comparisons. A new tensor/view misses and uses the full oracle path.

On live decode the first three calls missed and every subsequent lookup hit
(33/36). Median `_prepare_jit_inputs` time fell from 78.095 and 75.840 us in
the adjacent controls to 28.092 us. Hermetic tests prove that a different
allocation with the same contract binds its new contents, while changed rank,
dtype, and device reach the existing fail-closed `JitError` checks.

## B: reusable alias-safe feedback shadow

The diagnostic allocates one private shadow per captured-JIT input slot and
reuses that allocation. It does **not** remove the defensive copy: every replay
still builds and executes `source.copy_to_device(...).call(shadow, source)` via
`run_linear` before the captured graph, preserving stream ordering and the
written-input contract. The retained copy path measured 89.226 us median and
the whole-token reverse bracket improved by 28.372 us.

This proves that fresh shadow allocation/UOp construction was avoidable work;
it does not prove the copy itself can be removed.

## Interpretation and limits

The experiments causally recover about 69 us/token together, or 4.2% of the
authoritative 1.646170 ms native-minus-llama gap. They validate two real host
costs but do not explain the remaining gap. Exact output-token equality is the
available production graph-output contract; the sampled graph does not expose
its internal full logits buffer, so this record does not claim an independent
full-logit hash.

No core/default file changed. The candidates live only in
`scratchpad/nv_decode_predispatch_ab.py` and are installed/restored at runtime.
The compact payload is
`docs/task_workflow/output/nv-decode-native-d512-predispatch-ab-20260805.json`.
Raw evidence is `/tmp/nv_decode_predispatch_ab_20260805.json`, SHA-256
`c5e8f3371b7b314ec89eaffd68cb18a7296d87b86906f4e983845b6967da1841`.
Six relevant hermetic tests pass. Nothing was committed or pushed.
