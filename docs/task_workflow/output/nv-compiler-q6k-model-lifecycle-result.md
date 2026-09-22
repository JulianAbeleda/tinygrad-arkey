# NVIDIA pp512 compiler Q6_K V/down lifecycle gate

## Verdict

**Compiler substrate PASS; model-lifecycle performance FAIL. Do not promote.**

The default-off `NV_COMPILER_Q6_IMMA_PP512=1` experiment correctly routes the
Qwen3-8B Q6_K population on top of compiler gate/up + K:

- 18 `attn_v` and 18 `ffn_down` compiler mains;
- 36 compact-Q8 producers and 36 distinct canonical uint16 model buffers;
- no Q6 FP16 overlays, expanded weights, weight copies, group partials, or
  fixups;
- 256-CTA V and 1024-CTA down launch geometries;
- Q4 V/down remain outside the route and retain all 36 FP16 overlays.

A/B/A replay is bit-exact and activation-sensitive. Candidate/control logits
are finite, select the same greedy token, and pass `allclose(rtol=.02,
atol=.5)` (maximum absolute difference 0.160705, mean 0.0315351).

Those facts qualify ownership and correctness, not speed. The clean pinned
wall bracket rejects the combined lifecycle.

## Baseline contamination caught and removed

The first apparent win was invalid. An integration edit flattened the
Q6-disabled FFN-down fallback from rank three to rank two. That changed 72
kernels across the model and moved the gate/up+K control from the qualified
70 ms class to 87 ms. The false bracket therefore compared against a damaged
control.

The fallback is now byte-for-byte rank preserving. Flattening occurs only
inside the enabled Q6 branch, and its result is reshaped back to the original
rank. After correction:

- the old and new K-authority `program_names` maps are exactly equal;
- new K-authority logits are bit-exact to the historical v3 artifact;
- the fresh K authority is 70.192152 ms minimum versus 70.390585 ms
  historically.

All final runs pin the HCQ policy to two compute queues, ready placement on,
and empty multi-queue program/index/cut selectors.

## Corrected wall result

Fresh isolated processes, synchronized R9, same model/flags/queue policy:

| route | minimum | median | prompt throughput | minimum delta vs control |
| --- | ---: | ---: | ---: | ---: |
| gate/up + K control | 70.332612 ms | 70.511929 ms | 7279.70 tok/s | — |
| Q6 V only | 70.208615 ms | 70.368576 ms | 7292.55 tok/s | -0.123997 ms |
| Q6 down only | 71.943946 ms | 72.219183 ms | 7116.65 tok/s | +1.611334 ms |
| Q6 V + down, A | 72.174425 ms | 72.350026 ms | 7093.93 tok/s | +1.841813 ms |
| Q6 V + down, confirm | 72.284369 ms | 72.344973 ms | 7083.14 tok/s | +1.951757 ms |

V alone is only a roughly 0.12-0.14 ms improvement, inside the range that
needs a tighter confirmation before any role-only claim. Down is decisively
negative and makes the combined route about 1.9 ms slower, or roughly 186-197
prompt tok/s below the matched control. The comparison harness has an explicit
`candidate_faster` predicate and therefore records the combined result as
`FAIL`, even though all structural and correctness predicates pass.

## Why the primitive does not translate

The Q6 compiler body is real and exact, but its current lifecycle pays three
costs that the resident FP16 down path does not:

1. each unique down activation first creates a new compact-Q8 record;
2. Q6's independent K16 scales require two masked K32 IMMAs, doubling the
   tensor-core integer instruction work for each logical K32;
3. FFN-down must pin the compiler-produced PROGRAM before the residual
   epilogue, while the resident FP16 graph retains its mature rank-preserving
   fused lifecycle.

The compact canonical weight eliminates the FP16 overlay and reduces weight
traffic, but at this pp512 down shape those savings do not repay producer,
paired-IMMA, and lifecycle costs. This is the same kernel-versus-token lesson:
an isolated valid body is not automatically a faster token path.

The plausible reopen conditions are a shared/fused Q8 producer (especially
for projections consuming the same activation), a native way to expose two
K16 subtotals without two full masked K32 instructions, or a compiler-owned
down residual epilogue that preserves the admitted identity. None is proven
here, so current token recovery booked from Q6 V/down is zero.

A final post-review fresh-process structural requalification was run after the
last model and harness edits. It again reports PASS for the exact 18 V + 18
down census and bit-exact, activation-sensitive A/B/A replay. The artifact
embeds and verifies source SHA-256 hashes and timestamps. Its single timing
sample is structural-only and does not replace the synchronized R9 performance
decision above.

## Evidence

- Compiler/oracle regression: `docs/task_workflow/evidence/nv-compiler-q6k-model-20260828/q6-regression-r9.json`
- Reproduced K authority: `docs/task_workflow/evidence/nv-compiler-q6k-model-20260828/repro-k-authority-fixed-r9.json`
- Corrected control: `docs/task_workflow/evidence/nv-compiler-q6k-model-20260828/final-control-r9.json`
- Combined candidate A/B: `final-candidate-a-r9.json`, `final-candidate-b-r9.json`
- Explicit failed performance comparison: `final-compare.json`
- Current-tree structural/hash authority: `current-tree-structural.json`
- Role splits: `role-v-r9.json`, `role-down-r9.json`
- Harness: `extra/llm_research/prefill/nv_compiler_q6k_model_arm.py`
