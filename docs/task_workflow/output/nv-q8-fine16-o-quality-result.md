# Fine-grain Q8 Flash-to-O quality result

## Decision

Scale-per-16 signed-Q8 activations are **not admissible at economically useful coverage** under the established recurrent full-logit contract. No timing was run and no production admission was added.

The representation is materially more accurate than the previous scale-per-32 Q8_1 experiment, but it remains above the required relative-L2 limit:

| Coverage | Recurrent logit rows | Relative L2 | Limit | Other semantic checks | Decision |
|---:|---:|---:|---:|---|---|
| 36 layers | 1 | 0.002056 | 0.001 | tokens, argmax, top-10 set/order, margin pass | Fail |
| 18-layer prefix | 3 | 0.001614 | 0.001 | tokens, argmax, top-10 set/order, margin pass | Fail |
| 8-layer prefix | 3 | 0.001465 | 0.001 | tokens, argmax, top-10 set/order, margin pass | Fail |

Eight layers were the lowest initially useful dose. Since that dose remains 47% above the numerical limit and the error declines weakly with coverage, isolated-layer hunting was stopped as economically unjustified.

## Representation contract

The producer preserves Flash combine's existing FP16 rounding point, then quantizes each aligned group of 16 activations with one FP16 scale. The Q4_K O consumer applies the matching scale independently to each half of a 32-value Q4 group.

| Item | Scale-per-32 Q8_1 | Scale-per-16 candidate |
|---|---:|---:|
| Signed-Q8 payload | 4,096 bytes | 4,096 bytes |
| Metadata groups | 128 | 256 |
| Metadata | 512 bytes | 1,024 bytes |
| Total packet | 4,608 bytes | 5,120 bytes |
| Saving versus 8,192-byte FP16 | 3,584 bytes | 3,072 bytes |

The finer representation therefore adds 512 bytes of metadata per layer boundary, an 11.1% packet increase over Q8_1, while retaining a 37.5% byte reduction versus FP16. It also requires twice as many max reductions, scale constructions, metadata stores, and scale loads as Q8_1. Those costs were deliberately not timed because quality failed first.

## Scope and evidence

The implementation is fail-closed behind the block-local `o_q8_fine_owned` research geometry. No loader, generated policy, or production route enables it. The existing Q8_1 route remains separate and unchanged.

Primary evidence is under `docs/task_workflow/evidence/nv-q8-fine16-o-quality-20260828/`. The qualification harness is `extra/llm_research/decode/nv_flash_combine_q8_fine_o_qualification.py`.
