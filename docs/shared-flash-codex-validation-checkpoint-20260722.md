# Shared flash attention validation checkpoint

## Scope

This checkpoint records only the state of the committed semantic attention
boundary and bounded Tensor primitive. It is not a promotion, performance, or
roofline claim.

## Evidence collected

Command:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  test/unit/test_attention_semantic.py \
  test/unit/test_shared_prefill_measurement.py \
  test/unit/test_attention_residency_contract.py -q
```

The first two files passed before this checkpoint test was added: `5 passed`.
The new structural test checks `T=129`, which crosses the 64-token KV block
boundary. The semantic primitive contains no Tensor-graph `BUFFER` with trailing
logical shape `(129, 129)`. This proves only that the constructor does not
create a full score/probability Tensor buffer.

Schedule inspection command:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
for t in (32, 129):
  q = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  k = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  v = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  linear = shared_prefill_attention(q, k, v).schedule_linear()
  topo = linear.toposort()
  print(t, sum(x.op is Ops.CALL for x in topo), sum(x.op is Ops.WMMA for x in topo))
PY
```

Observed results:

| T | Scheduled calls | WMMA nodes |
|---|---:|---:|
| 32 | 5 | 0 |
| 129 | 10 | 0 |

## Meaning

The current implementation is a correct, fail-closed semantic boundary plus a
bounded ordinary-Tensor online-softmax primitive. It is not the required fused
attention schedule. In particular, it has not passed the scope's D residency
gate, E dual-WMMA gate, F geometry gate, G roofline gate, or H model promotion
gate.

The primitive remains useful as the safe intermediate route: it avoids the old
unsound reverse matcher and retains ordinary SDPA fallback whenever semantic
eligibility cannot be proven. It must not be enabled or benchmarked as flash
attention until the remaining gates have direct generated-schedule evidence.

## Remaining validation gates

- One bounded score-resident compute schedule, not a per-KV-block call chain.
- QK and PV contraction provenance inside that schedule.
- Two generated AMD WMMA invocation sites under `NOOPT=0`.
- Allocation and code evidence that no score/probability global buffer exists.
- Numeric matrix through masks, GQA, `Hd=1/64/128`, and real long contexts.
- Shared geometry search evidence for 8B and 14B.
- Whole-model route census, deterministic parity, prefill GPU timing, and
  decode non-regression for both models.
