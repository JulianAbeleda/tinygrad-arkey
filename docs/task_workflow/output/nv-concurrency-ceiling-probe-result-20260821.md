# NV concurrency-ceiling probe result (forced 2-queue overlap)

Date: 2026-08-21

Commit: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-concurrency-ceiling-20260821/`

Probe: `extra/llm_research/decode/nv_overlap_substrate_wall_probe.py`

## 1. Decision

**Decision: forcing the Q->O support work onto the second GPFIFO does not
improve decode; it costs ~657 us/token.  The S1 support is a serial data
dependency chain, not hideable idle work, so queue concurrency is already at
its ceiling on this construction.**

This corrects the earlier ledger-only estimate that concurrency could recover
up to 634 us.  The estimate treated the support as independent, but the
device timeline shows it is a serial chain, and the forced-overlap A/B is
decisive: the mechanism is correct (identical token SHA) yet slower.

## 2. What was measured

Fresh process per arm, `DEV=NV`, unprofiled wall, depth 512, 3 settle tokens,
24 measured tokens, 4 reps, serialized under `flock /tmp/gpu-bench.lock`.
Token-stream SHA is the correctness gate.

| arm | queue placement | tok/s | wall us/token | token SHA |
| --- | --- | ---: | ---: | --- |
| landed | ready placement, 2 GPFIFOs (current default) | 211.43 | 4729.6 | `1d299b89...` |
| forced_flash | name-pinned flash + norms onto GPFIFO 1 | 185.65 | 5386.5 | `1d299b89...` |

The forced arm is **-25.78 tok/s, i.e. +656.9 us/token worse**, with byte-
identical output.

## 3. Why it is worse

The settled device timeline is fully serialized.  In the first attention
layer the Q -> O span is the back-to-back sum of:

```text
Q GEMV (9.25 us) -> norm -> K GEMV (5 us) -> V GEMV (5 us) -> norm
  -> flash score (7 us) -> flash combine (2.75 us) -> O GEMV (9.5 us)
```

Each kernel starts after its producer finishes (0.5-2.5 us gaps).  The flash
attention depends on K/V, and O depends on the flash output, so these kernels
form a critical path.  Moving a dependent kernel to GPFIFO 1 does not overlap
it with anything; it only adds a cross-queue timeline-signal handoff and
breaks the same-queue QMD chaining optimization, which is why the wall grows.

The only genuinely independent work in the chain (the small post-V norm and
residual kernels) is already moved by the landed ready-placement, worth
~140 us versus one GPFIFO (measured earlier in
`nv-overlap-substrate-reuse-lanes-20260819.json`).

## 4. Consequence for the gap budget

The surviving S1 gap is not a scheduling problem.  It is the cost of a longer,
unfused serial chain (separate Q/K/V GEMVs, two flash kernels, two norms, two
residuals) versus llama's leaner chain.  The remaining attack surface is
fusion and leaner kernels, not queue concurrency.
