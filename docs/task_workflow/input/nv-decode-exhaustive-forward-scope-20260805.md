# NV decode exhaustive forward scope: overlap -> fusion -> host

Date: 2026-08-05
Status: **exhaustive scope; CPU dev in flight; all GPU arms parked**
Authority: `nv-decode-final-composed-same-session-record-20260805.md`
Constraint in force: no GPU use. Every GPU arm below is gated behind a
CPU-verified forecast or an exact-output correctness contract.

## Recovery target and equation

Parity means matching llama's composed steady state: **249.65 tok/s**
(`4.0056768 ms/token`) from native's **187.82 tok/s**
(`5.3242440 ms/token`), a gap of **1318.5672 us/token**. The user accepts
recovering the full enumerated gap, in campaign order: **overlap first,
fusion second, host last**.

The fixed-authority equation the recovery plan must respect:

```text
1646.170000 us/token
= support-work exposure 1108.082   (hidden-overlap 445.954
                                   + fusion/dataflow 662.128)
  + quant cores         302.788
  + outside window      239.805
  - llama gaps            8.111
  - bridge                1.143
  + outer reconciliation  4.749
```

Already booked against the fixed authority: P1 66.662 + P2 75.031 +
P5 91.637 + Q4-g12 24.676 + max17 12.462 = **270.468 us**, leaving
**1375.702 us**. The composed baseline already contains P1/P2/P5/Q4-g12, so
the remaining composed gap of 1318.567 us is what overlap + fusion + host
must close. The composed host residual is about 210.5 us after the host-side
P1/P2/P5 recoveries; the outside-window term is the single largest cheap host
target.

## Workstream ordering and owners

### 1. Overlap (445.954 us, +17.17 tok/s) -- first

Two-queue cuts are closed as scoped: the calibrated wait cost is
**3.1865 us/wait** on the redirect-on authority DAG (bisected so the Q-cut
forecast reproduces the P4 wall at -10.474 us); the K cut lands at
+42.962 us, below the +50 us promotion gate. The only remaining overlap
mechanism is **in-graph co-scheduling without cross-queue waits** (llama's
driver co-schedules independent nodes on one CUDA graph). Exhaustive
mechanism enumeration and the co-schedule candidate scan:
`nv-overlap-exhaustive-scope-20260805.md` + `nv_co_schedule_candidate_scan.py`.

Gate: no GPU arm until a dependency-independent (support, quant) pair with a
positive resource-complementarity bound is named by the resource join, or a
co-schedule candidate clears +50 us on a fresh calibrated forecast.

### 2. Fusion / dataflow (662.128 us, +26.67 tok/s) -- second

Attribution rows (norms 574.654, flash 247.989, residual/cast/contiguous
240.319, vocab/feedback 71.215, RoPE/KV 33.543, llama Q8 pack -59.639) are
ownership, not savings. The one admissible construction is an ordinary-UOp
in-core projection epilogue with no custom boundary or adapters; all
custom-boundary epilogues are closed (attention-O +69, llama-O +21.2, KV
neutral, FFN-down neutral, RMSNorm +60.802 with 110 lazy-view kernels).
Exhaustive population enumeration and the exact-output A/B design per
population: `nv-fusion-exhaustive-scope-20260805.md` +
`nv_fusion_population_ledger.py`.

Gate: nothing in the 662.128 us row is bookable before an exact-output
native A/B per population (full-logit SHA-256, token stream, argmax).

### 3. Host / outside window (239.805 us, +8.86 tok/s) -- last

Native outside partition (321.784 vs llama 81.979): pre-first-graph 247.557
(defensive copy/rebind 121.380 + JIT input/signature reconstruction 77.927
incl. 49.284 structural rewrite + pre-TinyJit 35.637 + cache 1.112),
copyout/`Tensor.item()` 83.247, yield 3.096, redundant sync 4.358. The
predispatch combined A/B signal is -69.166 us, causal but UNBOOKED until the
full-logit oracle runs. Packed greedy argmax is NO-GO (+70.8 us) and stays
closed. Exhaustive enumeration and the recovery calculator:
`nv-host-exhaustive-scope-20260805.md` + `nv_host_recovery_ledger.py`.

Gate: any host booking requires full-logit equality; JIT signature caching
is closed-default and CPU-testable.

## Cross-workstream acceptance

Stacking the disjoint buckets reproduces parity by construction against the
fixed authority. The composed baseline target is 249.65 tok/s; the
per-bucket marginal contributions are in the parent scope. Work proceeds in
the order overlap -> fusion -> host, each with its own HARD STOP before any
GPU arm.

## References

- `nv-decode-exposure-overlap-host-forward-scope-20260805.md` (parent scope)
- `nv-decode-final-accounting-audit-20260805.md` (location PASS / recoverability FAIL)
- `nv-decode-parity-campaign-reconciled-ledger-20260805.md` (booked recoveries)
- `nv-decode-p4-dependency-closed-cut-record-20260805.md` (wait-cost calibration)
- `nv-decode-native-d512-host-partition-record-20260804.md` (host partition)
- `nv-overlap-exhaustive-scope-20260805.md` (workstream 1, in flight)
- `nv-fusion-exhaustive-scope-20260805.md` (workstream 2, in flight)
- `nv-host-exhaustive-scope-20260805.md` (workstream 3, in flight)
