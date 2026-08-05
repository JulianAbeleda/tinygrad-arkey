# NV decode overlap: bootstrap compute-channel construction record

Date: 2026-08-05
Target: native `DEV=NV`, RTX 5090 / 595.84
Status: **G1 PASS for light/support work; heavy-GEMM policy remains closed**

## Question

Earlier native arms created additional compute GPFIFOs after tinygrad had already
issued the first group-level `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE`.  CUDA's trace
instead creates all observed stream compute channels under one async context share
and only then schedules the group.  Does that construction ordering make two
native compute channels executable and co-schedulable?

## Construction

The probe-only `bootstrap_cuda` arm sets `HCQ_NUM_COMPUTE=2` before `Device["NV"]`
is constructed.  `NVDevice` creates both compute GPFIFOs before its sole initial
group schedule; the default remains one channel.  Each channel gets its compute
object and normal UVM registration.  The experiment deliberately does not attach
a DMA object to the compute GPFIFOs: the CUDA-shaped attempt to do so was rejected
by RM with `NV_ERR_INVALID_STATE` before execution.

This differs from the earlier `cuda_mirror` arm in exactly the load-bearing order:
it does not unschedule/re-schedule an already-live native bootstrap group.

## Measurements

All runs used `flock /tmp/gpu-bench.lock`, independent timestamp signals, and the
hash plus declared max-error R1 contract.  Raw JSON stays in `/tmp` and is not a
repository artifact.

| arm | R1 cross-GPFIFO contract | R3 work | R3 interval-union overlap |
| --- | --- | --- | ---: |
| bootstrap, initial discovery | pass; both hashes exact, max error 0 | `n=2^20`, 16 replays/queue | 17.8% |
| bootstrap repeat 1 | pass | `n=2^22`, 16 replays/queue | 9.90% |
| bootstrap repeat 2 | pass | `n=2^22`, 16 replays/queue | 7.08% |
| bootstrap repeat 3 | pass | `n=2^22`, 16 replays/queue | 9.89% |
| bootstrap R5 | numeric contract passes | two 1024 GEMMs | -0.1% |

The R3 result clears the predeclared >=5% G1 gate in every repeat.  The GEMM row
is intentionally a non-promotion result: it demonstrates bandwidth/engine
contention for this heavy pair, not failure of the construction mechanism.

## Causal conclusion

Native hardware concurrency is now demonstrated for independent light work.  The
prior blocker was not a missing public `BIND`, UVM registration, context-buffer
size, notifier size, or runqueue bit.  It was that secondary channels had not been
members of the group's first scheduling construction.  Re-scheduling an already
bootstrapped group did not recreate that state.

This result does not justify scheduling the decode MMQ chain across queues.  The
next gate is an opt-in graph schedule that admits only exact support-kernel names
from a full-token DAG, retains `_resolve_deps` cross-queue waits, joins auxiliary
queues through the primary timeline owner, and proves full logits/tokens plus a
real-token >=50-us wall win.

## Code and default contract

- `NVComputeQueue(queue_idx)` targets `compute_gpfifos[queue_idx]`.
- `NVDevice.hw_compute_queues()` exposes one or two factories.
- `HCQ_NUM_COMPUTE` defaults to `1`; values above the currently qualified two
  channels fail closed by cap rather than silently claiming an unmeasured route.
- HCQ graph multi-queue selection is closed by default and only activates for
  exact program names supplied in `HCQ_NV_MULTI_QUEUE_PROGRAMS`; no role or
  kernel-class inference admits GEMV/MMQ work.

## D3/D2 smoke qualification

The smallest HCQ graph integration was run with `HCQ_NUM_COMPUTE=2` and the
explicit support-only `prefix:E_` admission on a three-program synthetic graph
(two independent multiplies followed by their add).  Five captured replays
returned the exact expected output (`2.0`) without a cross-queue join hang.
The primary queue explicitly waits for an auxiliary terminal signal before
advancing the normal device timeline; this is a required correctness property,
not a performance optimization.

That synthetic graph did not earn a decode recovery: its large two-multiply
nodes still saturated the device and profile timestamps serialized.  A first
token test was started only after other flocked qualification work was already
holding the GPU lock, so it was cancelled rather than queue a stale wall arm.
The next valid measurement is a fresh, flocked d512 reverse A/B using the
full-token-DAG-derived support allowlist, followed by full logits/tokens and a
>=50-us wall gate.  No token-time credit is booked from this record.

## P4 exact-support result and close gate

The closed six-name allowlist (SHA-256 over comma-joined names
`98f2a136dfb615f80c8f3f5fd6c2c48ee2ba3289134dde4743314817f728b130`) passed
the native d512 full-logits gate: float32 logits, returned argmax, and generated
tokens were bitwise identical to the one-queue `DEV=NV` control.  The opt-in
construction census proves this was a real placement experiment, not a dormant
switch: the actual 467-call decode graph placed 51 calls on queue 1 (17 each of
`E_4_2_8_16_4_...`, `r_2_8_4_4_16_...`, and `E_2_8_16_4_4_...`).

The reverse timing bracket is a wall NO-GO:

| arm | median ms/token |
| --- | ---: |
| A1, one queue | 5.5321315 |
| B, two queues + six exact names | 5.5564875 |
| A2, one queue | 5.5012025 |

Thus B was 24.356--55.285 us slower than its bracketing controls.  It cannot
claim the predeclared >=50-us recovery.

Before spending another GPU arm, each of the three families actually present in
the 467-call graph was ranked independently using the captured 948-node
duration/dependency ledger and the same least-populated-queue policy.  The
single-family raw CPU savings / cross-queue waits were respectively 41.312 us /
319 (`E_4_2...`), 70.304 us / 250 (`r_2...`), and 63.328 us / 181
(`E_2_8...`).  Calibrating the six-name regression conservatively gives a
per-cross-edge cost of approximately `(236.608 us predicted saving + 39.820 us
observed regression) / 761 = 0.363 us`.  Net predictions are consequently
-74, -20, and -2 us; the 467-call graph has fewer instances than the ledger, so
these are optimistic upper bounds for the real token.  No individual family
meets the required >=50-us predicted net, and no MMQ/flash family was admitted.

**P4 is closed:** native support-program concurrency is construction- and
correctness-qualified, but the only safe, dependency-accounted support split is
wall-negative.  Reopening requires a new independently captured candidate with
both a >=50-us net prediction after wait-cost calibration and a materially
smaller cross-queue boundary; broad prefixes, MMQ movement, and another
unranked GPU timing arm remain out of scope.
