# NV launch-hiding substrate - exhaustive account and verdict (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `45b59b4a3`)
Status: **account + verdict (read-only).** Reconciles every launch-hiding arm
run to date and states why there is no buildable substrate left to buy the 240
line. The recoverable launch-hiding mass is ~18-33 us, not the ~946 us overlap
mass or the ~0.48 ms sometimes quoted in the ladder.

## 1. The two layers of "substrate"

Launch hiding has two independent layers, and they were built/tested separately:

1. **Hardware substrate** - can two native compute queues physically co-schedule
   independent kernels on GB202 / driver 595.84.
2. **Graph substrate** - does the decode DAG contain dependency-independent
   nodes whose durations can hide inside another kernel's shadow (llama's
   in-graph mechanism).

## 2. Hardware substrate - built and PASSED, economics negative

`nv-rank2-native-concurrency-construction-verdict-20260805.md` closed the RM
construction unknown. Two native compute GPFIFOs under one async ctxshare,
scheduled once at group creation, do co-schedule:

- R1 cross-GPFIFO dependency: exact, max error 0.
- R2 serial calibration: span == node-sum within tolerance.
- R3 independent light work: 9.7% interval-union overlap (repeats 7.1/9.9/9.9%,
  clears the 5% gate every time).

So the hardware substrate exists and is reproducible. The reason it does not
carry the decode split is economics, not construction: the exact support-kernel
split across queues was wall-negative because the cross-queue waits cost more
than the overlap recovered. Heavy GEMV pairs serialize/contend anyway.

## 3. Graph substrate - exhaustively enumerated, every arm below the gate

`nv-overlap-exhaustive-scope-20260805.md` enumerated six mechanisms (a-f) and
`nv-decode-overlap-p2-verdict-20260812.md` reran the decision tools at HEAD on a
fresh duration-bearing DAG + compiled-descriptor census:

| mechanism | result | gate (+50 us) | verdict |
| --- | ---: | --- | --- |
| (a) in-graph co-schedule scan | ceiling 17.952 us (old) / 33.36 us (fresh), greedy 10.016/32.49 us | fail | CPU_NO_GO |
| (b) static reorder | subsumed by (a), adds 0 headroom | fail | no arm |
| (c) support->quant fusion | the fusion workstream, not overlap | - | separate row |
| (d) support->flash fusion | flash single-stage NO-GO (+82.5 us) | fail | closed |
| (e) resource-complementarity join | best pair 3.97 us | fail | INCONCLUSIVE_FAIL_CLOSED |
| (f) algebraic elimination | fusion workstream | - | separate row |
| two-queue Q/K cuts | Q -10.474 us, K +42.962 us (at 3.1865 us/wait) | fail | closed |
| host replay busy | 95.6% busy, host gap 230.1 us | n/a | not the lever |

The scan's key structural fact: tinygrad has 586 support nodes but only ~200
dependency-independent (support, quant/flash) pairs, and flash has **zero**
independent support partners (it sits downstream of rope/KV and upstream of the
residual chain). Most support duration is dependency-bound to the quant
backbone, so there is no shadow to hide it in.

## 4. Why llama's 946 us is not transferable (the reconciliation)

llama replays one CUDA graph with 946.4 us of overlap mass (node-sum 4774.4 vs
span 3835.2 us, 19.7% discount, one stream). tinygrad runs seven streams with
zero overlap mass (span exceeds node-sum by 232 us). The naive reading says
"we are missing 946 us of launch hiding". The per-class subtraction says
otherwise:

| llama hidden class | hidden us | tinygrad fate |
| --- | ---: | --- |
| quantize_q8_1 | 391.3 | folded into GEMV bodies (anchor at parity) |
| rms_norm | 155.9 | fused into norm epilogues (ahead of llama) |
| rope | 32.5 | fused into q/k epilogue (ahead of llama) |
| flash score + combine | 143.0 | the only remaining exposed hidden mass |

The 578 us of quantize/norm/rope hiding is llama's own non-fused cost structure.
tinygrad already captured it by fusion, so there is no corresponding node left
to overlap. The flash pair (143 us) is the only interval still hidden in llama
and exposed in tinygrad, but it too is not an independent lever: the scan
reports zero dependency-independent flash pairs (flash sits on the critical
path between the QKV GEMV and the output-projection GEMV, so it has no MMQ
partner it does not depend on), and the weight-prep phase llama overlaps its
flash against is the quantize stage tinygrad folded into the GEMV bodies. There
is no corresponding node left to co-schedule.

Net: the transferable launch-hiding ceiling is the scan's 17.9-33 us, not
946 us.

## 5. Verdict

The launch-hiding substrate is **exhausted**. The hardware layer was built and
proved to co-schedule; the graph layer was enumerated across all six mechanisms
plus two-queue cuts plus host replay, and every arm is below the +50 us gate.
The reason is not a missing primitive - it is that fusion already absorbed the
work llama overlaps, so there is nothing left to hide.

The gap attribution ladder's step 8 labels the last ~0.48 ms "launch hiding".
That label is a llama-topology artifact and must not drive a substrate build:
the recoverable overlap mass is 17.9-33 us (~0.7-1.3 tok/s), not 0.48 ms. The
authoritative direction remains the kernel-work rows - reduce-output epilogue
(392 us), residual/plumbing (472 us), vocab aux (57.3 us) - which are fusion,
not launch hiding.

## Evidence

- `nv-rank2-native-concurrency-construction-verdict-20260805.md`
- `nv-overlap-exhaustive-scope-20260805.md`
- `nv-decode-overlap-p2-verdict-20260812.md`
- `nv-decode-overlap-reopen-resource-join-scope-20260812.md`
- `nv-decode-native-flash-causal-record-20260805.md`
- ledgers: `nv-llama-d512-node-ledger-20260812.json`,
  `nv-tinygrad-d512-node-ledger-20260813.json`
- path proof: `test/unit/test_nv_parity_path_proof.py`
