# NV queue/PDL theory test result (2026-08-20)

Status: measurement result. No runtime files were changed; the two test arms
are environment-only. The token stream was byte-identical in every bracket.

## 1. Question

Two competing theories explained why llama.cpp beats tinygrad on the d512
decode route:

- Theory A: llama's overlap is recoverable on tinygrad with programmatic
  dependent launch (PDL), armed in the correct direction, big GEMV producers
  releasing small norm/rope/flash/residual consumers. The current exposure is
  a schedule artifact, not a data dependency.
- Theory B: the independent branches are memory-bound, the transferable
  overlap is small, and the remaining win is fusion, not launch overlap.

Both were tested as reverse control/candidate/control wall brackets in fresh
processes on the same RTX 5090 session.

## 2. Harness

| item | value |
| --- | --- |
| commit | `6570abc02` |
| model | `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` |
| route | `DEV=NV`, d512, 24 count x 4 reps, 96 timed tokens per process |
| wall tool | `extra/llm_research/decode/nv_norm_native_wall_ab.py` |
| token gate | SHA-256 stream hash, all arms identical |
| GPU serialization | `flock /tmp/gpu-bench.lock`, one process at a time |

PDL candidate environment:

```text
NV_PDL_PRODUCER_PROGRAMS=prefix:q4k_,prefix:q6k_
NV_PDL_CONSUMER_PROGRAMS=prefix:reduce_output_rmsnorm,prefix:E_,prefix:r_,prefix:flash_,prefix:rmsnorm_q8_1_llama_provider
NV_PDL_TRIGGER_POSITION=end (default)
```

## 3. Results

| bracket | control median ms | candidate median ms | delta us | tokens |
| --- | ---: | ---: | ---: | --- |
| Theory B: default 2q control vs `HCQ_NUM_COMPUTE=1` | 4.715511 | 4.828731 | -113.219 | identical |
| Theory A: 2q control vs 2q PDL | 4.725950 | 4.737590 | -11.641 | identical |
| Theory A: 1q control vs 1q PDL | 4.845295 | 4.853496 | -8.201 | identical |

Delta is `(control median - candidate median) * 1000`. Negative means the
candidate is slower. A positive delta would have been wall recovery.

## 4. Verdicts

Theory B is partly falsified. The second compute queue is worth 113.219 us of
real wall time at current HEAD. That is not negligible, so "transferable
overlap is small" is wrong. It is still only about one sixth of the 717.505 us
llama gap and does not close the route. The default two-channel construction
should stay on.

Theory A is falsified in both queue modes. Correct-direction PDL costs
11.641 us on two queues and 8.201 us on one queue with no wall recovery. The
single-queue arm maximizes same-queue QMD latch pairs, so this is not a
pairing-count artifact. The native latch release is bounded to the producer's
last wave, and the signaling it buys is cancelled by its launch cost on the
real route. Extending the full-chain producer set is not worth the next run.

The surviving path is the existing fusion ranking: fold the residual
elementwise add/mul into the GEMV epilogue, then fold the output-reduction
work in-kernel. Those are the two positive legal ceilings after the measured
copy-free RMSNorm arms.

## 5. Topology caveat

The PDL `PROFILE=1` capture could not be booked. PDL changes HCQ graph
finalizer order, so the last five profile records come from different replay
cycles and their timestamps are not one token. The profile-window matcher was
hardened to prefer per-group program-name sequences, but name sequences repeat
every replay and still do not disambiguate the cycles. The wall brackets above
are the authoritative measurement. A future PDL topology capture needs an
invocation id in the graph-profile payload, which would be runtime tooling and
was out of scope for this environment-only test.

## 6. Artifacts

- merged evidence:
  `docs/task_workflow/evidence/nv-pdl-queue-theories-20260820/summary.json`
- raw child outputs:
  `docs/task_workflow/evidence/nv-pdl-queue-theories-20260820/`

## 7. Reconciliation

All nine child runs share token SHA
`1e73e557e48b0c2f0792318e1a306f06a1412cd9800ba7a1e667b9c09c4a1254` and had no
rejected high samples. Closing control drift was below 16 us in every bracket.
The service was restored after the test and `/health` was verified.
