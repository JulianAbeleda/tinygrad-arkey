# NV compiler Q4_K gate/up + K + Q/O combined authority

Status: **PASS as an isolated, default-off research arm. Not promoted.**

This gate composes only the independently passing Q4_K roles: 72 gate/up,
36 K, and 72 Q/O projections. Q6 V/down is explicitly excluded. The result is
a correct whole-model pp512 graph with 180 compact-Q8 producers and 180
compiler-generated packed mains, and it is faster than the corrected gate/up+K
control under the conservative Flash-dependency policy.

## Composition finding

The first direct composition failed before capture. Adding Q/O changed the
nested attention scheduling boundary, so K's re-spelled ordinary carrier
matmul reached the typed packed contract without one canonical compact-Q8
PARAM. This was not a leaked Q/O warmstart key: the contract correctly failed
closed with `packed A carrier ... found 0`.

The isolated arm resolves that lifecycle seam without changing the compiler,
scheduler, runtime, or model:

- K invokes its already-qualified frozen compiler PROGRAM with lazy,
  graph-owned record/output storage.
- Q/O invokes its already-qualified frozen compiler PROGRAM with the same
  graph-owned ownership rule.
- Gate/up retains its existing compiler capture.
- Q/O's shape warmstart is not installed into the ambient model table because
  the composed route invokes the frozen PROGRAM directly.

This preserves the exact typed contracts and compiler-emitted binaries while
giving the HCQ graph ownership of all mutable stage allocations.

## Structural authority

| Item | Required | Measured |
|---|---:|---:|
| Compact Q8 producers | 180 | 180 |
| Compiler mains | 180 | 180 |
| Gate/up mains | 72 | 72 |
| K mains | 36 | 36 |
| Q/O mains | 72 | 72 |
| Main weight arguments | 180 | 180 |
| Unique canonical packed weights | 180 | 180 |
| Admitted FP16 overlays | 0 | 0 |
| Remaining V/down FP16 overlays | 72 | 72 |
| Weight-copy kernels | 0 | 0 |
| Old fixups | 0 | 0 |
| Partial workspace | 0 | 0 bytes |

The deep graph census additionally found distinct bound allocations for all
72 gate/up records and outputs, 36 K records and outputs, and 72 Q/O records
and outputs.

## Fresh Flash dependency cut

The policy was generated from this combined graph's own six-segment profile
and queue census. No Q/O-only digest was reused. Every direct predecessor of a
Flash call is pinned to the primary queue; only unrelated default-aux work is
eligible for the auxiliary queue.

One 64-call segment was an exact prefix of the final 457-call graph. Since the
runtime policy intentionally matches prefixes, retaining both rows would be
ambiguous. The generator therefore keeps the more specific final row and
conservatively leaves the shorter segment entirely on the primary queue.

| Graph calls | Prefix digest | Selected auxiliary calls |
|---:|---|---:|
| 32 | `cb49911bfca8ddec361dc86cba1798ebed5edd6fd0e31bfc6def93d5e7cde226` | 9 |
| 64 | omitted exact-prefix row | 0 |
| 128 | `2f4b50058814be595e9428d0a446e87789220ade12544c509f2dc896118860a7` | 51 |
| 256 | `45b888a1716f1901a825fd17c8b3bd901be562d05cf4a13523d2ba5b2a9e08c7` | 110 |
| 512 | `4b0461f2fcfaa9e0efee976bc7942e2c634bcb5920b354fdcc7f83fc5cefb65e` | 213 |
| 457 | `662af1d43ea112949b7f2b7baba44761c71035e290d6696584d05f8ccd8d0a8a` | 189 |

Every actual captured segment was checked against the finished policy and had
exactly one match, or zero for the deliberately primary-only 64-call segment.

## Replay and model correctness

The cut arm passed 20 consecutive B/A replay cycles. On every A replay:

- all 72 gate/up Q8 records and outputs were byte-exact;
- all 36 K Q8 records and outputs were byte-exact;
- all 72 Q/O Q8 records and outputs were byte-exact;
- all 36 live KV cache slices were byte-exact;
- full logits and token were exact against the first A invocation.

Against the fresh gate/up+K control, the final token was the same (`198`) and
full logits passed the established tolerance: maximum absolute difference
`0.083171`, mean absolute difference `0.014305`, and
`allclose(rtol=0.02, atol=0.5)`.

## Fresh synchronized R9 wall

All bookable rows below are non-profiled, synchronized, three-warmup R9 runs.
The service rate is the pp512 batch rate, not interactive decode throughput.

| Arm | Replay | Minimum | Median | pp512 service rate | Reading |
|---|---|---:|---:|---:|---|
| gate/up+K control, default ready | 20/20 exact | 70.173 ms | 70.353 ms | 7,296 tok/s | corrected authority |
| combined, fresh direct-dependency cut | 20/20 deep exact | 69.378 ms | 69.492 ms | 7,380 tok/s | **PASS** |
| combined, default ready | 19/20 exact | 68.965 ms | 69.113 ms | 7,424 tok/s | **NO_GO: recurrent drift** |
| combined, primary-only | 20/20 exact | 74.074 ms | 74.099 ms | 6,912 tok/s | exact, slower |
| combined, one queue | 20/20 exact | 74.075 ms | 74.116 ms | 6,912 tok/s | exact, slower |

The valid combined arm recovers `0.795 ms`, or `1.13%` of the corrected
gate/up+K wall. The faster default-ready number is not admissible because its
cycle 10 logits drifted by `0.096721` maximum absolute error.

The earlier `74.931 ms` default capture was produced with `PROFILE=1`; it is
retained only as capture/census evidence and is not a performance result.

## Boundary

This result proves that gate/up + K + Q/O can compose correctly and recover
whole-model prefill wall, provided the mutable K/Q/O stages are graph-owned and
the combined graph uses its own digest-bound Flash dependency cut. It does not
authorize Q6, an environment-only production switch, reuse of a stale graph
digest, or default ready placement without the cut.

No shared source file was changed for the arm, and nothing was promoted,
committed, or pushed.

## Evidence

- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/control-ready-r9.json`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/candidate-safe-cut-v2-deep20-r9.json`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/compare-safe-cut-v2-vs-gatek.json`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-default-v2.profile.jsonl`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-default-v2.census.jsonl`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-flash-direct-deps-cut-v2.json`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/ablation-default-ready-r9.json`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/ablation-primary-only-r9.json`
- `docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/ablation-one-queue-r9.json`
