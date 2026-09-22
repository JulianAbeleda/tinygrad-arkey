# Native NV d512 pre-dispatch breakdown record

Date: 2026-08-05. Route: `DEV=NV`; Qwen3-8B-Q4_K_M; depth 512;
RTX 5090 / driver 595.84. Status: **pre-first-graph host boundary causally
reconciled.**

## Result

The prior host ledger exposed 247.557 us before the first native graph call.
Nine alternating marker-off/instrumented observations reproduce that boundary
at 256.997 us median (11.983 us MAD). The row-wise component sum is 255.634 us,
leaving 1.493 us or 0.582% unexplained. This is inside both requested closure
limits (`25 us` and `5%`). The instrumentation changed wall by +25.719 us
(5632.909 versus 5607.190 us) and did not change the observed token.

| pre-first-HCQGraph component | median us |
| --- | ---: |
| generator/model work before `TinyJit.__call__` | 35.637 |
| `_prepare_jit_inputs` | 77.927 |
| reuse name/expected-info checks | 2.966 |
| concrete input copy/rebind | 121.380 |
| captured-call/run-linear handoff | 0.742 |
| `run_linear` entry to graph lookup | 3.246 |
| settled graph-cache lookup | 1.112 |
| lookup return to first `HCQGraph.__call__` entry | 2.605 |

The two largest terms account for 199.307 us, 77.6% of the observed boundary.

## What those two terms are

`_prepare_jit_inputs` is not one-time compilation. It reruns on every settled
token. Of its 77.927 us, 49.284 us rebuilds the structural signature using UOp
substitution, graph rewrite, and unbinding; 15.479 us rediscovers/validates the
two tensor inputs; and 6.312 us rebuilds variable and expected-input metadata.
Only the subsequent 2.966 us equality checks are the cheap cache-reuse test.

`CapturedJit` receives two concrete inputs and classifies one as written. Its
generic alias-safety path calls `_copy_input` for that feedback token. Building
the fresh buffer/copy UOp costs 16.892 us and executing the generic copy through
`run_linear` costs 99.088 us. The mutation is semantically required today:
the capture writes one input and therefore cannot safely bind the caller's
allocation in place. The measured fresh allocation plus generic eager-copy
mechanism is an implementation choice, not an established irreducible cost.

This work is per-token Python/runtime work. It is distinct from one-time JIT
capture, graph construction, compilation, and memory planning, none of which
occurred in these settled replay observations.

## HCQ work is on the other side of the boundary

The first graph's host call takes 196.954 us across all five graph submissions.
Within that total, node-variable updates cost 58.460 us, queue submission
89.248 us, wait/restore 15.920 us, signal finalization 17.443 us, variable-dict
construction 7.002 us, and buffer mapping 3.277 us. These intervals begin at
the recorded `HCQGraph.__call__` entry and overlap already submitted GPU work
after the first group. They are therefore diagnostic optimization targets, but
must not be added to the pre-first boundary or the outside-device ledger.

## Cheapest decisive optimization experiment

First test a default-off settled-replay input-binding cache, not a production
route change:

1. Cache the structural expected-input descriptor for the captured JIT and
   admit a fast path only when names, base shape/stride/dtype/device, and bound
   variable schema are identical. Preserve the current full reconstruction as
   the fallback and periodic oracle.
2. Separately preallocate the one alias-safe feedback shadow and reuse its
   allocation while preserving the copy and dependency edge. Do not skip the
   copy: a no-copy arm would violate the measured written-input contract.
3. Use reverse-bracket marker-off A/B/A, require the same token and exact graph
   census, and reject either experiment unless wall moves by at least 25 us
   with the full-contract oracle agreeing.

The signature cache is the cheapest low-risk first experiment because its
49.284 us target is pure repeated host reconstruction. The reusable feedback
shadow has a larger 121.380 us ceiling but needs stronger lifetime and
dependency proofs.

## Artifacts

The diagnostic is
`scratchpad/nv_decode_predispatch_breakdown.py`; its hermetic tests are
`test/unit/test_nv_decode_predispatch_breakdown.py` (3 passed). The compact
payload is
`docs/task_workflow/output/nv-decode-native-d512-predispatch-breakdown-20260805.json`.
Raw evidence is `/tmp/nv_decode_predispatch_breakdown_v3_20260805.json`, SHA-256
`b0d23e644e48f6f334da3f35545973e8e451dac829b378e8c2ff78c6a1c21ee7`.

No production/default file was changed, and nothing was committed or pushed.
