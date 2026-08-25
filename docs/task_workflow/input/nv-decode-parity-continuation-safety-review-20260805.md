# NV decode continuation: maintainability and safety review

Date: 2026-08-05. Scope: current intended generic core, renderer/spec,
shared-Q8/decode route, and research-harness diff. Excluded: user-owned
`docs/README.md`, `beating-llama*`, `what-makes-inference-fast*`,
`tinygrad/function.py`, scratch/raw artifacts, and binaries. No GPU was used.

## Result

One pre-merge default-off issue was found. No unsafe alias/lifetime widening,
renderer/spec failure, or evidence/JSON contradiction was found in the
reviewed material. The issue should be fixed before treating the current diff
as a strict closed-default implementation set.

## Finding: MMVQ hook executes an unconditional import on ordinary Q4 calls

`tinygrad/llm/decode_routes.py`, `_Q4KDecodeCandidate.execute`, imports
`tinygrad.llm.q4k_ffn_down_mmvq` before checking whether the linear has a
`Q4KFFNDownMMVQAdmission`. Normal loads never attach that admission, and the
callee immediately returns `None`, so functional route selection is closed.
However, every ordinary Q4 candidate invocation still takes the research-hook
import/check path (and the first ordinary invocation can initialize that
module). That is weaker than the intended “default unused” / byte-identical
hot-path discipline.

Required trivial remediation: check for the explicit admission attribute
first; only then import and invoke the research module. Preserve the exact
ordinary execution path when the attribute is absent. Re-run the existing
FFN-down unit test plus a default-no-admission regression test.

## Reviewed safeguards

* **Callify:** `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT` remains default 0.
  The owned-input match requires a unique precompiled output, matching
  dtype/span, read-only non-repeated parameter, and only reshape/semantic
  carriers. The P2b audit correctly remains `UNPROVEN` for the second copy
  class; no broad movement stripping was introduced.
* **PostBarrierRegion/codegen/renderers:** typed regions are rejected on
  renderers without explicit support; CUDA/PTX opt in. Validation checks the
  barrier anchor, lexical pairing, body dependency, and rejects internal
  barriers. The existing post-barrier test coverage passes.
* **Shared Q8:** Q6 direct output is selected only by the explicit,
  NV-sm120-validated `SharedQ8AttentionAdmission`; default is false. The
  Q6-direct record and JSON agree on semantic pass through g12 and settled
  `+7.00909375 us/token` wall no-go.
* **RMS scale-only lease:** exact shape is guarded at `(12288,4096)` and the
  model path requires a harness-installed resident norm-weight attribute.
  Normal loads create neither. The record/JSON agree: isolated `-6.42915 us`
  does not survive the one-layer wall gate (`+8.3694375 us/token`).
* **Evidence:** Q4 max17 incremental JSON/record agrees on
  `+12.462046875 us` beyond g12; the reconciled ledger books it once. Q6 and
  P2a/P2b records are consistently zero-credit closures.

## CPU validation

Focused CPU-only command completed successfully:

```text
89 passed in 2.96s
```

Coverage included post-barrier, projection/callify, shared-Q8 qualification,
multi-queue construction, Q4/Q6 substrate contracts, FFN-down MMVQ, and RMS
scale-only algebra/emitter tests. Pytest emitted only the repository's known
unknown-config warnings (`timeout`, `timeout_func_only`,
`timeout_method`); there were no test failures. `git diff --check` passed.

## Disposition

Do not promote, merge, or run a model timing from this review alone. Fix the
single MMVQ default-path hook first, then repeat the focused CPU suite. The
remaining research leases stay explicitly default-off and their measured
no-go records remain authoritative.
