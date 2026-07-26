# ctx128 production integration handoff

## Scope

This branch is based on `feature/14b-decode-ctx128-and-depth-decay` at
`09c2da834` and contains only the two production-relevant safety fixes below.
No speculative G5 changes are included.

## Included commits

- `160198093` (cherry-pick of `f54f06e27`): scalarizes ctx128 mixed
  broadcast/distinct GLOBAL output-projection stores. It ADD-reduces only
  distinct groups, scalarizes exact broadcasts, and declines unsupported
  mixed groups and non-GLOBAL targets.
- `36901fa72` (cherry-pick of `54070758c`): fails closed when an explicit
  prefill model profile disagrees with the supplied model path.

## Deliberately excluded

- `88afc520c` is observability/tooling-only, but it changes artifact-write and
  lifecycle-output behavior. Keep it as a separate tooling promotion.
- G5/depth-decay changes remain excluded pending a non-speculative validation
  case and promotion decision.

## Known validation artifacts

- `test/unit/test_logits_only_reg_store.py` covers scalar and mixed
  broadcast/distinct output-projection lowering plus the GLOBAL-only boundary.
- `test/unit/test_prefill_harness.py` covers rejecting the 8B default when an
  explicit 14B profile is selected.
- `docs/14b-direct-packed-prefill-authority-baseline-20260710.md` records the
  existing 14B authority baseline and canonical explicit model/profile command.

## Not run here

- No unit tests, GPU runs, benchmarks, compilation probes, or model loads.
- CPU/static-only integration preparation; the existing validation artifacts
  above are not revalidated by this branch assembly.

## Remaining blocked work

- GPU ctx128 compile/render and end-to-end decode validation.
- Numerical and performance validation against the target 14B model.
- Evidence for any G5/depth-decay change.
