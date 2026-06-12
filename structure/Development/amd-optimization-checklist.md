# AMD Decode Optimization Checklist

Date: 2026-06-12

Status: paused at a good stopping point. Local inference is consolidated; shared
storage is validated as an opt-in path for 32B.

## Completed

- [x] Verified the original fp32-spill thesis was false: tinygrad already fuses
      Q4_K dequant into the GEMV and does not materialize fp32 weights.
- [x] Ran BEAM/generic scheduling tests and recorded that generic BEAM did not
      solve the gap; added safety guardrails for risky schedule search.
- [x] Probed expression-level vectorization in `gguf.py`; rejected it because
      codegen still emitted scalar byte loads.
- [x] Built and correctness-gated Q4_K/Q6_K primitive paths.
- [x] Ran kernel-boundary and greedy end-to-end output A/B checks.
- [x] Added generated-policy harness stages: search, policy, semantic report,
      parity, decode, A/B, profile, decision, report.
- [x] Committed harness source and matrix reproducibility tests so benchmark
      matrices regenerate from committed `decision.json` directories.
- [x] Accepted 8B generated policy as a modest win:
      `53.49` vs `49.35 tok/s`.
- [x] Accepted 14B generated policy as a strong win:
      `39.61` vs `22.76 tok/s`.
- [x] Proved 32B sidecar pressure was a storage architecture issue, not a search
      or correctness issue.
- [x] Added storage accounting, runtime caps, and tensor-scoped capped policy
      support.
- [x] Tested and rejected `QK_PRIMITIVE_STORAGE=q4_ondemand` as too slow.
- [x] Added `QK_PRIMITIVE_STORAGE=shared` using typed views over raw GGUF
      storage.
- [x] Validated shared storage on 8B smoke plus greedy A/B.
- [x] Validated full 32B shared-storage harness:
      `17.23` generated vs `11.15 tok/s` explicit, `54.56%` gain,
      `storage_bytes=0`, greedy A/B pass.
- [x] Regenerated current matrix at
      `bench/qk-shared-storage-20260612/matrix-summary.md`.
- [x] Fetched remote refs with `git fetch --all --prune`.

## Open But Not Urgent

- [ ] Full 8B shared-storage harness comparison against sidecar.
- [ ] Full 14B shared-storage harness comparison against sidecar.
- [ ] Decide whether shared storage remains 32B-only opt-in or becomes the
      generated-policy default.
- [ ] Optional clean-room reproducibility check from a fresh clone, to catch
      untracked dependency regressions that normal local tests can miss.
- [ ] Optional cleanup of old home-directory scratch logs after confirming all
      relevant artifacts are committed under `bench/`.

## Do Not Do Next

- [ ] Do not add more q8 arithmetic variants in `extra/`.
- [ ] Do not resume kernel search from the storage track.
- [ ] Do not hand-tune 32B as a standalone project.
- [ ] Do not make generated policies global defaults.
- [ ] Do not run BEAM/risky schedule search on Mac/TinyGPU/remote paths.

## Reasonable Resume Tracks

1. Practical track: use the speedup for a smallest-real training/eval run.
2. Infrastructure track: run full 8B/14B shared-storage harnesses and decide
   whether to promote shared storage.
3. Research track: continue the Ansor-style semantic packed-layout/codegen
   direction from `docs/amd-decode-ansor-direction.md`.

Default recommendation: pause here, then resume with either practical training
or shared-storage promotion checks. Do not restart low-level kernel variants by
default.
