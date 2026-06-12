# AMD Decode Optimization Checklist

Date: 2026-06-12

Status: paused at a good stopping point. Local inference is consolidated; shared
storage is validated across 8B, 14B, and 32B as the recommended explicit
generated-policy storage mode.

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
- [x] Validated full 8B shared-storage harness:
      `52.07` generated vs `50.41 tok/s` explicit, `3.31%` gain,
      `storage_bytes=0`, greedy A/B pass.
- [x] Validated full 14B shared-storage harness:
      `40.55` generated vs `21.77 tok/s` explicit, `86.29%` gain,
      `storage_bytes=0`, greedy A/B pass and profile complete.
- [x] Decided shared storage should be the recommended explicit generated-policy
      storage mode, while sidecar remains available and slightly faster for the
      old 8B peak artifact.
- [x] Regenerated current matrix at
      `bench/qk-shared-storage-20260612/matrix-summary.md`.
- [x] Fetched remote refs with `git fetch --all --prune`.
- [x] Added Track 1 smallest-real Qwen eval/rollout gate:
      `extra/qwen_eval_harness.py` compares explicit Q4/Q6 primitives against
      a pinned generated policy on fixed prompts with exact greedy-token parity.
- [x] Ran Track 1 on 8B shared storage:
      `bench/qwen-eval-20260612/8b-shared/README.md`, status `pass`,
      3/3 prompts exact token parity. This validates the faster inference path
      as a deterministic rollout/eval backend, not a tinygrad LLM training
      implementation.

## Open But Not Urgent

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

1. Practical track: decide whether to build a real SFT/RLVR training loop on
   top of the validated eval/rollout backend, or stop at inference/eval.
2. Infrastructure track: keep shared storage explicit and run occasional soak
   checks before making any runtime-default change.
3. Research track: continue the Ansor-style semantic packed-layout/codegen
   direction from `docs/amd-decode-ansor-direction.md`.

Default recommendation: pause here, then resume with practical training/eval
or the Ansor-style research track. Do not restart low-level kernel variants by
default.
