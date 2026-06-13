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
- [x] Added Track 1 generic LLM eval/parity gate:
      `extra/llm_eval_harness.py` compares explicit Q4/Q6 primitives against a
      pinned generated policy on fixed prompts with exact greedy-token parity
      and separate answer-quality scoring. `extra/qwen_eval_harness.py` is now a
      Qwen-default compatibility wrapper.
- [x] Added Track 1 generic LLM eval matrix source of truth:
      `extra/llm_eval_matrix.py`,
      `bench/qwen-eval-20260612/manifest.json`, and
      `bench/qwen-eval-20260612/matrix-summary.md`.
      `extra/qwen_eval_matrix.py` is now a Qwen-default compatibility wrapper.
- [x] Ran Track 1 matrix on 8B and 14B shared storage:
      both rows pass exact parity and score `10/10` on the small prompt suite.
      This validates the faster inference path as a deterministic rollout/eval
      backend, not a tinygrad LLM training implementation. The 32B row is
      recorded in the manifest but disabled as an optional heavy run.
- [x] Added generic dataset-style rollout runner:
      `extra/llm_rollout.py`, backed by shared scoring helpers in
      `extra/llm_eval_common.py`.
- [x] Ran Qwen 8B generated-policy rollout smoke:
      `bench/qwen-rollout-20260612/8b-generated/summary.md`, quality `pass`,
      `10/10`. This produces reusable JSONL completions for future eval,
      SFT/RLVR data generation, and compiler/search behavior gates.
- [x] Scaled Track 1 rollout to a 75-prompt Qwen 8B generated-policy dataset:
      `bench/qwen-rollout-20260612/dataset-small.jsonl` and
      `bench/qwen-rollout-20260612/8b-generated-small/summary.md`, quality
      `pass`, `75/75`, `608` generated tokens. Tests now enforce the dataset
      size/scoring rules and committed summary reproducibility.
- [x] Added Track 1 offline rollout comparator:
      `extra/llm_rollout_compare.py`,
      `bench/qwen-rollout-20260612/8b-explicit-small/summary.md`, and
      `bench/qwen-rollout-20260612/compare-8b-small/report.md`. Generated
      policy vs explicit primitive flags on the 75-prompt 8B dataset has
      quality delta `0`, regressions `0`, text changes `0/75`, and token
      changes `0/75`.
- [x] Added the Ansor-transition foundation for the llama.cpp-comparable goal:
      `extra/qk_llama_scorecard.py`, `extra/qk_gap_profile.py`, and
      `extra/qk_semantic_descriptor.py`, with committed artifacts under
      `bench/qk-ansor-transition-20260612/`. Current scorecard:
      8B `51.46%`, 14B `61.63%`, 32B `55.94%` of llama.cpp; all are below the
      first `70%` comparable-speed target.
- [x] Closed the shared 8B DEBUG=2 profile gap and regenerated the
      Ansor-transition gap profile. 8B/14B/32B are all profiled, and named
      attribution still points at QK semantic schedule/codegen as the next
      bottleneck.
- [x] Added descriptor policy reproduction and the candidate/search-loop v0
      surface: `extra/qk_descriptor_policy.py`,
      `extra/qk_candidate_generator.py`, `extra/qk_candidate_static_gate.py`,
      and `extra/qk_ansor_transition_loop.py`. Descriptors reproduce accepted
      runtime policy semantics with zero diff; the bounded candidate sets are
      8B `19`, 14B `27`, 32B `32`; all static gates pass; each model has
      `current` plus six `benchmark_next` policy files under
      `bench/qk-ansor-transition-20260612/search/`.
- [x] Benchmarked loop-v0 `benchmark_next` policies policy-vs-policy through
      the QK harness: `bench/qk-ansor-transition-20260612/benchmarks/`. 8B and
      14B had no accepts. 32B had one raw `+3.24%` accept, but the confirmation
      rerun was a tie at `-2.29%`, so no candidate is promoted. Verdict:
      descriptor-level `parts`/`LOCAL` knob search is exhausted.
- [x] Generated and gated semantic schedule v0 candidates:
      `extra/qk_semantic_schedule.py`,
      `extra/qk_semantic_schedule_bench.py`, and
      `extra/qk_semantic_schedule_verdict.py`. The first richer schedule surface
      (`direct_out`, `row_upcast2`, `reduce_unroll4`, `two_dim_local4`) passed
      static gates and found isolated attention microbench wins, but full decode
      rejected the supported winner on both targets: 8B `-10.28%`, 14B
      `-5.21%`. Verdict: `semantic_schedule_v0_rejected`; 32B skipped by rule.
- [x] Generated and gated semantic codegen v1 candidates:
      `extra/qk_semantic_codegen.py`,
      `extra/qk_semantic_codegen_verdict.py`, runtime support for
      `q4_k_packed_u32_direct`, and the shared semantic microbench runner. The
      direct-output Q4 family is now full-decode installable as an exact-tensor
      generated-policy override, but the 8B/14B microbench gate produced no
      accepts: 8B `0` accepts (`2` ties, `1` reject), 14B `0` accepts (`2`
      ties, `2` rejects). Verdict: `semantic_codegen_v1_rejected`; full decode
      and 32B skipped by rule.
- [x] Hardened semantic schedule/codegen gates before adding another family:
      microbench wins are now `raw_accept` only, verdict tools require a
      matching full-decode confirmation rerun before promotion, and semantic
      candidate artifacts carry explicit storage deltas plus correctness
      provenance. CPU/Mac tests cover reference unpacking; AMD microbench runs
      still cover GEMV numerics; full-decode A/B covers model assembly.
- [x] Scoped and tested semantic codegen v2 / Family B:
      `docs/amd-decode-semantic-family-b.md`,
      `extra/qk_semantic_codegen_v2.py`, and
      `bench/qk-ansor-transition-20260612/semantic-codegen-v2/`. The bounded
      row-grouped Q4_K `ffn_down` surface rejected on the 8B/14B microbench
      gate: 8B `-31.03%` / `-71.54%`, 14B `-52.59%` / invalid illegal opt.
      Verdict: `semantic_codegen_v2_rejected`; no runtime install, full-decode
      run, or 32B run is justified.
- [x] Fixed loop-v0 matrix portability: benchmark matrix `path` and `policy`
      fields are repo-relative, tests assert they are not absolute paths, and
      the regenerated matrices reproduce outside `/home/ubuntu/tinygrad-arkey`.
- [x] Resolved the `structure/` tracking policy: `.gitignore` ignores the tree
      by default while explicitly allowing the tracked session handoff and AMD
      optimization checklist files.
- [x] Added the model-scope QK bandwidth roofline:
      `extra/qk_bandwidth_roofline.py`,
      `bench/qk-bandwidth-roofline-20260613/roofline.md`, and
      `docs/amd-decode-bandwidth-roofline.md`. By logical full-GGUF bytes,
      tinygrad generated reaches `27-38%` of the RX 7900 XTX peak while
      llama.cpp reaches `53-63%`; this supports treating the remaining gap as
      memory-load efficiency.
- [x] Scoped the next compiler-research surface as packed-load lowering:
      `docs/amd-decode-packed-load-lowering.md`, with prior-art framing in
      `docs/amd-decode-prior-art.md`.
- [x] Implemented and gated semantic codegen v3 / Family C v0:
      `extra/qk_semantic_codegen_v3.py`,
      `extra/qk_load_width_report.py`, and
      `bench/qk-ansor-transition-20260612/semantic-codegen-v3/`. The packed-load
      Q4_K `ffn_gate` rewrite tied on both target models: 8B `-0.65%`, 14B
      `-0.31%`; DEBUG=4 parsing found scalar `u32` loads and no vector-load
      evidence. Verdict: `semantic_codegen_v3_rejected`; no full decode or 32B.
- [x] Implemented and gated semantic codegen v4 / Family C v1:
      core AMD lowering can preserve raw aligned `uint32x4` load/store, but the
      real Q4_K GEMV candidate cannot yet consume the loaded vector through
      current tensor/UOp shape rules. Verdict: `semantic_codegen_v4_rejected`
      at construction; no benchmark, full decode, or 32B run.
- [x] Scoped and added the next representation layer:
      `docs/amd-decode-packed-qk-tile-design.md` and `extra/qk_packed_tile.py`.
      Family C v4 artifacts now record the `PackedQKTile` and legal
      `u32x4_aligned` load tile metadata instead of treating vector load as
      prose. This is a static IR/provenance step, not a performance claim.
- [x] Ran the `PackedQKTile` consumption construction probe:
      `extra/qk_packed_tile_consumption_probe.py` and
      `bench/qk-packed-tile-consumption-20260613/`. Normal UOps cannot consume
      `uint32x4` Q4_K loads (`GEP` verifier failure and vector-arith shape
      failure), while a custom semantic kernel passes exactly and DEBUG=4
      source parsing confirms `vector_u32x4`. Verdict:
      `semantic_custom_op_required`; no microbench/full-decode run.
- [x] Implemented the first real custom semantic Q4_K tile lowering:
      `q4k_gemv_tile_custom_partial_kernel` consumes packed Q4_K payload words
      with `tg_uint4`, keeps fp16 activations, supports `parts=1` and `parts=4`,
      and passes AMD correctness. Artifact:
      `bench/qk-packed-tile-lowering-20260613/`. Microbench is weak-positive
      only (`+7.20%` `ffn_gate`, `+5.83%` `attn_output`), below the `>=10%`
      full-decode promotion bar. Verdict:
      `semantic_custom_lowering_constructed_but_not_promoted`.
- [x] Ran repeated packed-tile lowering analysis:
      `extra/qk_packed_tile_lowering_analysis.py` and
      `bench/qk-packed-tile-lowering-analysis-20260613/`. Source-shape parsing
      confirms v1 `u32_scalar` versus `tile_custom` `vector_u32x4`, but repeated
      8B Q4_K microbench does not generalize: gain range `-2.04%` to `+7.51%`,
      median `-0.36%`; only `ffn_up` is materially positive. Verdict:
      `diagnose_only_not_promoted`; no full decode or runtime integration.
- [x] Closed out the raw packed-tile custom path with DEBUG=7 disassembly:
      `extra/qk_packed_tile_closeout_diagnostic.py` and
      `bench/qk-packed-tile-research-closeout-20260613/`. `tile_custom` emits
      real target `global_load_b128` instructions, but only by giving up v1's
      32-lane scheduled shape for a workgroup-size-1 opaque custom kernel with a
      much larger target body. Verdict:
      `raw_custom_tile_path_closed_not_promoted`.
- [x] Defined the first packed-QK semantic op contract:
      `docs/amd-decode-packed-qk-semantic-op.md`,
      `extra/qk_semantic_op.py`, and
      `bench/qk-packed-semantic-op-20260613/`. `QK_BLOCK_DOT` is scoped to one
      Q4_K packed block dot, leaves row/K/split scheduling visible, records
      eight 8B/14B Q4_K contract rows, and makes no runtime or speed claim.
- [x] Implemented the minimal `QK_BLOCK_DOT` compile gate:
      `extra/qk_block_dot_compile_gate.py`,
      `test/external/test_qk_block_dot_compile_gate.py`, and
      `bench/qk-block-dot-compile-gate-20260613/`. The core
      `Ops.QK_BLOCK_DOT` lowering keeps the v1 32-lane scheduled shape, passes
      AMD GEMV correctness for the fixed 8B Q4_K `ffn_gate` shape, and emits
      target `global_load_b128` evidence (`5` vs `1` for v1). Verdict:
      `qk_block_dot_compile_gate_passed_compile_shape`. This authorizes a
      repeated dominant-shape microbench only; no runtime integration or full
      decode yet.

## Open But Not Urgent

- [ ] Optional clean-room reproducibility check from a fresh clone, to catch
      untracked dependency regressions and artifact-portability bugs that normal
      local tests can miss.
- [ ] Optional cleanup of old home-directory scratch logs after confirming all
      relevant artifacts are committed under `bench/`.

## Do Not Do Next

- [ ] Do not add more q8 arithmetic variants in `extra/`.
- [ ] Do not resume kernel search from the storage track.
- [ ] Do not hand-tune 32B as a standalone project.
- [ ] Do not make generated policies global defaults.
- [ ] Do not run BEAM/risky schedule search on Mac/TinyGPU/remote paths.
- [ ] Do not promote semantic schedule/codegen `raw_accept` rows without a
      matching confirmation rerun.
- [ ] Do not broaden the Family B row-group surface to more roles or 32B; it
      failed on the targeted Q4_K `ffn_down` mechanism.
- [ ] Do not broaden the Family C v0 packed-word-lane rewrite; it tied on
      8B/14B and did not produce vector-load evidence.
- [ ] Do not add another Family C variant until it consumes `PackedQKTile` or a
      successor semantic op; raw vector load/store support alone is not enough.
- [ ] Do not run vector-load Q4_K microbench/full-decode gates until a semantic
      packed QK load/decode/dot lowering exists.
- [ ] Do not add more raw `Ops.CUSTOM` `tg_uint4` Q4_K variants; DEBUG=7
      close-out already explains why vector source loads alone are insufficient.
- [ ] Do not implement full-GEMV semantic hiding. The next implementation must
      keep `QK_BLOCK_DOT` block-local so row/K/split axes remain schedulable.
- [ ] Do not integrate `QK_BLOCK_DOT` into runtime or run full decode until its
      repeated dominant-shape microbench clears the promotion bar.
- [ ] Do not add another schedule/codegen family without an explicit
      memory-traffic mechanism and generated-source/load-width evidence.
- [ ] Do not move WMMA into the batch-1 decode track unless a source/counter
      artifact proves it is used by the reference decode path on gfx1100.

## Reasonable Resume Tracks

1. Practical track: build a real SFT/RLVR loop or a richer judge on top of the
   validated rollout/comparator backend.
2. Infrastructure track: keep shared storage explicit and run occasional soak
   checks before making any runtime-default change.
3. Research track: continue the Ansor-style semantic packed-layout/codegen
   direction from `docs/amd-decode-ansor-direction.md`, starting from the
   rejected semantic schedule/codegen surfaces in
   `bench/qk-ansor-transition-20260612/`. The current negative bound now covers
   descriptor `parts`/`LOCAL`, schedule v0, direct-output v1, and row-grouped
   Family B v2. Family C v0 then tested the first packed-load rewrite and tied;
   Family C v1 proved raw `uint32x4` loads lower but cannot yet be consumed by
   the real GEMV graph. The packed-tile consumption probe then showed normal
   UOps cannot consume the tile but a custom semantic kernel can. The minimal
   `QK_BLOCK_DOT` compile gate now passes. Resume by running a repeated
   dominant-shape microbench for the fixed 8B Q4_K `ffn_gate` shape. Any future
   microbench win starts as `raw_accept` and needs a confirmation rerun before
   promotion.

Default recommendation: pause here, then resume with practical training/eval
or the Ansor-style research track. Do not restart low-level kernel variants by
default.
