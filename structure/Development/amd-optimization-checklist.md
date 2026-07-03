# AMD Decode Optimization Checklist

Date: 2026-06-12

Status: paused at a good stopping point. Local inference is consolidated; shared
storage is validated across 8B, 14B, and 32B as the recommended explicit
generated-policy storage mode.

Reference: `docs/gpu-performance-first-principles.md` is the canonical bytes/math/overhead
+ roofline guide -- diagnose the binding bucket (and measure it with counters) before
optimizing any kernel.

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
- [x] Extended the same rollout/comparator contract to 14B:
      `bench/qwen-rollout-20260612/14b-generated-small/summary.md`,
      `bench/qwen-rollout-20260612/14b-explicit-small/summary.md`, and
      `bench/qwen-rollout-20260612/compare-14b-small/report.md`. Both modes
      pass `75/75`, generate `644` tokens, and compare with quality delta `0`,
      regressions `0`, text changes `0/75`, and token changes `0/75`.
- [x] Added the top-level LLM runtime artifact contract:
      `extra/llm_runtime_contract.py` and
      `bench/llm-runtime-contract-20260613/`. The committed contract now
      validates eval, rollout, rollout-compare, training-data, and
      smoke-training artifacts with `8/8` rows passing and no missing
      artifacts.
- [x] Added the first training-data dry-run exporter:
      `extra/llm_training_data_probe.py` and
      `bench/qwen-rollout-20260612/training-data-v1/`. The current SFT-style
      probe exports `150` rows from the 8B and 14B generated rollouts with
      `0` filtered rows. This validates data shape and filtering only; it is
      not a training loop.
- [x] Added the smallest real Track 1 training/eval loop:
      `extra/llm_sft_smoke_train.py` and
      `bench/qwen-rollout-20260612/sft-smoke-v1/`. The tinygrad byte-context
      softmax probe trains on the rollout-derived SFT rows (`120` train,
      `30` eval), writes a small `model.npz`, and passes the held-out gate:
      eval loss `4.8483 -> 1.5290`, eval accuracy `0.0065 -> 0.6320`.
      This is a training-loop smoke test, not a Qwen adapter or LoRA stack.
- [x] Added the first real Qwen adapter V0:
      `extra/llm_adapter.py`, `extra/llm_adapter_train.py`, optional
      `--adapter` rollout loading, and artifacts under
      `bench/qwen-adapter-20260613/`. The V0 adapter is output-head LoRA only
      for Qwen3-8B (`rank=4`, `alpha=8`). Adapter tensors changed
      (`adapter_l2=0.003541`), the 75-prompt rollout still passes `75/75`, and
      base vs adapter comparison has `0` regressions, `0/75` text changes, and
      `0/75` token changes. The self-generated SFT rows are already
      near-zero-loss, so this validates adapter plumbing, not quality
      improvement.
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
      surface. Descriptor policy has since moved to BoltBeam (`boltbeam.policy.descriptor`);
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
- [x] Ran the repeated `QK_BLOCK_DOT` dominant-shape microbench:
      `extra/qk_block_dot_microbench.py`,
      `test/external/test_qk_block_dot_microbench.py`, and
      `bench/qk-block-dot-microbench-20260613/`. The full 8B Q4_K
      `ffn_gate` tensor rejects the lowering: v1 median `407.99` device Q4
      GB/s, `QK_BLOCK_DOT` median `285.01`, gain `-30.14%`, correctness pass.
      Verdict: `qk_block_dot_microbench_rejected`; no full decode, runtime
      integration, 14B/32B broadening, or policy promotion.
- [x] Ran the three-way packed-load diagnostic:
      `extra/qk_threeway_load_microbench.py`,
      `test/external/test_qk_threeway_load_microbench.py`, and
      `bench/qk-threeway-load-microbench-20260613/`. On the full 8B Q4_K
      `ffn_gate` tensor, v1 partial reaches `382.01` device Q4 GB/s. After the
      vector-lane reduction fix, schedulable `vector_load` passes correctness
      and reaches `349.25` (`-8.58%`). Opaque no-LOCAL `tile_custom` passes
      correctness but reaches only `36.99` device Q4 GB/s (`-90.32%`). Verdict:
      `wide_load_not_sufficient`; no full decode, runtime integration, or
      further wide-load-only retry.
- [x] Hardened the QK evidence/tooling edges after the bug audit:
      devectorizer tests now cover uint32 vec4 folding, scalar tails, unaligned
      fallback, and non-empty `VCAT`; Q6K storage reporting records requested
      versus effective storage mode; matrix/profile/eval parsers fail loudly on
      malformed artifacts; q8_1 vdot-parallel cannot be promoted to runtime
      policy; and three-way microbench repeats keep activation inputs fixed
      unless `--vary-seed` is requested.

## Open But Not Urgent

- [ ] Optional clean-room reproducibility check from a fresh clone, to catch
      untracked dependency regressions and artifact-portability bugs that normal
      local tests can miss.
- [ ] Optional cleanup of old home-directory scratch logs after confirming all
      relevant artifacts are committed under `bench/`.

## Practical Training / Eval Loop

- [x] Added the minimal Qwen output-LoRA adapter path:
      `extra/llm_adapter.py`, `extra/llm_adapter_train.py`, rollout
      `--adapter` support, and runtime-contract rows. V0 artifact:
      `bench/qwen-adapter-20260613/8b-output-lora-r4/`. This proves adapter
      install/save/load/rollout/compare plumbing, but the self-distilled data
      was saturated and produced no behavior change.
- [x] Added a real-signal held-out behavior-change gate:
      `extra/llm_adapter_signal_data.py` and
      `bench/qwen-adapter-20260613/training-data-v2/`. The base generated
      rollout fails the held-out sentinel exact-match gate (`0/12`), while the
      rank-8 output LoRA passes (`12/12`) with `+12` compare improvement and
      `0` regressions. Runtime contract now requires this improvement.
- [x] Replaced the synthetic sentinel target with a human-authored strict JSON
      answer gate:
      `extra/llm_adapter_json_data.py` and
      `bench/qwen-adapter-20260613/training-data-v3/`. The base generated
      rollout fails (`0/12`). Rank-16 output LoRA with EOS targets improves
      teacher-forced held-out token accuracy (`0.5000 -> 0.8542`) and generation
      reaches `3/12`, with `+3` compare improvement and `0` regressions.
      Verdict: failed promotion; output-only LoRA is insufficient for this
      conditional strict-JSON task.
- [x] Scoped a small non-output adapter target set for the same strict JSON
      train/rollout/compare gate. Do not keep tuning output-head-only LoRA as a
      promoted path.
- [x] Scoped and implemented the first non-output adapter target policy:
      `lastN_ffn` expands to exact dense FFN module paths and fails loudly on
      invalid groups. Internal adapters preserve activation gradients, and the
      adapter trainer has a plain-block path so internal params are visible to
      autograd.
- [x] Ran the first internal-adapter training diagnostic:
      `bench/qwen-adapter-20260613/internal-adapter-v4-diagnostic/`. Result:
      one-step baseline/no-REALIZE `last4_ffn` smoke passes, but full 8B
      internal-adapter training is blocked. Generated-QK training hits
      unsupported quant bit-op gradients; `REALIZE=1` OOMs at `23.78 GB`; the
      plain-block no-REALIZE workaround is too slow for full gates.
- [x] Built the dedicated suffix-cache internal-adapter training mode:
      `extra/llm_adapter_suffix_train.py` caches frozen prefix hidden states at
      the `lastN_ffn` boundary and trains the selected suffix only. V5
      `last1_ffn` rank-4 passes suffix parity (`max_abs=0.0`) and the training
      gate (`eval loss 7.4458 -> 0.2680`, eval token accuracy `0.5000 ->
      0.9167`) without generated-QK bit-op gradients or full fp16 realization.
      Held-out generation reaches only `4/12`; the apparent `+1` over V3
      output-LoRA is not meaningful at `N=12` and includes one regression. This
      is a training-path win, not a behavior-gate win.
- [x] Built the strict JSON V4 eval/objective foundation before more
      adapter-capacity sweeps. The held-out eval set is now `204` prompts, with
      deterministic JSON axes for parse/schema/type/value/no-extra-text and
      Wilson intervals. V4 free-generation rebaseline: base `0/204`, V3 output
      LoRA `69/204`, V5 suffix-cache `105/204`; V5 is the current-best behavior
      artifact.
- [x] Resumed Phase 4 after host reboot and verified the AMD path with the
      `DEV=AMD` smoke test (`[2, 3, 4]`). The V6 gold-control suffix adapter is
      now a real free-generation behavior control: `162/204` strict JSON passes
      on the V4 eval set versus V5's `105/204`, with `+57` strict passes,
      `59` improvements, and `2` regressions.
- [x] Ran V5-generated K=4 rejection sampling in bounded/resumable chunks.
      Artifact: `bench/qwen-adapter-20260613/training-data-v4-rs-v5-k4/`.
      Result: `216/1632` accepted attempts and `215` selected train rows, but
      coverage is not adequate for V7 because compiler has `0` selected rows
      and code/string are sparse.
- [x] Added category-focused RS continuation and an explicit coverage gate.
      `extra/llm_json_rejection_sample.py --sample-categories` appends samples
      only for weak categories while preserving full-artifact accounting, and
      `extra/llm_json_rs_coverage_gate.py` checks minimum selected rows per
      category.
- [x] Ran the stratified V5 RS continuation:
      `bench/qwen-adapter-20260613/training-data-v4-rs-v5-stratified-v1/`.
      Result: `257/2448` accepted attempts and `217` selected train rows; code
      reaches the `20`-row gate and string has `23`, but compiler remains `0`
      selected rows after `544` attempts and `158` near misses. Coverage gate:
      `fail`.
- [x] Audited compiler near misses:
      `bench/qwen-adapter-20260613/compiler-nearmiss-audit-v1/`. Verdict:
      `prompt_data_fix`. Compiler failures are mostly valid JSON wrong-value
      rows where V5 emits broad prefixes (`"train_qk"`, `"train"`) or stems
      without the row-specific numeric suffix. This is not a normalization
      fix; accepting prefixes would change the task contract.
- [x] Ran the Phase 4.2 compiler prompt/data fix:
      `extra/llm_adapter_json_data_v4_1_compiler.py` and
      `bench/qwen-adapter-20260613/training-data-v4_1-compiler/`. The new
      compiler-only dataset uses stable concept keys (`qk_gemv`) instead of
      row-specific suffix keys (`train_qk_gemv_005`), with prompt/template
      train/eval overlap `0` and intentional answer overlap `12`. V5 reaches
      `30/34` strict passes on the V4.1 compiler eval, and V5 RS produces
      `68/68` selected compiler train rows in
      `bench/qwen-adapter-20260613/training-data-v4_1-compiler-rs-v5-k4/`.
      The compiler coverage gate passes at min `20`.
- [x] Built the Phase 1/2 historical flywheel proof benchmark:
      `extra/qk_flywheel_dataset.py`, `extra/qk_flywheel_triage_eval.py`, and
      `bench/amd-decode-flywheel-proof-20260614/`. The dataset has `83`
      kernel-history rows, `45` train rows, and `38` family-split holdout rows.
      Best deterministic baseline is `mechanism_prior` /
      `simple_family_heuristic` at macro-F1 `0.185`. The strict no-adapter
      Qwen3-8B rollout scores macro-F1 `0.000` with `38/38`
      `invalid_output` rows, even after `/no_think`; it emits `<think>` tags
      and out-of-taxonomy reasons. Conclusion: `no_signal` for the current
      strict base model, so the full model-to-kernel flywheel is not proven.
- [ ] Continue Phase 4 by building a combined RS-SFT artifact: keep usable
      non-compiler rows from the original V4/stratified RS artifacts and
      replace the compiler slice with V4.1 stable-key compiler rows. Do not
      train V7 directly from either failed original RS artifact. Only promote
      future RS-SFT on V4/V4.1 free-generation strict JSON pass rate, Wilson
      intervals, regressions, and category balance; do not use teacher-forced
      token accuracy as the promotion signal.
- [ ] Continue flywheel proof only with a schema-capable model or adapter that
      can beat `mechanism_prior` on
      `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/`. If it
      cannot beat the deterministic baseline, keep the kernel loop and model
      loop separate except for the one-way benefit that faster kernels make
      model/eval experiments cheaper. Phase 3 is fully scoped in
      `docs/amd-decode-flywheel-proof-plan.md`: optional protocol diagnostic,
      strict SFT export from the `45` train rows, adapter rollout on the `38`
      holdout rows, and hard gates of `>=37/38` strict JSON outputs,
      macro-F1 above `0.185`, low false-positive accepts, and improved ranking.
      Phase 3.0/3.1 are now complete: protocol extraction still loses
      (`0.036` macro-F1, `0.763` false-positive accepts), and the SFT artifact
      has `45` train / `38` eval rows with `0` holdout ids in train. Phase 3.2
      is blocked by suffix-cache adapter training latency on the long
      kernel-context prompts; next step is progress reporting, prompt
      compression, or a smaller predeclared adapter smoke before a full rollout.
- [x] Added Phase 3.2A suffix-trainer instrumentation and ran the tiny local-8B
      adapter smoke:
      `bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0/`.
      The trainer now writes progress JSONL and has split-specific row caps.
      Smoke settings were `4` train rows, `2` eval rows, `8` steps. It reduced
      tiny-slice teacher-forced loss but did not move held-out generation:
      strict score stayed `0/38`, extracted macro-F1 stayed `0.036`, and
      false-positive accept rate stayed `0.763`. Measured prefix-cache latency:
      `4` train prefixes in `32.8s`, `2` eval prefixes in `21.0s`. This
      confirms the negative; do not rank-sweep local 8B before either prompt
      compression or a stronger-proposer benchmark.
- [x] Added Phase 3B learned cost-model triage:
      `extra/qk_flywheel_cost_model.py` and
      `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v0/`.
      The feature extractor uses only pre-result candidate/context fields and
      audits out raw ids, target labels, reasons, retry flags, evidence,
      result status, gains, GB/s, correctness decisions, source files, and
      split markers. It supports optional XGBoost through the native API and a
      no-dependency centroid fallback. Local XGBoost `3.2.0` ran with
      `rank:ndcg`, but still lost to `mechanism_prior`: macro-F1 `0.137`
      versus `0.185`, precision@3 `0.000` versus `0.083`, NDCG `0.189`
      versus `0.218`, with false-positive accept rate `0.000`. Conclusion:
      cost-model triage is the right tool class, but this `45`-row training
      set and current feature policy are not enough. Do not build ML from
      scratch; grow labeled outcomes and richer tinygrad/UOp/profile features
      before retrying.
- [x] Added Phase 3C cost-model feature/data audit:
      `extra/qk_flywheel_feature_audit.py` and
      `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v0/`.
      The audit records `needs_data_and_feature_expansion`: `24` unseen
      holdout categorical values, `56` weak rows, `9` post-full-decode train
      rows, and no target/result feature leakage. Next targets are explicit:
      label coverage for `construction_blocked`, `raw_accept_unconfirmed`, and
      `diagnostic_only`; normalize `18` `unknown` holdout mechanisms; add
      mechanism coverage for `packed_word_lane_unroll`, `qk_block_dot`,
      `vector_load`, and `wide_load_only`; and add first-class
      tinygrad/UOp/profile features for rows without structural kernel detail.
      Do not rerun or promote the cost model until these gaps are addressed.
- [x] Added Phase 3D cost-model schema v1:
      `extra/qk_flywheel_dataset_v1.py`,
      `test/external/test_qk_flywheel_phase3d.py`,
      `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1/`, and
      `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1/`.
      This preserves the `45` train / `38` family-split holdout rows while
      adding `candidate_outcome_v1`, normalized mechanisms, and leak-free v1
      feature groups. Unknown mechanisms drop to `0` after `26` mechanism
      normalizations. The audit improves but still does not prove the flywheel:
      unseen holdout categorical values fall `24 -> 15`, weak rows fall
      `56 -> 43`, and no target/result leakage is detected. Next flywheel work
      is targeted train coverage plus first-class tinygrad/UOp/profile
      features, not another immediate XGBoost rerun.
- [x] Added Phase 3E real source/compile feature extraction and coverage plan:
      `extra/qk_flywheel_feature_enrich.py`,
      `extra/qk_flywheel_coverage_plan.py`,
      `test/external/test_qk_flywheel_phase3e.py`,
      `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/`,
      `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured/`,
      and `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1/`.
      The featured dataset keeps the same `83` rows and same split, but adds
      real UOp/source features for `13` rows (`7` train, `6` holdout) from
      committed load-width and compile-gate artifacts. Leakage audit remains
      clean. This still does not authorize a cost-model rerun as a decision
      point: unseen holdout categorical values stay `15`, weak rows stay `43`,
      and the plan requires real targeted outcomes for seven uncovered
      mechanisms before rerunning XGBoost against `mechanism_prior`.
- [x] Added Phase 3F targeted real-outcome batch v1:
      `extra/qk_flywheel_targeted_outcomes.py`,
      `test/external/test_qk_flywheel_phase3f.py`,
      `bench/amd-decode-flywheel-proof-20260614/targeted-outcomes-v1/`,
      `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/`,
      `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured-plus/`,
      `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1-plus/`,
      and `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v1-plus/`.
      This converts unused committed real probe/source diagnostics into `38`
      new train rows without moving holdout rows or using design-only contracts
      as labels.

      Added coverage: `4` `vector_load`, `2` `wide_load_only`, `1`
      `qk_block_dot`, `5` `direct_output`, `8` `reduce_unroll`,
      `10` `row_upcast`, and `8` `two_dim_local` from existing
      in-train families. Labels added naturally: `19`
      `construction_blocked`, `5` `diagnostic_only`, `6`
      `raw_accept_unconfirmed`, `4` `reject`, and `4` `tie`.

      The plus run pushes holdout metrics to `macro-F1 0.821` (`xgboost` on
      `38` holdout rows) and preserves `rerun_phase3b_allowed=false` because
      mechanism coverage is still short by `13` rows: `5` `packed_word_lane_unroll`,
      `4` `qk_block_dot`, `1` `vector_load`, and `3` `wide_load_only`. No
      label rows are now required.

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
- [ ] Do not integrate the current `QK_BLOCK_DOT` lowering into runtime or run
      full decode; its repeated dominant-shape microbench failed the promotion
      bar.
- [ ] Do not retry the current C-style `QK_BLOCK_DOT` lowering as a runtime
      family; its repeated microbench regressed against v1.
- [ ] Do not continue the `vector_load` / raw `tile_custom` wide-load-only
      branch; the device-timed three-way diagnostic rejected it.
- [ ] Do not add another schedule/codegen family without an explicit
      memory-traffic mechanism and generated-source/load-width evidence.
- [ ] Do not move WMMA into the batch-1 decode track unless a source/counter
      artifact proves it is used by the reference decode path on gfx1100.
- [ ] Do not promote the strict JSON V3 output-LoRA result; it is a diagnostic
      negative (`3/12` held-out) rather than a training win.
- [ ] Do not rerun full 8B `last4_ffn` / `last1_ffn` through the current
      plain-block no-REALIZE trainer as the next gate; fix the training path
      first.

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
   `QK_BLOCK_DOT` compile gate passes, but the repeated full-shape microbench
   rejects it at `-30.14%` versus v1. The three-way packed-load diagnostic then
   rejects the cheap load-width-only branch too after fixing the construction
   bug: corrected `vector_load` is a `-8.58%` device-time regression and
   `tile_custom` is a `-90.32%` no-LOCAL control. Resume only with diagnosis of
   instruction mix / memory transactions / occupancy, or with a lower-level
   renderer/assembly-quality lowering. Any future microbench win starts as
   `raw_accept` and needs a confirmation rerun before promotion.

Default recommendation: pause here, then resume with practical training/eval
or the Ansor-style research track. Do not restart low-level kernel variants by
default.
