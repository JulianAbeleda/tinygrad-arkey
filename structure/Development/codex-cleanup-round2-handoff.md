# Codex Round-2 Cleanup — Handoff (executed)

Executes [`codex-cleanup-scope-round2.md`](codex-cleanup-scope-round2.md). Baseline
**246 passed**; final **239 passed** (= 246 − 7 intentionally-deleted block_dot
test methods, zero new failures). `git diff --check` clean. Every commit ran the
full `test/external/` suite green. One owning prefix per commit.

> Concurrency note: a parallel session (same git identity) committed an unrelated
> "hard-fork prune" task interleaved with these on `master`. Those commits are
> NOT part of this packet; the 7 SHAs below are this work. No file overlap; suite
> stayed green throughout.

## Phase A — medium-confidence probe deletions → **none deleted (correct)**

All four candidates are genuinely `import`ed by tests (not artifact-reads), so per
the packet rule ("a test that `import`s the module means LEAVE BOTH") all are KEPT.
Grep evidence (`git grep -n <mod> -- extra/ test/ tinygrad/`):

- `qk_batched_b0` ← `test_qk_batched_b0.py:3` imports `SEQ_LENS, _q4_bytes`
- `qk_packed_tile_consumption_probe` ← `test_qk_packed_tile.py:16` imports `_q4_words_and_x`
- `qk_packed_tile_lowering_analysis` ← `test_qk_packed_tile.py:8` imports `report_markdown, summarize_runs`
- `qk_packed_tile_closeout_diagnostic` ← `test_qk_packed_tile.py:9` (+ was in the deleted compile-gate test) imports `parse_debug7_log`

## Phase B — dead block_dot modules → **done** (`e68c681e6`, `[test]`)

Deleted `extra/qk_block_dot_compile_gate.py`, `extra/qk_block_dot_microbench.py`,
`test/external/test_qk_block_dot_compile_gate.py`, `test_qk_block_dot_microbench.py`
(~804 LOC). Round 1 had already stripped the shadow subprocess calls, leaving a
vestigial no-op block-dot loop in `qk_flywheel_shadow.run_outcomes` — removed it
(run_outcomes raises at the packed-load gate above it anyway).

- `git grep` for the module symbols across extra/ test/ tinygrad/ is empty;
  residual `qk_block_dot*` hits are committed-artifact replay rows in
  `qk_flywheel_targeted_outcomes.py` + string literals in test data (not imports).
- **`qk_semantic_op` verdict: KEPT.** It is self-contained — `QK_BLOCK_DOT` there
  is a local string constant, not the removed core op — depends only on
  `qk_packed_tile` (kept) and generates a design-contract artifact. Its test
  passes. Not dead → "else leave it".

## Phase C — byte-safe consolidations

| # | item | status | commit |
|--|--|--|--|
| 1 | llm_* json readers → llm_eval_common | **done** | `13458f2d1` |
| 2 | matrix/roofline/scorecard `_fmt`/`_last_runtime_storage`/`LLAMA_REFS` | **done (safe subset)** | `86ec43690` |
| 3 | opt-string + load-width parse rule | **done** | `8359a5570` |
| 4 | shadow `_read_jsonl` → llm_eval_common | **done** | `7261b8f03` |
| 5 | staged-shadow v0/v2 constants + CLI enum collapse | **SKIP — report** | — |
| 6 | q4k_bench device-metric driver | **partial (regexes only)** | `705e292ed` |
| 7 | Q4_K/Q6_K dequant → gguf.py authority | **done (delegation)** | `809b4e20e` |
| 8 | test fixtures → `_qk_testutil.py` | **SKIP — report** | — |
| 9 | `llm_json_rejection_sample` → `llm_generate` | **SKIP — empirically refuted** | — |
| 10 | dead majority/branch/ternary | **done** | `a0250a41d` |

### Done (each NFC, byte-proven)

- **C1** (`13458f2d1`): 3 `_load_json`/`_read_json` were byte-identical to
  `llm_eval_common.load_json` → aliased; matrix's bare copy aliased too (happy-path
  identical). Extracted one `read_id_jsonl` for the byte-identical id-validated core
  shared by 4 readers; each caller keeps its divergent wrapper (exists/no-rows/subpath).
  Proof: identical messages + suite 239.
- **C2** (`86ec43690`): canonical `_fmt(x, digits=2)` + `_last_runtime_storage` +
  `LLAMA_REFS` single-sourced in `qk_experiment_matrix` (leaf, no cycle); scorecard/gap
  `_load_json` → `llm_eval_common.load_json`. Roofline passes no explicit `digits`,
  so output is unchanged. **Left divergent (reported in-commit): `_git_commit`
  (try/except vs not), `_decision_path` (check-order + messages), the bare
  `_load_json` in matrix/roofline.**
- **C3** (`8359a5570`): `parse_load_width_words` + `parse_opts` (int-valued, cost_model
  key schema) added to `qk_flywheel_dataset` (leaf); callers cast/rename locally.
  Proof: `test_flywheel_dataset_golden` all 4 incl. `test_cost_model_centroid_output_is_pinned`.
- **C4** (`7261b8f03`): shadow `_read_jsonl` → `llm_eval_common.read_jsonl`; kept
  `_jsonl_bytes`/`_write_jsonl`/`_sha256` (freeze-hash source). Proof: phase4 11 pass.
- **C6 partial** (`705e292ed`): `DEVICE_RE`/`CORRECT_RE` (q4_k_bench stdout grammar,
  verbatim in 3 files) → new `extra/q4k_bench_metrics.py`; dropped unused `re`.
- **C7** (`809b4e20e`): `qk_layout.q4_k_reference`/`q6_k_reference` now delegate to
  `tinygrad/llm/gguf.ggml_data_to_tensor` (the live decode loader = the format
  authority); the duplicated dequant math + local `q_to_uint8` removed. Done as a
  pure `extra/` delegation, NOT the core refactor the packet sketched (gguf.py
  untouched → no risk). Proof: the pre-existing exact byte-pin tests
  `test_q*_k_reference_matches_current_gguf_expression` already pinned
  `q*_k_reference == ggml_data_to_tensor`; suite 239.
- **C10** (`a0250a41d`): `cost_model._majority` → `triage_eval._majority` (superset;
  cost_model never passes empty); removed a dead `if` branch in
  `feature_enrich._load_width_report_paths` (re-appended already-present paths the
  dedup dropped) and `status = "reject" if … else "reject"` → `"reject"` in
  targeted_outcomes; dropped unused `Counter`.

### Skipped — needs maintainer (with rationale)

- **C5** — *interface change, not a byte-NFC.* Collapsing the 14-way `step` enum to
  `freeze/run/score --batch <id>` changes the CLI contract `test_qk_flywheel_cli`
  asserts, and folding v0/v2 into `STAGED_BATCHES` would change `--batch` choices
  **and `pool_batches`** (which iterates the table). Do deliberately with CLI-test
  updates; the artifact-byte proof does not cover the interface.
- **C6 driver** — *divergent extractors.* `_classify` (g0 6 branches vs g0prime 4),
  `_measure`/`_run_candidate` (mode handling, status semantics, tail lengths) genuinely
  differ; merging is the wrong abstraction the principles forbid. Only the regexes
  were safe (done above).
- **C8** — *divergent row-builders.* The per-test `_row(...)` factories have different
  schemas/signatures (audit §F explicitly lists divergent row-builders as do-NOT-merge).
  The only shareable residue (`REPO = parents[2]` idiom ×31, a `read_jsonl`) is idiom,
  not knowledge-duplication; centralizing it is 31-file churn for no real payoff.
- **C9** — *empirically refuted; fix attempted and also refuted (not byte-safe).*
  Implemented the swap and tested with the packet's exact protocol: OLD (committed)
  vs NEW code, same fixed seed, 8B model + 8b-last1-ffn-suffix-lora-r4-v5 adapter +
  8b policy, on training-data-v4/sft.jsonl
  (`--limit-train-rows 1 --k 2 --temperatures 0.0 0.5`).
  - **Greedy (temp=0.0): byte-identical** — OLD==NEW==NEW2==OLD2.
  - **Sampling (temp=0.5): deterministically diverges** — OLD `[271]` vs
    NEW `[4913,9217,788,330,16,20,9207]`. Both are reproducible (OLD==OLD2,
    NEW==NEW2), so it is a real code-induced difference, not float-reduction noise.

  Per the user's request the fix was attempted: gave
  `llm_generate.load_model_and_tokenizer` a `seed:int|None` (skip the load-time
  `Tensor.manual_seed` when `None`; the two existing callers pass an int and are
  unchanged) and routed rejection_sample with `seed=None`. **The re-run produced the
  identical NEW tokens — i.e. the fix did NOT close the gap, disproving the
  load-time-seed hypothesis.** Two further hypotheses were tested and ruled out:
  *build order* (prompt_ids built before vs after the per-sample `manual_seed` —
  identical on a 1.7B fresh-process A/B) and *reduction non-determinism* (OLD is
  reproducible). So the divergence is a deterministic shift in the per-sample RNG
  trajectory that appears only in the full 8B+adapter sampling path and is not
  reproduced by isolated tests; its exact mechanism was not pinned.

  Since greedy is exact and `accepted.jsonl`/`sft.jsonl` were byte-identical in the
  test (the accepted row was the greedy sample), the pipeline is *functionally*
  equivalent — but raw temperature>0 samples change, so it fails the packet's strict
  "tokens must be identical" bar. **Reverted.** A maintainer who values the SSOT over
  byte-parity of stochastic samples could accept it deliberately; pinning the exact
  RNG mechanism (instrument the THREEFRY counter at each sample step in both paths)
  is the prerequisite for a parity-preserving version.

## Out of scope (untouched, as instructed)

Audit §D (`QKConfig`, BEAM globals), §E (adapter scaffold, dataset/verdict rewrites),
the divergent row/dataset builders, `bench/**` artifacts, the §F "already good" list.
