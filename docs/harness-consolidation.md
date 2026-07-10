# Harness consolidation: the canonical harness + the dedup plan

Decision record + scoped plan from the 2026-07-03 harness-sprawl audit. Reinforces one rule:
**do not rebuild a measurement harness — route to the canonical entry.** Rebuild only when a
genuinely new measurement job exists (not a new *caller* of an existing job).

## The canonical harness (decided)

**`extra/qk/bench.py` is THE throughput entry** (committed `8a22fba05`). It is a thin dispatcher —
it measures nothing itself — to the two sanctioned authorities:

- **prefill** → `extra/qk/prefill_whole_synced.py :: prefill_authority()` — warmed TinyJit at a
  concrete start_pos, synced burst (`dev.synchronize` before+after), min-over-K → pure
  prefill-kernel tok/s. Never `model.generate` TTFT (understates prefill ~3×).
- **decode** → `extra/qk/decode_runtime_overhead.py` — clean W==D per-token wall (TinyJit replay +
  `.item()` readback, synced, NMEAS=40).

`DEV=AMD PYTHONPATH=. python extra/qk/bench.py --model <gguf> [--prefill|--decode]`. See memory
`prefill-bench-authority-not-ttft`. Per-kernel A/B harnesses should keep using their existing local
timing loops unless they are being actively consolidated; do not recreate the removed
`extra/qk/harness_contract.py` just to share `synchronize()+perf_counter()+median`.

**Prefill process policy** lives in `extra/qk/prefill_harness.py`. It owns the sanctioned
`authority` and `smoke` profiles, CSV parsing, subprocess env, and child argv construction.
`bench.py` and `prefill_whole_synced.py` both consume that module; the timing loop itself remains
inside `prefill_whole_synced.py`.

**Decode process policy** lives in `extra/qk/decode_harness.py`. It owns checkpoint contexts,
measurement count, max-context validation, subprocess env, and child argv construction.
`bench.py` and `decode_runtime_overhead.py` both consume that module; the W==D timing method itself
remains inside `decode_runtime_overhead.py`.

## Harness jobs, one owner each (target)

| Job | Canonical owner | Status |
|---|---|---|
| gate/probe → verdict → artifact → exit | `extra/qk/gate_registry.py` | DONE (consolidated) |
| whole-model throughput | `extra/qk/bench.py` → the two authorities | DONE |
| prefill process profile/env/argv | `extra/qk/prefill_harness.py` | DONE |
| decode process profile/env/argv | `extra/qk/decode_harness.py` | DONE |
| per-kernel timing loop | local harness loop / future explicit owner | open |
| eval/scoring (NLL + JSON) | `extra/llm/eval_common.py` + `json_scorer.py` | ok |
| provenance / repro-band | local to each live harness | ok (`extra/qk/harness_contract.py` removed) |
| GPU clock pinning | `extra/qk/clock_pin.py` | ok (2 idioms, justified) |
| model load + generate | `extra/llm/generate.load_model_and_tokenizer` | ok (loader); dead `generate_one` (item 3) |
| remote device orchestration | `extra/remote/*` | ok |
| AOT bundle / startup | `extra/qk/kernel_aot.py`, `startup_measure.py` | ok |
| JSON artifact IO | `probe_harness.probe_io` / `gate_registry` | ok (3 intentional conventions) |

## Dedup plan (ranked) — most items are DEFERRED behind active prefill work

> Coordination: as of 2026-07-03 another agent is actively editing the prefill/measurement
> surface on `master` (`bench.py`, `prefill_whole_synced`, `decode_runtime_overhead`,
> `model_authority_bench`, `generate.py`). Items that touch those files are
> **deferred** to avoid collision — execute once that work lands. Do them IN the canonical file,
> never in a new parallel module (that would re-fragment the wheel).

1. ~~Add `time_fn` beside `repro_band` in `harness_contract.py`~~ **WITHDRAWN (2026-07-10).**
   `extra/qk/harness_contract.py` has been removed. Do not restore a broad shared contract module
   only to satisfy stale benchmark imports or docs. If the remaining `*_ab`/`*_wd` timing loops are
   consolidated later, pick an explicit live owner and migrate callers in that same slice.
2. ~~Retire `model_e2e_bench.py`~~ **WITHDRAWN (2026-07-03, verified in-tree; refreshed 2026-07-10).**
   The supersession was inverted: `model_authority_bench.py` writes a **different** fixed-context authority
   artifact (`<id>.authority.json`), but has **zero importers/refs**
   — it was written as a successor and never adopted. `model_e2e_bench.py` is the LIVE tool: the README's
   current decode perf table is measured with it, and `llama_cpp_bench.py` merges llama numbers into its
   artifacts. They are also not clones — decode is measured differently (e2e = generate-window median;
   authority = fixed-ctx 128/512 W==D matched to llama depth). Consolidating means *adopting* the authority
   bench (migrate README + `llama_cpp_bench`, a methodology change), not deleting the live one. Left for an
   explicit owner decision, not a mechanical dedup.
3. **Delete dead `generate.generate_one` + `configure_process_env`** — zero importers; the callers
   its docstring names (`llm_rollout.py`, `llm_eval_harness.py`) no longer exist. **DEFERRED**
   (edits `generate.py`).
4. ~~Unify the 3 `llama-bench` wrappers~~ **DONE.** `extra/llm/llama_bench.py` owns the shared
   binary path, argv construction, json parse, and pp/tg row classification used by
   `llama_cpp_bench.py` and `model_authority_bench.py`.
5. ~~Merge the two `child_env` builders~~ **WITHDRAWN (2026-07-03, verified in-tree).** Same name, different
   job, and they share only `os.environ.copy()` (one line). `harness_contract.child_env(extra)` setdefaults
   DEV=AMD, uses PYTHONPATH=absolute ROOT, and adds QK_MODEL (launch a QK *eval* child).
   `generate.child_env(mode, *, device, storage, ...)` takes DEV from a required param, uses PYTHONPATH=".",
   clears `_CLEAR_KEYS`, and applies policy/storage/Q4K/Q6K flags (launch a *policy-mode rollout* child). A
   shared `_base_child_env()` that setdefaults DEV=AMD would be semantically wrong for the rollout builder
   (DEV comes from an explicit param there). Merging couples two unrelated launchers for one trivial line —
   the "intentional difference, don't consolidate" case (cf. the 3 JSON writers in "Not to do").
6. ~~`model_e2e_bench` redefines its own `_git` despite importing `harness_contract`~~
   **WITHDRAWN (2026-07-10).** The script no longer imports `harness_contract`; its small
   provenance helper remains local.

None are safe to do while the prefill agent holds those files; forcing them now trades a merge
collision (or a fragmented `time_fn`) for a marginal early landing. The decision above (canonical
`bench.py`) is the load-bearing outcome and is already shipped.

---

# Historical Execution Spec

The steps below are retained as history from the 2026-07-03 audit. Do not execute the
`harness_contract.py` steps as written: that file is gone, and restoring it is not the current prune
direction. Timing numbers are run-volatile, so any future migration still needs methodological
parity rather than byte-identical artifacts. All timing steps need the GPU.

## Step 1 — ~~add the shared `time_fn` to `harness_contract.py`~~ withdrawn

Withdrawn 2026-07-10. `extra/qk/harness_contract.py` is deleted; do not recreate it for this.

## Step 2 — migrate the clone sites (batch, run-verified)

17 files carried a `synchronize()+perf_counter()+statistics.median` loop. If this is revisited,
choose a live owner first and replace inline loops while **preserving that caller's semantics** (its
`n`, any warmup, and its unit).

Bench/A-B tier (do first — pure benches, no verdict artifact to break):
`decode_warp_flash_tile_ab` (this is the source of the de-facto `time_fn`; keep its re-export or
update `north_star`'s import), `north_star_flash_attn_tile_ab`, `decode_fused_flash_tile_ab`,
`fused_flash_concrete_gate_ab`, `fused_softmax_v_tail_ab`, `matmul_pv_diagnostic_ab`,
`ffn_gemv_warp_ab`, `ffn_gemv_warp_wd`, `proj_gemv_warp_wd`, `q4k_packed_gemv_wd`,
`prefix_cache_bench`.

Gate tier (OPTIONAL, lower priority — these were consolidated into `gate_registry` on 2026-07-03
and are stable; migrating their timing is nice-to-have, and it edits gate modules):
`decode_score_broadcast`, `decode_physical_tile`, `attention_reopen_gate`, `q6k_generated_coop_gate`,
`decode_hotloop_schedule_diff`.

**Do NOT touch `decode_runtime_overhead.py`** — its W==D loop is the decode *authority* (part of
`bench.py`'s contract); its methodology is load-bearing, not a clone to dedup.

Verify per file: run old vs new on the GPU, confirm the reported median is within the old
`spread_pct`. Commit in batches of ~5.

## Step 3 — delete dead `generate.py` scaffolding (reachability-verified)

`generate.generate_one` and `configure_process_env` have zero importers; the callers their docstring
names (`llm_rollout.py`, `llm_eval_harness.py`) no longer exist. Re-grep to confirm, then delete.
Keep `load_model_and_tokenizer` (imported by ~22 files) and `child_env` (see step 5).

## Step 4 — ~~retire `model_e2e_bench.py`~~ WITHDRAWN

Do not delete `model_e2e_bench.py`. On inspection (2026-07-03) the supersession claim is inverted:
`model_authority_bench.py` is the unadopted successor (zero refs, writes `<id>.authority.json`), while
`model_e2e_bench.py` is what the README perf table and `llama_cpp_bench.py` actually use. See dedup-plan
item 2 above. Consolidating is a real methodology decision (adopt the authority bench, migrate its two
consumers) — an owner call, not a mechanical dedup. Its private `_git` is intentionally local now that
`harness_contract` has been removed.

## Step 5 — the two `child_env` builders: NOT merged — dead twin deleted + survivor renamed

Do not merge them (they share only `os.environ.copy()` — same name, different job). But a deeper check
(2026-07-03) showed **both had zero callers**: `generate.child_env` + `policy_overrides` were dead
scaffolding from the removed rollout harnesses (same generation as `generate_one`), and
`harness_contract.child_env` had no call site either. Resolution:
  - Deleted the entire dead `generate.py` policy-env surface (`policy_overrides`, `child_env`, `_CLEAR_KEYS`,
    the `extra.qk.modes` import) — `generate.py` is now just `load_model_and_tokenizer`.
  - Renamed the documented survivor `harness_contract.child_env` → **`qk_subprocess_env`** (declarative;
    no same-named twin remains). This is the "if it's the same name, rename it better" fix.
  - Side-fix: Step 3's deletion had removed a `build_prompt_ids` re-export that `prefilled_route_parity`
    relied on; repointed that caller at its canonical home `extra.llm.eval_common`.

## Step 6 — ~~unify the 3 `llama-bench` wrappers~~ done

`extra/llm/llama_bench.py` now owns this shared job for the retained callers in this slice.

## Not to do

- Do not restore `harness_contract` just to host `time_fn`; choose a live owner if timing-loop
  consolidation becomes an active task.
- Do not "consolidate" the 3 JSON writers (`probe_io` / `gate_registry` / `eval_common.write_json`) —
  their newline/sort differences are intentional byte-parity constraints (each docstring explains).
- Do not touch `bench.py` / `prefill_whole_synced` / `decode_runtime_overhead` methodology — they are
  the authorities `bench.py` dispatches to.
