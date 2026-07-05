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
`prefill-bench-authority-not-ttft`. For a per-kernel A/B (not whole-model throughput) the shared
loop is `time_fn` (see plan item 1) — still do not clone a `synchronize()+perf_counter()+median`.

**Prefill process policy** lives in `extra/qk/prefill_harness.py`. It owns the sanctioned
`authority` and `smoke` profiles, CSV parsing, subprocess env, and child argv construction.
`bench.py` and `prefill_whole_synced.py` both consume that module; the timing loop itself remains
inside `prefill_whole_synced.py`.

## Harness jobs, one owner each (target)

| Job | Canonical owner | Status |
|---|---|---|
| gate/probe → verdict → artifact → exit | `extra/qk/gate_registry.py` | DONE (consolidated) |
| whole-model throughput | `extra/qk/bench.py` → the two authorities | DONE |
| prefill process profile/env/argv | `extra/qk/prefill_harness.py` | DONE |
| per-kernel timing loop | `harness_contract.time_fn` | DONE |
| eval/scoring (NLL + JSON) | `extra/llm/eval_common.py` + `json_scorer.py` | ok |
| provenance / repro-band | `extra/qk/harness_contract.py` | ok |
| GPU clock pinning | `extra/qk/clock_pin.py` | ok (2 idioms, justified) |
| model load + generate | `extra/llm/generate.load_model_and_tokenizer` | ok (loader); dead `generate_one` (item 3) |
| remote device orchestration | `extra/remote/*` | ok |
| AOT bundle / startup | `extra/qk/kernel_aot.py`, `startup_measure.py` | ok |
| JSON artifact IO | `probe_harness.probe_io` / `gate_registry` | ok (3 intentional conventions) |

## Dedup plan (ranked) — most items are DEFERRED behind active prefill work

> Coordination: as of 2026-07-03 another agent is actively editing the prefill/measurement
> surface on `master` (`bench.py`, `prefill_whole_synced`, `decode_runtime_overhead`,
> `model_authority_bench`, `generate.py`, `harness_contract`). Items that touch those files are
> **deferred** to avoid collision — execute once that work lands. Do them IN the canonical file,
> never in a new parallel module (that would re-fragment the wheel).

1. **Add `time_fn` beside `repro_band` in `harness_contract.py`** and route the ~18 `*_ab`/`*_wd`
   drivers + `prefix_cache_bench` through it. They each clone a `synchronize()+perf_counter()+
   `statistics.median` loop; `north_star_flash_attn_tile_ab` already imports `time_fn` from
   `decode_warp_flash_tile_ab` (proof the share is wanted). **DEFERRED** (edits `harness_contract`).
   Highest value: kills the biggest single clone.
2. ~~Retire `model_e2e_bench.py`~~ **WITHDRAWN (2026-07-03, verified in-tree).** The supersession was
   inverted: `model_authority_bench.py` *claims* to replace it ("Replaces the diagnostic end-to-end
   numbers") and writes a **different** artifact (`<id>.authority.json`), but has **zero importers/refs**
   — it was written as a successor and never adopted. `model_e2e_bench.py` is the LIVE tool: the README's
   current decode perf table is measured with it, and `llama_cpp_bench.py` merges llama numbers into its
   artifacts. They are also not clones — decode is measured differently (e2e = generate-window median;
   authority = fixed-ctx 128/512 W==D matched to llama depth). Consolidating means *adopting* the authority
   bench (migrate README + `llama_cpp_bench`, a methodology change), not deleting the live one. Left for an
   explicit owner decision, not a mechanical dedup.
3. **Delete dead `generate.generate_one` + `configure_process_env`** — zero importers; the callers
   its docstring names (`llm_rollout.py`, `llm_eval_harness.py`) no longer exist. **DEFERRED**
   (edits `generate.py`).
4. **Unify the 3 `llama-bench` wrappers** (`llama_cpp_bench`, `model_authority_bench.run_llama`,
   `llama_kv_ctx_slope_bench`) into one — each rebuilds the `llama-bench` argv + `-o json` parse and
   hardcodes the same binary path. **DEFERRED** (touches `model_authority_bench`).
5. ~~Merge the two `child_env` builders~~ **WITHDRAWN (2026-07-03, verified in-tree).** Same name, different
   job, and they share only `os.environ.copy()` (one line). `harness_contract.child_env(extra)` setdefaults
   DEV=AMD, uses PYTHONPATH=absolute ROOT, and adds QK_MODEL (launch a QK *eval* child).
   `generate.child_env(mode, *, device, storage, ...)` takes DEV from a required param, uses PYTHONPATH=".",
   clears `_CLEAR_KEYS`, and applies policy/storage/Q4K/Q6K flags (launch a *policy-mode rollout* child). A
   shared `_base_child_env()` that setdefaults DEV=AMD would be semantically wrong for the rollout builder
   (DEV comes from an explicit param there). Merging couples two unrelated launchers for one trivial line —
   the "intentional difference, don't consolidate" case (cf. the 3 JSON writers in "Not to do").
6. **`model_e2e_bench` redefines its own `_git`** despite importing `harness_contract` — use
   `harness_contract.provenance`. **DEFERRED** (bundled with item 2).

None are safe to do while the prefill agent holds those files; forcing them now trades a merge
collision (or a fragmented `time_fn`) for a marginal early landing. The decision above (canonical
`bench.py`) is the load-bearing outcome and is already shipped.

---

# Execution spec (hand-off to the agent already in these files)

Whoever owns `harness_contract.py` / `generate.py` should run this. Ordered; each step is one
`[test]` commit. Timing numbers are run-volatile, so **parity here is methodological, not
byte-identical**: after a migration the reported median must land within the pre-migration
`spread_pct` (same loop shape → same number modulo GPU jitter). All timing steps need the GPU.

## Step 1 — add the shared `time_fn` to `harness_contract.py` (functional, additive)

Put it directly beside `repro_band` (same file, so the timing loop and its stats live together).
Return the **sample list** (not a bare median) so it composes with `repro_band`. Keep the tinygrad
import **lazy** — `harness_contract` is imported before tinygrad on env-ordering-sensitive paths.

```python
def time_fn(fn, n:int=200, warmup:int=0, device:str="AMD") -> list[float]:
  """Per-call wall times (µs) for a synced GPU callable. Pair with repro_band() for the noise band,
  or statistics.median() for a point estimate. The ONE timing loop -- do not clone this."""
  from tinygrad import Device                      # lazy: keep this module importable pre-tinygrad
  dev = Device[device]
  for _ in range(warmup): fn(); dev.synchronize()
  dev.synchronize(); ts = []
  for _ in range(n):
    t0 = time.perf_counter(); fn(); dev.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
  return ts
```

This matches the de-facto shared shape (`decode_warp_flash_tile_ab.time_fn`, which
`north_star_flash_attn_tile_ab` already imports) except it returns the list and adds optional
`warmup`. Commit alone; nothing consumes it yet.

## Step 2 — migrate the clone sites (batch, run-verified)

17 files carry a `synchronize()+perf_counter()+statistics.median` loop. Migrate each to
`from extra.qk.harness_contract import time_fn` and replace its inline loop, **preserving that
caller's semantics** (its `n`, any warmup, and its unit — several report ms, `time_fn` is µs; keep
the caller's presentation by dividing, don't change the reported unit). A caller that wants a point
does `statistics.median(time_fn(f, n))`; one that wants the band does `repro_band(time_fn(f, n))`.

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
consumers) — an owner call, not a mechanical dedup. Its private `_git` still duplicates
`harness_contract.provenance`; fold that in only if/when the bench is touched for the adoption decision.

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

## Step 6 — unify the 3 `llama-bench` wrappers

`llama_cpp_bench.py`, `model_authority_bench.run_llama`, and `llama_kv_ctx_slope_bench.py` each build
a `llama-bench` argv + parse `-o json` and hardcode the same binary path under different constants.
Extract one `run_llama_bench(argv_extra, bin=...)` (in `eval_common` or a small `llama_bench.py`) and
route all three through it. Justified as a real shared job, not a new wheel.

## Not to do

- Do not add `time_fn` anywhere but `harness_contract` (a new timing module re-fragments the wheel).
- Do not "consolidate" the 3 JSON writers (`probe_io` / `gate_registry` / `eval_common.write_json`) —
  their newline/sort differences are intentional byte-parity constraints (each docstring explains).
- Do not touch `bench.py` / `prefill_whole_synced` / `decode_runtime_overhead` methodology — they are
  the authorities `bench.py` dispatches to.
