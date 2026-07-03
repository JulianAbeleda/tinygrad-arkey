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

## Ten harness jobs, one owner each (target)

| Job | Canonical owner | Status |
|---|---|---|
| gate/probe → verdict → artifact → exit | `extra/qk/gate_registry.py` | DONE (consolidated) |
| whole-model throughput | `extra/qk/bench.py` → the two authorities | DONE |
| per-kernel timing loop | `harness_contract.time_fn` (to add) | PLANNED (item 1) |
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
2. **Retire `model_e2e_bench.py`** — its own docstring says `model_authority_bench.py` "replaces"
   it; both write the same `bench/models/qwen/.../<id>.json`. **DEFERRED** (replacement is prefill turf).
3. **Delete dead `generate.generate_one` + `configure_process_env`** — zero importers; the callers
   its docstring names (`llm_rollout.py`, `llm_eval_harness.py`) no longer exist. **DEFERRED**
   (edits `generate.py`).
4. **Unify the 3 `llama-bench` wrappers** (`llama_cpp_bench`, `model_authority_bench.run_llama`,
   `llama_kv_ctx_slope_bench`) into one — each rebuilds the `llama-bench` argv + `-o json` parse and
   hardcodes the same binary path. **DEFERRED** (touches `model_authority_bench`).
5. **Merge the two `child_env` builders** (`harness_contract.child_env`, `generate.child_env`) into a
   base + two overrides. **DEFERRED** (both prefill turf).
6. **`model_e2e_bench` redefines its own `_git`** despite importing `harness_contract` — use
   `harness_contract.provenance`. **DEFERRED** (bundled with item 2).

None are safe to do while the prefill agent holds those files; forcing them now trades a merge
collision (or a fragmented `time_fn`) for a marginal early landing. The decision above (canonical
`bench.py`) is the load-bearing outcome and is already shipped.
