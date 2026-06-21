# Phase 2 result: server/long-prompt `PREFILL_CONCRETE_KV` policy

Date: 2026-06-20. Scope: `docs/prefill-policy-integration-scope-20260620.md` Phase 2. gfx1100 RX 7900 XTX 24GB,
Qwen3-8B. Harness: `extra/qk_prefill_concrete_kv_policy_probe.py` → `bench/qk-prefill-policy-integration/concrete_kv_policy.json`.

## What shipped (model.py)

- `PREFILL_CONCRETE_KV=auto` (new) + `PREFILL_SERVER_PROFILE=1` (new convenience switch).
- `prefill_concrete_kv_auto_decision(server_profile, prefill_v2_on)`: ON iff `PREFILL_V2` active AND server profile.
  Precompile only pays across repeated/long generation, which can't be detected at load → the auto signal is the
  explicit server profile (one-shot short prompts must leave it off; the precompile load tax loses — measured in
  `docs/prefill-default-policy-evaluation-result-20260620.md`).
- `PREFILL_SERVER_PROFILE=1` implies `PREFILL_V2=auto` (when V2 unset) + concrete-KV on (when V2 ends up on) — one
  switch for the server/long-prompt profile. Explicit `PREFILL_V2=0/1` and `PREFILL_CONCRETE_KV=0/1` always win.
- Resolved in `from_gguf` before `Transformer()` (so `precompile_concrete_prefill_jits()` and `generate()` see it).

## Evidence (all gates PASS)

| case (env) | resolved V2 | resolved CKV | precompiled jits |
|---|---|---|---:|
| default (unset) | off | off | 0 |
| `V2=1` (one-shot) | on | **off** | 0 |
| `V2=1 CKV=auto` (no server) | on | **off** | 0 |
| `V2=1 CKV=1` (explicit) | on | on | 2 |
| **`SERVER_PROFILE=1`** | **on** (VRAM-auto) | **on** | **2** |
| `V2=0 SERVER_PROFILE=1` | off (explicit wins) | off | 0 |

Unit decisions: v2on+server→ON ✓, v2on+noserver→OFF ✓, v2off→OFF ✓. `all_gates_pass`.

## Policy

- **`PREFILL_SERVER_PROFILE=1`** is the recommended single switch for servers / repeated / long prompts: it
  VRAM-resolves `PREFILL_V2` and turns on concrete-KV precompile → the best warm prefill (**0.17–1.6 s**, vs
  7.5–73.5 s default; see Phase-0 policy eval), at a one-time load precompile cost (`ceil(max_context/512)` jits).
- **One-shot / short-prompt CLI: do NOT set it** — concrete-KV auto stays off without the profile, so the default
  cold-one-shot path isn't penalized by precompile. Users who know they want it can still force `PREFILL_CONCRETE_KV=1`.
- Default behavior unchanged (off) unless the profile/flag is set.

Reproduce: `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_concrete_kv_policy_probe.py`
