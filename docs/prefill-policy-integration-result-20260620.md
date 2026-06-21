# Prefill Policy Integration — FINAL REPORT

Date: 2026-06-20. Scope: `docs/prefill-policy-integration-scope-20260620.md`. gfx1100 RX 7900 XTX 24GB, Qwen3-8B-Q4_K_M.
Prefill is solved at the kernel level; this delivers the integration/policy layer (no new kernels).

## 1. VRAM-aware `PREFILL_V2` auto-policy (Phase 1)
`PREFILL_V2=auto` resolves on/off from detected VRAM in `from_gguf` before `Transformer()`. `prefill_v2_auto_decision`:
enable iff total VRAM ≥ 23GB floor AND `Q4 + fp16_covered + KV + 3GB margin` fits; None VRAM / small card → OFF.
Evidence (`bench/qk-prefill-policy-integration/prefill_v2_auto_policy.json`, all gates PASS): 24GB→ON (est fp16
covered 13.9GB matches measured +14GB), 16GB→OFF, unknown→OFF, explicit 0/1 force, unset→off. Detail:
`docs/prefill-v2-auto-policy-result-20260620.md`.

## 2. Concrete-KV server/long-prompt policy (Phase 2)
`PREFILL_CONCRETE_KV=auto` + `PREFILL_SERVER_PROFILE=1`. `prefill_concrete_kv_auto_decision`: ON iff PREFILL_V2
active AND server profile (precompile pays only across repeated/long generation, undetectable at load → the
explicit profile is the signal). `PREFILL_SERVER_PROFILE=1` = one switch → V2=auto + concrete-KV on. Evidence
(`concrete_kv_policy.json`, all gates PASS): server_profile→V2 on + CKV on + 2 precompiled jits; one-shot/no-profile→
CKV off; explicit forces. Detail: `docs/prefill-concrete-kv-policy-result-20260620.md`.

## 3 & 4. Route-schedule probe + the 32-token fallback fix (Phase 3)
`PREFILL_REMAINDER_FIX` (default-on under PREFILL_V2): a sub-512 prompt remainder (fresh tail OR prefix-cache
resume) is routed through ONE prefill-v2 chunk shifted back to end at `prompt_len` (all-real tokens, byte-identical),
instead of many 32-token symbolic calls. Before/after (`route_schedule_probe.json`, all gates PASS):

| prompt | scenario | sched OFF → ON | prefill | tok0 |
|---:|---|---|---|:--:|
| 600 | fresh remainder | int512+32tok×3 → int512+sym512 | 23.6→9.0s (2.6×) | ✓ |
| 1024 | prefix-cache resume | 32tok×14 → sym512 | 6.6→3.0s (2.2×) | ✓ |
| 1500 | prefix-cache resume | 32tok×15 → sym512 | 7.7→0.55s (**14.0×**) | ✓ |
| 2100 | full + remainder | sym512+32tok×3 → sym512×2 | 5.2→1.4s (3.8×) | ✓ |

Detail: `docs/prefill-route-schedule-result-20260620.md`.

## 5. TTFT / load / VRAM table (from the Phase-0 policy eval, `docs/prefill-default-policy-evaluation-result-20260620.md`)

| mode | prompt | load_s | peak_vram | cold_prefill | warm_prefill | TTFT_cold | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| default `V2=0` | 512→4096 | 6.6 | 5.3→6.4GB | 27→92s | 7.5→73.5s | 34→98s | 77→61 |
| `V2=1` | 512→4096 | 13 | 19→21GB | 5.6→17.4s | 2.0→7.5s | 18→30s | 77→61 |
| `V2=1 CKV=1` | 512→4096 | 19→35 | 19→22GB | 1.9→12.3s | **0.17→1.6s** | 21→47s | 77→62 |

Decode (~61–78 tok/s, ~67% llama) dominates total request latency; prefill policy only moves TTFT.

## 6. Correctness / tok0
Byte-identical greedy (tok0 match) across every probe: auto-policy loads, concrete-KV cases, and ALL remainder-fix
A/B rows. No decode regression (the decode path is untouched).

## 7. Default behavior changed?
- **`PREFILL_V2` default: UNCHANGED (off).** New value `auto` is opt-in. Flipping the global default to `auto` is
  the one remaining owner call (conservative floor makes it low-risk on 24GB+, a no-op elsewhere).
- **`PREFILL_REMAINDER_FIX`: DEFAULT-ON** — but only fires *when `PREFILL_V2` is already on*, and is byte-identical,
  so it changes nothing for the default (`V2=0`) user; it strictly improves the opt-in fast path.
- `PREFILL_CONCRETE_KV` / `PREFILL_SERVER_PROFILE`: opt-in, default off.

## 8. Exact commands
```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_v2_auto_policy_probe.py
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_concrete_kv_policy_probe.py
DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_route_schedule_probe.py
# fast prefill (auto-fit), or server profile:
DEV=AMD PREFILL_V2=auto PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m <model.gguf> --warmup --benchmark 1
DEV=AMD PREFILL_SERVER_PROFILE=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm -m <model.gguf> --serve
```

## 9. Artifacts
`bench/qk-prefill-policy-integration/{prefill_v2_auto_policy,concrete_kv_policy,route_schedule_probe}.json`;
probes `extra/qk_prefill_{v2_auto_policy_probe,concrete_kv_policy_probe,route_schedule_probe}.py`; result docs
`docs/prefill-{v2-auto-policy,concrete-kv-policy,route-schedule}-result-20260620.md`; code in `tinygrad/llm/model.py`
(decision fns + generate fix) and `tinygrad/llm/cli.py` (hints).

## 10. Recommended user-facing policy (advertise three profiles)
| profile | invocation | for |
|---|---|---|
| **universal default** | (nothing) | any card incl. 16GB; safe; slow for long prompts (CLI prints a hint) |
| **fast (auto-fit)** | `PREFILL_V2=auto` | 24GB+ cards; ~5–15× faster prefill, enabled only where it fits |
| **server / long-prompt** | `PREFILL_SERVER_PROFILE=1` | servers / repeated / long prompts; best warm prefill (0.17–1.6s) |

Remaining: (a) owner decision on flipping the global default to `PREFILL_V2=auto`; (b) optional VRAM reduction to
fit 16GB cards (`docs/prefill-v2-vram-reduction-scope-20260620.md`). **Decode (~67% llama) is the real frontier —
prefill policy will not move it.**
