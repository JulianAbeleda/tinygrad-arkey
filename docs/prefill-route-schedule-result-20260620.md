# Phase 3 result: fix the 32-token symbolic fallback trap (`PREFILL_REMAINDER_FIX`)

Date: 2026-06-20. Scope: `docs/prefill-policy-integration-scope-20260620.md` Phase 3. gfx1100 RX 7900 XTX, Qwen3-8B.
Harness: `extra/qk_prefill_route_schedule_probe.py` → `bench/qk-prefill-policy-integration/route_schedule_probe.json`.

## The trap

Under `PREFILL_V2`, the full-chunk branch fires only while `prompt_len - start_pos >= PREFILL_UBATCH(512)`.
A sub-512 **prompt remainder** falls to the `else` branch = many slow 32-token symbolic decode-style calls. Two
ways this happens: (a) a fresh prompt whose length isn't a multiple of 512 (the tail), and (b) **prefix-cache
resume** — `start_pos` jumps near the end, leaving a sub-512 tail. Both spend seconds on the tail.

## The fix (model.py `generate`, default-on `PREFILL_REMAINDER_FIX=1`)

Process the sub-512 remainder as ONE prefill-v2 chunk by **shifting the 512-window back to end exactly at
`prompt_len`**: `sp = prompt_len - PREFILL_UBATCH` (symbolic, reuses the one `prefill_v2_jit` — no per-remainder
compile). All-real tokens (no padding); the chunk's last position is `prompt_len-1` so `out.item()` is the next
token. It re-processes the small overlap with the prior chunk (same tokens → same KV) → correct. Reverts with
`PREFILL_REMAINDER_FIX=0`.

## Evidence (A/B fix OFF vs ON; all gates PASS)

| prompt | scenario | tok0 off==on | sched OFF | sched ON | prefill OFF→ON | speedup |
|---:|---|:--:|---|---|---|---:|
| 600 | fresh remainder (88) | ✓ 916 | int512 + 32-tok×3 | int512 + sym512 | 23.6→9.0 s | 2.6× |
| 1024 | prefix-cache resume (~425 tail) | ✓ 13284 | 32-tok×14 | sym512 | 6.6→3.0 s | 2.2× |
| 1500 | prefix-cache resume (~476 tail) | ✓ 916 | 32-tok×15 | sym512 | 7.7→0.55 s | **14.0×** |
| 2100 | full chunk + remainder | ✓ 323 | sym512 + 32-tok×3 | sym512×2 | 5.2→1.4 s | 3.8× |

Gates: **all tok0 match (byte-identical greedy)** ✓; **all 32-token remainder calls eliminated** (3/14/15/3 → 0) ✓;
**never slower** ✓. Decode path (`start_pos >= prompt_len`) untouched — no decode regression.

## Policy

`PREFILL_REMAINDER_FIX` is **default-on under `PREFILL_V2`** — it is a strict, byte-identical routing improvement
that removes the worst remaining prefill pathology (the prefix-cache-resume tail was up to 14× slower). Reverts
with `=0`. No effect when `PREFILL_V2` is off (the legacy path is unchanged) or for prompts `< 512` (which can't
shift back; they stay on the 32-token path, but are short/cheap).

Reproduce: `DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_route_schedule_probe.py`
