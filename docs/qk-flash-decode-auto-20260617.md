# Flash-decode auto-enable for long-context decode (2026-06-17)

> **PARTIALLY SUPERSEDED.** Auto-enable shipped, but the default threshold was later lowered to **512** and the
> partial kernel replaced by `FLASH_VARIANT=hoisted` (+ `FLASH_L=128`). Current authority:
> `qk-8b-decode-banked-20260617.md` + `qk-8b-flash-variant-result-20260617.md`.

The first measured win after the decode gap analysis + small-op audit: auto-select flash-decode when it's
expected to win (long context), preserving short-context behavior and override controls. No new kernel.

## Before the change
Flash-decode was **default-off, manual-only**: selected at `model.py:_attention` iff
`(self._use_flash or getenv("FLASH_DECODE")) and start_pos is symbolic and T==1`. `FLASH_DECODE` was a binary
truthy flag (default 0 → never). So every decode used SDPA unless the user set `FLASH_DECODE=1`, leaving the
measured long-context win (1.73× @4096) unclaimed by default.

## Policy implemented
Centralized `should_use_flash_decode(start_pos, T, use_flash)` (tinygrad/llm/model.py), called from `_attention`:
- **Decode-only invariant:** `T==1` and `start_pos` is symbolic (UOp) — else SDPA (prefill/unsupported).
- **`FLASH_DECODE=0`** → force SDPA (overrides everything, incl. programmatic `_use_flash`).
- **`FLASH_DECODE=1`** (or `self._use_flash`) → force flash where invariants hold.
- **`FLASH_DECODE=auto` (default, unset)** → flash iff trace-time context ≥ `FLASH_DECODE_THRESHOLD` (default
  **1024**). The context is read from the bound `start_pos` at JIT capture; if it can't be read → conservative SDPA.

## Threshold chosen: 1024, and why
The selection (flash vs SDPA) **bakes at JIT capture** (the decode graph is captured once with a symbolic
start_pos; flash-decode's kernel handles all KV lengths, but which path is chosen is fixed at capture). So the
policy keys on the **decode-start context** (= prompt length). 1024 is conservative: the long-context bench
shows flash is 1.05× @512 (a slight win) but the gains are material only from 1024 up (1.25×) — so 1024 captures
the real wins while leaving short-context decode (the banked ~64 tok/s regime) exactly as SDPA. Tunable via
`FLASH_DECODE_THRESHOLD`.

## Benchmark: auto vs off vs on (8B Q4_K_M, gfx1100; `bench/qk-flash-decode-auto-20260617/`)

Fresh JIT per depth (so auto evaluates the policy at each decode-start context):

| ctx | auto | off (SDPA) | on (flash) | auto / off | argmax match |
|---:|---:|---:|---:|---:|:--:|
| 512 | 37.9 | 37.9 | 40.9 | **1.00×** | ✓ |
| 1024 | 35.2 | 28.1 | 35.2 | **1.25×** | ✓ |
| 2048 | 27.8 | 18.9 | 27.8 | **1.47×** | ✓ |
| 4096 | 19.7 | 11.4 | 19.7 | **1.73×** | ✓ |

Auto == off below 1024 (chose SDPA, identical — zero regression); auto == on at ≥1024 (chose flash, full win).
**All acceptance gates pass:** ctx 512 no regression, ctx 1024+ positive, output sane.

## Output / correctness
**argmax matches SDPA at every context** — flash-decode is exact-vs-SDPA up to fp reassociation, and the greedy
token is unchanged. Full external suite green (287 passed). Default-auto does not regress existing short-context
tests (they decode at <1024 → SDPA, unchanged).

## Override behavior
- `FLASH_DECODE=0` — force SDPA everywhere (the prior default; for A/B or debugging).
- `FLASH_DECODE=1` — force flash for all symbolic decode (incl. short ctx; ~1.05–1.08× @512).
- `FLASH_DECODE=auto` / unset — threshold policy (default).
- `FLASH_DECODE_THRESHOLD=N` — tune the auto cutoff.

## Known limitations
- The flash-vs-SDPA choice **bakes at JIT capture from the decode-start context** (prompt length). A session that
  starts short (<1024) but generates past 1024 stays on SDPA for that capture — conservative (won't enable
  flash mid-session), and it misses that tail. Capturing per-depth (as the bench does) or a long prompt gets the
  win. A true per-token switch would need two captured graphs + dispatch (out of scope; not worth it given the
  common long-context case is a long prompt).
- Below 1024, a small ~1.05–1.08× flash win is intentionally left on the table for short-context safety.

## Why this is the first action after the small-op audit
The audit showed decode attention (SDPA) is the only non-GEMV primitive big enough to matter (~16% GPU, grows
with context), and its fusion already exists (flash-decode). RMSNorm/SwiGLU/residual are each <3.5% and not
worth a fusion arc. So the highest-value, lowest-risk, no-new-kernel action is exactly this: auto-enable the
existing flash-decode for long context. **It should now be considered the default long-context decode path.**
