---
description: Run the measure-first pure-search decode loop (diagnose→solve→gate→decide) with a hard escape hatch
argument-hint: "[max_iterations=3] [optional: starting lever, e.g. 'work-removal exp on PV']"
---

You are running the **pure-machine-search decode loop** in `/home/ubuntu/tinygrad-arkey` (AMD gfx1100,
Qwen3-8B-Q4_K_M). This is a measure-first, bounded loop. **The escape hatch below is non-negotiable — never run
indefinitely.**

## Arguments
- `$1` = max iterations (default **3** if absent). **Hard ceiling: 6** — refuse to exceed it regardless of input.
- `$2..` = optional starting lever hint. If absent, take the rank-1 lever from the audit.

## Read first (don't re-derive)
- `docs/pure-machine-search-roadmap.md` (authoritative live state)
- `docs/decode-attention-pure-search-state-and-learnings-20260627.md` (diagnostic truths + refuted levers)
- The latest `docs/*-result-*.md` for the active lever.

## Pre-flight (once, before iterating)
1. `git status` must be clean (or stash/commit first). Confirm you're in `tinygrad-arkey` on `master`.
2. Run the diagnose tool and the smoke; if either is broken, **STOP** (fix the harness first — a broken
   instrument poisons the loop):
   ```
   DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_pure_search_gap_audit.py   # must NOT be DEGRADED
   ```

## One iteration (repeat until an escape condition fires)

**1. DIAGNOSE** — run the audit; read its `verdict`, `pure_search_score`, rank-1 `next_actions`, and live gates:
```
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_pure_search_gap_audit.py
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_hotloop_schedule_diff.py   # what's the bound
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_occupancy_guardrail.py            # headroom
```
Pick the lever: `$2` if given on the first iteration, else the audit's rank-1 action. **Check the refutation
ledger** (the `*-result-*.md` docs + the outer-b contract `refuted` flags): if the lever is already recorded as
refuted, **do not re-chase it** — pick the next-layer lever it names, or if none, escape (no-new-lever).

**2. SOLVE** — implement the lever as a **default-off, cache-keyed** codegen/work-removal change (flag unset ⇒
byte-identical; add the flag to the `to_program` cache key in `tinygrad/codegen/__init__.py`). Prefer
**work-removal** levers (no new state) — they strictly dominate on this occupancy-bound tile. Decline the
unverifiable (non-const bounds, untested shapes).

**3. GATE** (authority order — stop at the first failure, revert clean):
```
# a. correctness (authority): must print BLOCK_TILE_MICROGATE_PASS, max_abs <= 5e-3
DEV=AMD JIT=1 <FLAGS> PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_block_tile_microgate.py
# b. occupancy: capture ISA then guardrail — must NOT regress vgpr/scratch/wg-per-CU vs baseline
DEV=AMD JIT=1 <FLAGS> PYTHONPATH=. .venv/bin/python extra/qk_decode_isa_vectorization_gate.py
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_occupancy_guardrail.py
# c. did it move the RIGHT loop / bend the slope? (diagnostic, NOT promotion authority)
DEV=AMD JIT=1 <FLAGS> PYTHONPATH=. .venv/bin/python extra/qk_decode_block_tile_isolated_timing.py
```
where `<FLAGS>` = the best stack + your new flag:
`DECODE_ATTN_BLOCK_TILE=1 DECODE_STAGE_COALESCE=4 COALESCED_LOAD_LOWERING=1 SCHED_UNROLL=8 SCHED_LIST=1 DECODE_FAST_EXP2=1 <YOUR_FLAG>`.
Give every GPU command a timeout (e.g. 540s); a hang counts as a failed gate.

**4. RECORD** — update the relevant contract artifact (`built` / `bends_slope` / `refuted` + measured ms/resources),
re-run the audit to confirm the score moves (or honestly stays), and write/update a dated `docs/*-result-*.md`.
Then **commit** the iteration (subsystem prefix `[codegen]`/`[nn]`/`[test]`/`[docs]`, **no Co-Authored-By**) or
**revert clean** on failure. Surface the SHA + title. **Never leave the tree dirty between iterations.**

**5. DECIDE** — classify the outcome and either escape or continue:
- **PROMOTABLE** (correctness PASS + occupancy no-regress + bends slope) → run W==D
  (`QK_CKPTS=512,4096 ... extra/qk_decode_runtime_overhead.py` + token-match). **ESCAPE** (hand off; do not auto-promote a default).
- **REFUTED** (correct but doesn't bend slope, or occupancy regresses) → record the refutation + the next-layer
  lever it implies, then **continue** to the next iteration with that lever.
- **HARD WALL** (`SEARCH_BLOCKED_BY_CODEGEN`/`_RUNTIME` with no tractable lever) → **ESCAPE** and report.

## ESCAPE HATCH (any one fires → stop the loop and write a final summary)
1. **Iteration cap reached** (`$1`, default 3, hard max 6).
2. **PROMOTABLE** candidate found (hand off to W==D — do not loop further on it).
3. **HARD WALL**: audit/gates report a codegen/runtime block with no tractable next lever.
4. **DEGRADED**: the audit reports `DEGRADED` / missing inputs (instrument is broken — fix harness, don't loop).
5. **No new lever / loop-until-dry**: the only available lever is already in the refutation ledger, or two
   consecutive iterations produce no new lever or no score movement.
6. **Dirty/again-failing**: a gate fails to revert clean, or correctness can't be restored — stop and report.

## Final summary (always, on escape)
State: iterations run, levers tried, each outcome (promotable/refuted/walled), the current audit score + verdict,
the commits made (SHA + title), and the single recommended next lever. Be honest — a refutation is a result.
**Do not push** unless explicitly asked.
