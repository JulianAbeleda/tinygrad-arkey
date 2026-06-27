# Pure-search loop run — result (2026-06-27)

First bounded run of the `/loop`-wired pure-search decode loop (`.claude/loop.md`, escape hatch active). State:
`bench/qk-pure-search-loop/state.json`.

## Run

| iter | lever | gate | outcome |
|---|---|---|---|
| (prior, this session) | outer-b LDS independent split-combine (`DECODE_OUTER_B_SPLIT`) | microgate PASS; isolated +18%; VGPR 88→176, LDS 8K→16K | **REFUTED** (occupancy tax) |
| 1 | reduce-strategy knob `DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE` | microgate PASS; isolated ~2× slower (0.40→0.78 ctx512, 2.84→5.60 ctx4096) | **REFUTED** |
| 2 | (diagnose) | — | **NO_NEW_LEVER** → escape |

## Escape: `NO_NEW_LEVER`

The within-tile slope-bending search space is **exhausted**:
- outer-b independent split — built + correct + refuted (occupancy tax) this session;
- `INLINE_REDUCE` reduce-strategy knob — refuted (~2× slower; staged LDS reduce already better);
- `SCHED_UNROLL_SPLIT`, `DECODE_Q_HOIST`, `ds_permute` — ledger-refuted (prior).

Work-removal: the big one (`DECODE_FAST_EXP2`) is already shipped/in the stack; no further genuine removal found
(scale-fold is not a removal here — q is re-read per `rp`). Cross-lane is at per-token parity with owned
(diagnostic truth #4), so no cross-lane primitive is warranted.

Audit score unchanged: **60/100** (`PURE_SEARCH_PARTIAL...NOT_PROMOTABLE_YET`); no code changed this run.

## Recommended next layer (separately funded — not a within-tile knob)

`b`-parallelism must come from **more workgroups** (smaller `L` → more `s`-splits), paired with a **cheaper/fused
split-KV combine** — gated by **W==D**, not isolated timing (the combine tax only appears in-model). This is capped
by the split-KV combine economics already characterized in
`docs/split-kv-economics-audit-result-20260621.md` (`COMBINE_TAX_DOMINATES` / `COMBINE_SMALL_AMDAHL_LIMIT`). It is a
larger structural effort, not a knob — so the bounded loop correctly stops here rather than grinding refutations.
