# Pure-search loop — first full X→Y→Z run (2026-06-27)

The strict state machine (`docs/pure-machine-search-xyz-loop-codex-handoff-20260627.md`) run end-to-end, inline,
driven by the generator. It demonstrated the **self-improving escalation**: X exhausted → Y found the wall → Z
upgraded the instrument → search reopened.

## X — solve the declared knob space (EXHAUSTED, all refuted)

14 generator-emitted candidates (6 singles + 8 pairs), each gated (microgate correctness → isolated slope) and
recorded to the JSONL ledger. **All `REFUTED_NO_SLOPE`** (isolated ms vs baseline 0.403/2.84):

| candidate | isolated ctx512/ctx4096 | note |
|---|---|---|
| STAGE_COALESCE=2 | 0.487/3.515 | slower |
| STAGE_COALESCE=8 | 0.387/2.758 | **uniform ~3% faster but slope ratio UNCHANGED** (7.13 vs 7.05) — not a bend |
| SCHED_UNROLL=4 | 0.635/4.545 | underhides tt-carry |
| SCHED_UNROLL=16 | 0.801/5.726 | no unroll (16≯16)/overflow |
| OUTER_B_SPLIT=4 | declines @NB=6 | correct + occupancy PASS (vgpr **80** — NOT exploded like K=2's 176; NB=4/K=4 degenerates); no slope bend |
| 8 pairs | all 0.63–0.90 / 4.5–6.4 | all slower (combinations of refuted knobs) |

`SEARCH_SPACE_EXHAUSTED`: tried_in_space 14, remaining 0, historical_refutations 4. This is *genuine* exhaustion
(the generator covered the declared space), not "the human ran out of ideas."

## Y — interpret the exhaustion

Instruments: hotloop = `CROSSLANE_OVERHEAD_BOUND`; occupancy = `PASS` (pinned but passing). The bound is
**structural**, and Y surfaced a genuinely **untried** lever the ledger lacks: **split granularity**
(`DECODE_ATTN_BLOCK_TILE_FIXED_S` + `DECODE_ATTN_FUSED_XLANE_SCORE_PV_S`, the "more workgroups" Level-2 lever, 0× in
ledger). **But** its cost is the split-KV combine tax, which is **in-model only** → X's isolated gate would mislead.
The lever exists, but the *searcher cannot judge it* → `ANSWERS_EXHAUSTED` → Z.

## Z — improve the searcher (and it reopened X)

Closed the highest-value tool gap: **the search space was knobs-only; it could not express/search topology levers,
and had no W==D gate.** Z added a `topology` axis kind to the generator (carries an `enable` flag +
`requires_wd: true`) and a topology axis to the manifest (`FIXED_S=1, S∈{64,96}`), and taught the loop to gate
`requires_wd` candidates with **W==D** (skip isolated). Verified: the generator — which had returned
`SEARCH_SPACE_EXHAUSTED` — now emits
`DECODE_ATTN_BLOCK_TILE_FIXED_S=1,DECODE_ATTN_FUSED_XLANE_SCORE_PV_S=64` (kind=topology, requires_wd=True,
gate=W==D). **The tool got smarter and the search reopened** (Level-1 knobs → Level-2 topology).

## Stopping point (clean)

The loop is back in X with a topology candidate queued that requires a **W==D run** (~30 min/candidate) — the
funded Level-2 work. Per the split-KV combine economics it is predicted combine-capped, but it is **untried for the
generated tile** and now correctly W==D-gated (not isolated-misled). That W==D run is the next step.

Net: the loop ran the full X→Y→Z, exhausted the knob space honestly, and *improved its own instrument* to reopen
the search at the next layer — exactly the self-improving design. Human lever-picking stayed off the hot path;
`PROMOTABLE` was never claimed (no W==D cleared).
