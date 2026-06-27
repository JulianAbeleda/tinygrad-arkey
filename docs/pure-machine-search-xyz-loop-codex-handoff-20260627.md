# Codex handoff — the X→Y→Z self-improving pure-machine-search loop (2026-06-27)

You are in `/home/ubuntu/tinygrad-arkey` (AMD gfx1100, Qwen3-8B-Q4_K_M). This describes, in pseudocode, the loop we
want to encode and run. Read `docs/pure-machine-search-roadmap.md` (authoritative live state) and
`docs/decode-attention-pure-search-state-and-learnings-20260627.md` (diagnostic truths) first.

────────────────────────────────────────────────────────────────────────
## 0. THE ASK (one sentence)

Drive the pure-machine-search problem with a **3-level escalating loop**: **X** = use the auditing tool to solve it;
when X's options are exhausted, **Y** = use the tool to *answer* why/what's-next; when Y is exhausted, **Z** =
*improve the auditing tool itself* so it can surface options/answers it couldn't before — then go back to X. Stop
only when X, Y, and Z are all dry (or a hard cap fires).

```
        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
        ▼                                                             │
   X: SOLVE ──options exhausted──▶ Y: ASK THE TOOL ──answers exhausted──▶ Z: IMPROVE THE TOOL
   (audit→lever→gate)             (interrogate instruments)              (close a tool capability gap)
        │                              │                                     │
   PROMOTABLE?                    NEW_LEVER?─────────────────────────────────┘ (TOOL_IMPROVED ⇒ back to X)
        │ yes                          │ no
        ▼                              ▼
      WIN                     all three dry ⇒ GENUINE_EXHAUSTION (escape)
```

────────────────────────────────────────────────────────────────────────
## 1. THE PROBLEM (what we're solving, and why it's hard)

Pure machine search = the **machine** (codegen + BubbleBeam) generates the default decode kernels, not hand-asm.
Decode GEMV is already pure. **Decode attention is the last hand kernel.** A generated block-tile route exists
(`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` in `extra/qk_flash_decode.py`, `DECODE_ATTN_BLOCK_TILE=1`)
with a stack of default-off codegen primitives; it is correct, transfers ~1.75× in-model, but is **3–15× slower than
owned**. The gap is **structural**, not a knob:

```
diagnostic_truths = {
  1: "gap is LATENCY/ILP-bound, not throughput (generated emits FEWER instr, WIDER loads, MATCHED occupancy)",
  2: "tile is OCCUPANCY-BOUND (vgpr 88/88, 4 wg/CU) ⇒ levers must REMOVE work, NOT add ILP-via-state",
  3: "ctx-slope = the serial OUTER b-block carry, not the inner tt carry",
  4: "ds_bpermute cross-lane is at per-token PARITY with owned ⇒ no cross-lane primitive warranted",
}
# Every slope lever that ADDS STATE has been refuted (it pays an occupancy tax > its ILP gain).
```

────────────────────────────────────────────────────────────────────────
## 2. TOP-LEVEL LOOP (the spine)

```python
def pure_machine_search(max_meta=3):           # max_meta = top-level escape hatch (hard ceiling)
    for meta in range(max_meta):
        x = solve_with_audit()                 # ── X ──
        if x == PROMOTED:            return WIN
        assert x == SOLVE_OPTIONS_EXHAUSTED

        y = query_the_tool_for_answers()       # ── Y ──
        if y == NEW_LEVER_FOUND:     continue   # tool's answer reopened X
        assert y == ANSWERS_EXHAUSTED

        z = improve_the_auditing_tool()        # ── Z ──
        if z == TOOL_IMPROVED:       continue   # better tool reopens X (and maybe Y)
        assert z == NOTHING_LEFT_TO_IMPROVE
        return GENUINE_EXHAUSTION              # X ∧ Y ∧ Z all dry → real wall
    return META_CAP_REACHED                    # hard cap → stop, hand off
```

────────────────────────────────────────────────────────────────────────
## 3. X — SOLVE (use the auditing tool to solve)

This is the inner generate→evaluate→prune→remember loop. **Built today** (the generator + ledger exist).

```python
def solve_with_audit():
    while True:
        audit = run("extra/qk_pure_search_gap_audit.py")          # DIAGNOSE (the auditing tool)
        if audit.verdict.startswith("PURE_SEARCH_PROMOTABLE"):
            return PROMOTED

        cand = run("extra/qk_pure_search_next_candidate.py")      # GENERATE + PRUNE (over the declared space)
        if cand.verdict == "SEARCH_SPACE_EXHAUSTED":
            return SOLVE_OPTIONS_EXHAUSTED                        # ← hands control to Y

        impl = implement(cand)              # default-off, cache-keyed (flag unset ⇒ byte-identical)

        # GATES in authority order — first failure ⇒ revert clean + remember, next candidate
        if not microgate(impl):           ledger.add(cand, "FAIL_CORRECTNESS");   revert(); continue
        if not occupancy_ok(impl):        ledger.add(cand, "REFUTED_OCCUPANCY");  revert(); continue
        if not isolated_bends_slope(impl):ledger.add(cand, "REFUTED_NO_SLOPE");   commit_or_revert(); continue

        wd = run("extra/qk_decode_runtime_overhead.py", ckpts=[512,4096]) + token_match()   # W==D authority
        if wd.pct_of_owned >= PROMOTION_THRESHOLD:
            ledger.add(cand, "PROMOTABLE"); return PROMOTED
        ledger.add(cand, "REFUTED_WD")     # bent isolated slope but didn't clear W==D
```

Gates (the actual tools):
```
microgate        = extra/qk_decode_attention_block_tile_microgate.py   # BLOCK_TILE_MICROGATE_PASS, max_abs ≤ 5e-3
occupancy_ok     = extra/qk_decode_occupancy_guardrail.py              # vgpr≤88, scratch=0, wg/CU≥4 vs baseline
isolated_slope   = extra/qk_decode_block_tile_isolated_timing.py       # diagnostic only, NEVER promotion authority
generator+space  = extra/qk_pure_search_next_candidate.py  +  bench/qk-search-spaces/decode_attention_loop_search_space.json
```

────────────────────────────────────────────────────────────────────────
## 4. Y — QUERY (when X is dry, use the tool for ANSWERS)

X exhausting means "every candidate in the declared space is in the ledger." Don't stop — **interrogate the
diagnostic instruments** for the *wall* and the *next layer*, and see if any answer implies a lever NOT yet in the
space. **Partly built** (the instruments exist; the synthesis/inference step is manual today).

```python
def query_the_tool_for_answers():
    bound     = run("extra/qk_decode_hotloop_schedule_diff.py")    # WHAT is the bound? cross-lane / waitcnt / sched
    headroom  = run("extra/qk_decode_occupancy_guardrail.py")      # HOW pinned? vgpr/lds/wg-per-CU
    economics = run("extra/qk_split_kv_economics_audit.py")        # is the next layer combine-capped?
    gapaudit  = run("extra/qk_pure_machine_search_gap_audit.py")   # top-level: which workload/axis is the blocker?

    answers = synthesize(bound, headroom, economics, gapaudit)
    # Example answers we've already produced manually:
    #   "bound = occupancy + structural per-token reduce; slope is the serial b-carry"
    #   "next layer = MORE WORKGROUPS (smaller L) + cheaper/fused combine, W==D-gated"
    #   "that layer is combine-economics-capped (COMBINE_TAX_DOMINATES / COMBINE_SMALL_AMDAHL_LIMIT)"

    lever = infer_lever_from_answers(answers, ledger)
    if lever and key(lever) not in ledger and key(lever) in declared_space_or_addable(lever):
        search_space.add_axis(lever)         # the tool's answer revealed a NEW searchable lever
        return NEW_LEVER_FOUND               # ← reopens X
    return ANSWERS_EXHAUSTED                  # the answers point only to refuted levers OR to a layer the tool
                                             #   cannot currently SEARCH → that is a TOOL capability gap → Z
```

────────────────────────────────────────────────────────────────────────
## 5. Z — IMPROVE THE AUDITING TOOL (when Y is dry, upgrade the instrument)

Y exhausting means the answers point at a layer the tool **cannot currently measure, generate, or search**. That is
a **capability gap in the auditor itself**. Close the highest-value gap so the tool surfaces options/answers it
couldn't before — then X (and Y) reopen. **This is the durable, compounding capability.**

```python
def improve_the_auditing_tool():
    gaps = audit_the_auditor()   # adversarially review the tool vs what the answers demand
    # CLOSED this session (examples of Z done):
    #   - headline scores were LITERALS         → now DERIVED from live artifacts (responsive, fail-loud)
    #   - no generator / no enumerable space    → added generate+prune (Section 3)
    #   - no durable ledger feedback            → added the manifest ledger
    # OPEN gaps (the ladder Z should climb next), highest value first:
    OPEN = [
      "search space is KNOBS only (Level-1); cannot express/search TOPOLOGY candidates (Level-2: more workgroups, "
        "batched reduce, fused combine, different tile shapes). ADD STRUCTURAL AXES + a candidate generator that "
        "emits tile-structure variants, not just flags.",
      "no W==D gate INSIDE the loop; isolated timing is the only in-loop speed signal (it is not authority). "
        "WIRE qk_decode_runtime_overhead W==D + token-match as the promotion gate.",
      "transfer snapshot is HAND-TYPED (wd_authority=session_reported). MAKE it harness-measured.",
      "the winning flags are MANUAL, not BubbleBeam-owned. BIND them into FutureSight/BubbleBeam candidate "
        "provenance so 'search-owned' becomes true (the flags_search_owned score component).",
      "the combine is a separate kernel; expose a FUSED/cheaper combine as a searchable primitive class.",
    ]
    gap = pick_highest_value(gaps + OPEN)
    if gap is None:
        return NOTHING_LEFT_TO_IMPROVE          # tool is as good as we can make it → GENUINE_EXHAUSTION
    extend_tool(gap)                            # implement it (default-off / new manifest axes / new gate wiring)
    assert tool_now_surfaces_new_options(gap)   # verify the improvement actually reopens X or Y
    return TOOL_IMPROVED                        # ← back to X with a more capable instrument
```

The KEY: **Level-1 (knobs) vs Level-2 (topologies).** The knob space (X today) tunes one fixed tile whose
*structure* is the bottleneck — so X will likely exhaust without closing the gap. The real gap-closer is Z making
the auditor able to **search over tile structures / primitive classes** (the roadmap's stated goal: *"expose the
missing low-level primitive classes, then search over those classes"* — not more knob search).

────────────────────────────────────────────────────────────────────────
## 6. DATA STRUCTURES (persistent)

```python
SearchSpace = {                         # bench/qk-search-spaces/decode_attention_loop_search_space.json (committed)
  "baseline_stack": {...flags...},
  "axes": [ {flag, values, kind: "knob"|"primitive"|"topology", cost, priority, hypothesis, predicted?} ],
  "ledger": [ {candidate, outcome, gate, why} ],   # durable refutation memory; the generator prunes against it
}
SessionState = {                        # bench/qk-pure-search-loop/state.json (gitignored; per-campaign counter)
  "iteration", "max_iterations", "status", "stop_reason"
}
Outcome ∈ { PROMOTABLE, REFUTED_OCCUPANCY, REFUTED_NO_SLOPE, REFUTED_WD, FAIL_CORRECTNESS, NO_NEW_LEVER }
```

────────────────────────────────────────────────────────────────────────
## 7. ESCAPE HATCH (exhaustive — the loop MUST be bounded)

```python
STOP if any:
  meta >= max_meta                       # top-level cap (hard ceiling, e.g. 3; never exceed 6)
  X returns PROMOTED                     # WIN → hand to W==D promotion review (do NOT auto-default)
  X∧Y∧Z all dry                          # GENUINE_EXHAUSTION → report wall + next funded layer
  audit/gates report DEGRADED            # broken instrument → fix harness FIRST, do not loop on bad data
  per-iteration: every GPU gate has a timeout (a hang = a failed gate, not a stall)
  per-iteration: commit-or-revert-clean  # NEVER leave the tree dirty between iterations
  built-in: Esc cancels a pending /loop fire; recurring tasks expire after 7 days
```

The loop is wired to the built-in `/loop` via `.claude/loop.md` (one iteration per fire) and to `/pure-search-loop`
(bounded run in one turn). Today `.claude/loop.md` implements **X only** with a flat stop. **Your job: add the Y and
Z escalation** so a dry X triggers Y, a dry Y triggers Z, and only X∧Y∧Z-dry stops.

────────────────────────────────────────────────────────────────────────
## 8. CURRENT STATE (what's built vs open)

```
BUILT:   X (generator + space + ledger + gates) ; the auditor is now DERIVED/fail-loud (Z done once) ;
         the outer-b split-combine codegen primitive (DECODE_OUTER_B_SPLIT) — BUILT + correct + REFUTED (occupancy)
LEDGER:  OUTER_B_SPLIT=2, INLINE_REDUCE=1, SCHED_UNROLL_SPLIT, Q_HOIST, ds_permute  → all REFUTED
UNTRIED: STAGE_COALESCE∈{2,8}, SCHED_UNROLL∈{4,16}  (Level-1 knobs the generator will emit next; cheap; isolated-gate)
OPEN Z:  Level-2 topology axes ; W==D in-loop ; harness-measured snapshot ; BubbleBeam flag-ownership ; fused combine
```

────────────────────────────────────────────────────────────────────────
## 9. CONCRETE TASKS FOR CODEX (in order)

1. Encode X→Y→Z in `.claude/loop.md` (today it's X-only): dry-X ⇒ run Y (Section 4), dry-Y ⇒ run Z (Section 5),
   X∧Y∧Z-dry ⇒ stop. Keep the escape hatch (Section 7).
2. Finish X's Level-1 sweep first (constructive): run the generator to exhaustion over the knob space, banking each
   outcome in the ledger (expected: knobs don't bend the slope — confirm, don't infer).
3. Implement the first **Z capability gap**: add **topology axes** to the search space (more-workgroups / smaller-L,
   batched-reduce, fused-combine) + a generator that emits *structural* candidates, and wire **W==D** as the in-loop
   promotion gate. This is the gap-closer; Level-1 knobs are not.
4. Every change: default-off, cache-keyed, microgate-gated, revert-clean, `[codegen]`/`[nn]`/`[test]`/`[docs]`
   commits (no Co-Authored-By), surface SHA. Do not push unless asked.
