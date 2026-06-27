---
description: Run the pure-machine-search X→Y→Z state machine in one turn (bounded), generator-driven, W==D-gated
argument-hint: "[max_meta=3] [--no-pairs]"
---

Run the pure-machine-search loop in `/home/ubuntu/tinygrad-arkey` (AMD gfx1100, Qwen3-8B-Q4_K_M) as a **strict
executable state machine**, in ONE turn (no scheduling). For the `/loop`-paced one-step-per-fire variant, a bare
`/loop` runs `.claude/loop.md`. Full spec: `docs/pure-machine-search-xyz-loop-codex-handoff-20260627.md`.

**Hard rules (make the loop hard to fool):**
- **Candidate authority = `extra/qk_pure_search_next_candidate.py` ONLY** (pairs on by default unless `--no-pairs`).
  No human lever-picking. Audit `next_actions` are **advisory only**.
- **`PROMOTABLE` requires W==D.** Local gates (microgate + occupancy + isolated-slope) only yield
  `LOCAL_PASS_WD_REQUIRED`. Only `token_match and pct_of_owned >= 90` → `PROMOTABLE`; a miss → `REFUTED_WD`.
- **Exhaustion is explicit:** `SEARCH_SPACE_EXHAUSTED` (X) → ask instruments (Y) → improve the tool (Z). Stop only
  when X **and** Y **and** Z all fail, or `meta >= max_meta`.
- **Ledger writes** go through the generator `--record '<json>'` (append-only JSONL).

```python
def loop(max_meta=$1 or 3, include_pairs=(not "--no-pairs")):
    for meta in range(max_meta):                         # absolute ceiling 6
        if run_gap_audit().degraded: return stop("DEGRADED")     # broken instrument → fix harness first

        # ---- X: solve the declared space ----
        while True:
            cand = next_candidate(include_pairs)                 # the generator is the ONLY authority
            if cand.verdict == "SEARCH_SPACE_EXHAUSTED": break   # → Y
            impl = implement_default_off_cache_keyed(cand)       # flag unset ⇒ byte-identical
            if not microgate(impl):            record(cand,"FAIL_CORRECTNESS");   revert_clean(); continue
            if not occupancy_guardrail(impl):  record(cand,"REFUTED_OCCUPANCY");  revert_clean(); continue
            if not isolated_bends_slope(impl): record(cand,"REFUTED_NO_SLOPE");   commit_or_revert(); continue
            record(cand,"LOCAL_PASS_WD_REQUIRED")
            wd = run_wd_and_token_match(impl, ckpts=[512,4096])   # the ONLY promotion authority
            if wd.token_match and wd.pct_of_owned >= 90:
                record(cand,"PROMOTABLE");  return stop("PROMOTABLE")   # hand off; do NOT auto-default
            record(cand,"REFUTED_WD");  commit_or_revert()

        # ---- Y: interpret exhaustion ----
        answers = synthesize(hotloop_diff(), occupancy_guardrail(), split_kv_economics(), top_level_gap_audit())
        lever = infer_new_searchable_lever(answers, ledger)       # a lever NOT in the ledger, addable as an axis
        if lever: add_axis_to_space(lever); continue              # reopens X

        # ---- Z: improve the auditor/searcher itself ----
        gap = highest_value_tool_gap(answers)                     # Level-2 topology axes / W==D-in-loop / etc.
        if not gap: return stop("GENUINE_EXHAUSTION")             # X∧Y∧Z all dry → real wall
        close_tool_gap(gap)                                       # default-off / new axes / new gate
        if not tool_now_surfaces_new_options_or_answers(gap):
            return stop("TOOL_IMPROVEMENT_DID_NOT_REOPEN_SEARCH")
        # meta += 1 happens via the for-loop; continue reopens X/Y with better tooling
    return stop("META_CAP_REACHED")
```

Tools (the only authorities; timeout every GPU call, a hang = failed gate):
`microgate=extra/qk_decode_attention_block_tile_microgate.py` · `occupancy=extra/qk_decode_occupancy_guardrail.py` ·
`isolated=extra/qk_decode_block_tile_isolated_timing.py` (diagnostic, NEVER promotion) ·
`W==D=extra/qk_decode_runtime_overhead.py`+`extra/qk_decode_token_match_check.py` ·
`generator/ledger=extra/qk_pure_search_next_candidate.py`.

Every change: default-off, cache-keyed, revert-clean, `[codegen]`/`[nn]`/`[test]`/`[docs]` commits (no
Co-Authored-By), surface SHA. **Do not push.** On stop, print the ledger counts and the single recommended next step.
