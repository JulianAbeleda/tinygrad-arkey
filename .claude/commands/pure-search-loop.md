---
description: Reconstruct the owned decode-attention kernel via delta closure (owned is the oracle), one turn, W==D-gated
argument-hint: "[max_meta=3] [--no-pairs]"
---

Run owned-kernel reconstruction in `/home/ubuntu/tinygrad-arkey` (AMD gfx1100, Qwen3-8B-Q4_K_M) as a **strict state
machine**, in ONE turn. The owned AMDGCN tile is the **ORACLE**; every iteration closes a NAMED owned-vs-generated
delta. Spec: `docs/pure-search-loop-owned-oracle-reconstruction-20260627.md`; taxonomy:
`bench/qk-search-spaces/owned_delta_taxonomy.json`. For the `/loop`-paced variant, a bare `/loop` runs `.claude/loop.md`.

**Rules (make the loop grounded):**
- Owned kernel is the oracle. Every candidate must target a named owned-vs-generated delta.
- Do NOT search knobs unless the auditor predicts which delta they move.
- If candidates do not move the named delta → classify `SEARCH_SPACE_BUG`, not "try more."
- `PROMOTABLE` requires W==D + token-match; isolated timing is diagnostic only.
- Exhaustion is valid only when every delta is closed, proven unrepresentable, or blocked by a named
  `INSTRUMENTATION_GAP`.
- Candidate authority = `extra/qk_pure_search_next_candidate.py` only; append-only JSONL ledger; **do not push**.

```python
def reconstruct_owned_kernel(max_meta=$1 or 3):                   # owned ASM tile = ORACLE; close the parity matrix
    for meta in range(max_meta):                                  # ceiling 6
        if run_gap_audit().degraded: return stop("DEGRADED")

        parity = run("extra/qk_owned_oracle_parity_audit.py")     # owned-vs-generated matrix (7 layers)
        if parity.verdict.startswith("PARITY_CLOSED") and wd_token_clears():
            return stop("PROMOTABLE")

        if parity.unknown_rows:                                   # INSTRUMENTATION_GAP -- do NOT search
            improve_responsible_tool(parity.unknown_rows[0])      # capture the missing owned-vs-gen datum
            assert parity_row_now_observable(); continue

        cand = next_candidate(failed_rows=parity.searchable_failed_rows)   # ONLY a candidate targeting a failed row
        if cand.verdict == "NO_UNTRIED_CANDIDATE_TARGETS_A_FAILED_ROW":
            # the failed row has no searchable axis (e.g. vgpr 88 vs 64 needs work-removal) or axes exhausted
            classify_SEARCH_SPACE_BUG_and_add_capability_or_escalate(parity.failed_rows)   # do NOT loosen the gate
            continue

        run_gates(cand)                                           # microgate -> occupancy
        moved = verify_target_row_moved(cand.targets_delta)       # re-run the row's responsible_tool toward owned
        if not moved:
            record(cand, "SEARCH_SPACE_BUG"); fix_generator_or_metric(cand.targets_delta); continue
        wd = run_wd_and_token_match(cand) if cand.requires_wd else run_isolated_then_wd(cand)
        record(cand, "PROMOTABLE" if wd.clears else "REFUTED_WD")
        if wd.clears: return stop("PROMOTABLE")
        continue
    return stop("META_CAP_REACHED")
```

Tools by role (the only authorities; timeout every GPU call): `isa_diff=extra/qk_decode_attention_isa_diff_gate.py`
+ `isa_vectorization=extra/qk_decode_isa_vectorization_gate.py` (structural primitive/placement deltas) ·
`hotloop=extra/qk_decode_hotloop_schedule_diff.py` (timing triggers) ·
`occupancy=extra/qk_decode_occupancy_guardrail.py` (pressure triggers) ·
`split_kv_economics=extra/qk_split_kv_economics_audit.py` (combine/lifecycle triggers) ·
`microgate=extra/qk_decode_attention_block_tile_microgate.py` (correctness + route-bound) ·
`W==D=extra/qk_decode_runtime_overhead.py`+`extra/qk_decode_token_match_check.py` (promotion authority) ·
`generator/ledger=extra/qk_pure_search_next_candidate.py`.

Every change default-off, cache-keyed, revert-clean, `[codegen]`/`[nn]`/`[test]`/`[docs]` commits (no Co-Authored-By),
surface SHA. **Do not push.** On stop print the open deltas, ledger counts, and the single recommended next step.
