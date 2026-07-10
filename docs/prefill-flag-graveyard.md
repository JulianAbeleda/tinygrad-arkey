# Prefill Flag Graveyard

Durable track record of prefill flags/paths removed (or classified for removal) during the flag-collapse
into the two-wrapper design (canonical route selector + debug wrapper). A deleted path is a refutation
asset: this file records what each flag did, why it existed, its verdict, and why it was safe to remove,
so no one re-adds it blind.

Status legend: `CLASSIFIED` = verdict banked here, removal pending the wrapper build; `REMOVED <sha>` = deleted.

Source: read-only flag classification 2026-07-10 (5-family audit). Route ground truth: hybrid =
`prefill_pipe_role_selective_generated` (GRAPH_GEMM only, ~4413 pp512); pure =
`prefill_wmma_pipe_lds_dbuf_primitive_generated` (+PIPE+LDS+DBUF, ~1332).

---

## DBUF family (audit/model probes)

- **PREFILL_DBUF_D3A_AUDIT** — CLASSIFIED. Appended proof rows to the in-process `DBUF_D3A_AUDIT_LOG` at each D3-A staging decision. Existed to prove B-side D3-A cadence/epoch windows. Verdict/bank: `docs/fast-prefill-prescheduler-dbuf-scope.md`, `docs/8b-prefill-epoch-aware-stage-movement-scope.md`, `docs/dbuf-epoch-lifecycle-checker-scope.md` (log formalized there). Safe: log-only, zero emitted-code effect.
- **PREFILL_DBUF_LIFECYCLE_AUDIT** — CLASSIFIED. Second trigger for the same audit log (lifecycle/anchor-alias rows). Every read site is `LIFECYCLE_AUDIT or D3A_AUDIT` → fully redundant. Safe: log-only.
- **PREFILL_DBUF_LDS_PROOF_KEY_DUMP** — CLASSIFIED. `print("LDS_PROOF_KEY", …)` of the normalized LDS proof key. Superseded by the banked byte-window proof (`docs/fast-prefill-prescheduler-dbuf-scope.md`). Safe: pure print.
- **PREFILL_DBUF_ROTATED_STAGE_LOWERING_AUDIT** — CLASSIFIED. Collected A/B owner rows in `bufferize_to_store`. Doc certifies "byte-for-byte behavior-equivalent, sees two owners A/B, nbuf=2" (`docs/8b-prefill-rotated-dbuf-pipeline-construction-scope.md`). Safe: audit-only.
- **PREFILL_DBUF_OWNED_B_STAGE_ROTATE_MATERIALIZE** — CLASSIFIED. Tried to materialize the rotate owned-B stage instead of fast-failing. Verdict: "default-off failed probe; do not promote" — native allocation fails / no-spill collapse (`docs/8b-prefill-generated-dbuf-clustering-blocker-scope.md`). Safe: failed probe, off, no route.
- **PREFILL_DBUF_A_WINDOW_KEY_MODEL / _A_WINDOW_LIVE_UNR / _B_WINDOW_KEY_MODEL** — CLASSIFIED. Gated/parameterized a static LDS-size *estimate prefilter* for the offline resource search. Superseded by the authoritative measured ELF group-segment size (`docs/generated-dbuf-lifecycle-trace-scope.md`; code self-labels "prefilter only"). Safe: offline-search only, no machine-code effect.

## WMMA family — the destructive-suppression subsystem (proven wrong)

- **PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS** — REMOVED (Phase 2, Group A). Rewrote matched prologue LDS stores to NOOP by slot/addr key. Proved the opposite of its goal: `WRONG rr=nan` because slot-only suppression deletes a phase-0 producer needed before the first WMMA (store key is gone by late lowering). Bank: `docs/8b-prefill-lifecycle-compression-audit-20260709.md`, `docs/kmajor-dbuf-stage-ownership-primitive-scope.md`. Verdict: destructive suppression invalid at renderer-matcher scope. Safe: off, no route, standalone `if`.
- **PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS_EPOCH** — REMOVED (Phase 2, Group A). Epoch-key variant of the above. Superseded — renderer-side epoch keys too weak (different streams share a weak LDS address-family key). Bank: `docs/8b-prefill-epoch-aware-d3-self-sufficiency-scope.md`. Safe: off, standalone `if`.
- **PREFILL_WMMA_KMAJOR_PIPELINE_EPOCHS** — REMOVED (Phase 2, Group A). Warmup/body slot tracking + body-slot store suppression. Self-labeled "diagnostic-only, not a proof"; measured `WRONG rr=1.4` — proved pipeline construction must carry a value recurrence, not physical LDS windows. Bank: `docs/8b-prefill-epoch-aware-d3-self-sufficiency-scope.md`. Safe: off; all sites no-op when off.
- **PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS (+ _ROLE, _PHASE, _AUDIT)** — REMOVED (Phase 2, Group A). Owner-key destructive suppression cluster. This is explicitly the **forbidden_fallback** named by `extra/qk/prefill/prefill_stage_owner_audit.py` and asserted against by `test/unit/test_prefill_stage_owner_audit.py` — stage identity is gone by late lowering. Bank: compression-audit, kmajor-stage-ownership docs. Safe: off; deletion must also update the forbidden_fallback string literal + its test assertion (both only *name* the flag).
- **PREFILL_WMMA_KMAJOR_STAGE_STEAL_AUDIT / _STAGE_KEY_AUDIT** — REMOVED (Phase 2, Group A). Append rows to `DBUF_D3A_AUDIT_LOG`; the findings they produced are banked in the compression-audit/kmajor docs. Safe: log-only; STEAL_AUDIT shares the `if AUDIT and STEAL` guard but its body is pure append, so the KEEP_DEBUG STEAL emission is untouched.
- **PREFILL_WMMA_AB_ADDR_KEY** — REMOVED (Phase 2, Group A). Earlier experimental address-structure fragment key. Superseded by PROOF_KEY + PHASE_SCOPED_KEY (which the canonical kmajor route uses); on that route the addr-key branch is never taken. Safe: off → `_wmma_frag_addr_key` returns None; only other caller is the (also-deleted) FRAG_KEY_DUMP.
- **PREFILL_WMMA_CHAIN_KEY_DUMP / _PROOF_CHAIN_DUMP / _FRAG_KEY_DUMP** — REMOVED (Phase 2, Group A). Pure `print(...JSON...)` diagnostics of operand carrier tags / fragment-key provability / chain keys. Provability conclusions are banked in the census route + fragment-reuse scope docs. Safe: print-only, return immediately when off.

## TC_LOCAL_STAGE family — cooperative + refuted staging variants

- **PREFILL_TC_LOCAL_STAGE_COOP_POST / _COOP_B_POST / _COOP_B_LIMIT** — CLASSIFIED. Cooperative packed LDS store/load rewrite (+ B-only entry + rewrite-count cap). Verifier-clean but central route-bound gate **non-finite**; "does not solve this route." Bank: `docs/generated-machine-code-lds-dbuf-100pct-scope.md`, `docs/hand-vs-generated-delta-current-scope.md`. Superseded by the correct B_TILEKEY path (kept). Safe: off, no route.
- **PREFILL_TC_LOCAL_STAGE_COOP_GLOBAL / _COOP_LOCAL / _COOP_DROP_GLOBAL / _COOP_DROP_LOCAL / _COOP_DROP_UNROLL / _COOP_DROP_UNROLL_SIZE** — CLASSIFIED. Tune which tile-axis types the coop materializer accepts/drops. Only ever exercised inside the non-finite coop probe. Safe: pure tuning for an abandoned path.
- **PREFILL_TC_LOCAL_STAGE_A_FULL_LANE** — CLASSIFIED. 512-elem full-lane A LDS layout attempt. "Diagnostic-only… does not fix the cooperative correctness failure," gate non-finite. Safe: off, superseded.
- **PREFILL_TC_LOCAL_STAGE_SPLIT_POST_A** — CLASSIFIED. In `both` mode, stage A post / B early. REFUTED: `wrong output (rr=nan)`. Bank: 100pct-scope. Safe: banked refuted.
- **PREFILL_TC_LOCAL_STAGE_SCALAR_POST** — CLASSIFIED. Post-opt scalar contract-src staging — scalar LDS is the failure mode being eliminated (B2); no PASS; superseded by B_TILEKEY. Safe: dead scalar precursor.
- **PREFILL_TC_LOCAL_STAGE_TILE_ONLY** — CLASSIFIED (lowest confidence). WARP-only stage-range restriction; no recipe/route uses it, no banked positive; superseded by the WITH_LOCAL lane-range default. Safe: no live effect off.
- **PREFILL_TC_LOCAL_STAGE_B_TILEKEY_GENERIC_LAYOUT / _GENERIC_NO_SLOT** — CLASSIFIED. Alternate 8192-elem "generic" B tile-key layout + slot-carrier-drop sub-toggle. No PASS; the banked-correct path is the non-generic 256-elem layout. Safe: unbanked experimental layout, base B path kept.
- **PREFILL_TC_LOCAL_STAGE_B_TILEKEY_DROP_GLOBAL** — CLASSIFIED. Stage one N-tile instance instead of all when tile_count>64. REFUTED: `WRONG rr=1.2e+00`, density regressed ~94.75 inst/WMMA. Bank: 100pct-scope. Safe: banked refuted.

## LDS_PACK family — dumps, refuted packs, verifier-dead carriers

- **PREFILL_LDS_PACK_WITHLOCAL_DUMP / _B_TILEKEY_DUMP / PREFILL_WARMSTART_LOCAL_STAGE_DUMP** — CLASSIFIED. Pure `print(...)` diagnostics (class-2 in `amd.py`/`postrange.py` docstrings). The real decisions come from `_warmstart_local_stage_allowed` etc., independent of these. Safe: print-only.
- **PREFILL_LDS_PACK_WITHLOCAL_B64** — CLASSIFIED. Per-store `half.vec4→2×v_pack→ds_store_b64`, the first packed-store attempt. "Keep as a diagnostic substrate only"; fails native 4x4 (`NotImplementedError: no spills`). Superseded by WITHLOCAL_B128 (promoted). Bank: 100pct-scope. Safe: superseded, never correct on the real path.
- **PREFILL_LDS_PACK_LATE_MATCHER** — CLASSIFIED. Late pre-isel cooperative packed `ds_store_b128` matcher. Reaches the instruction shape but "central route-bound gate is non-finite" (NaN); correctness blocked by cooperative A-local layout. Superseded by WITHLOCAL_B128. Bank: 100pct-scope. Note: shares the `reserve_lds_pack` top-lift at amd.py:217 with WITHLOCAL_B128, which triggers it independently — safe to remove.
- **PREFILL_LDS_PACK_CARRIER / _POST_EXPAND** — CLASSIFIED. Neutral packed-fragment carrier / intended post-expander pack. Both "fail verifier on Ops.UNROLL dtypes.half over Ops.STACK"; inserted too early / never implemented as designed. Bank: 100pct-scope. Safe: never passed verifier.
- **PREFILL_LDS_PACK_GLOBAL_B128** — CLASSIFIED. Direct `global_load_b128→ds_store_b128` producer in the late-matcher carrier. Structural compile passes but route-bound gate non-finite; "opt-in diagnostic, not the default route." Only reachable via the also-dead LATE_MATCHER. Bank: 100pct-scope. Safe: value proof never held.
- **PREFILL_LDS_PACK_ALLOW_POOL** — CLASSIFIED. Returned the VGPR pool without excluding the fixed b128 scratch. Verdict: "Reject; fixed b128 scratch must stay excluded"; enabling worsens spills. Bank: `docs/dbuf-address-rematerialization-primitive-scope.md`. Safe: refuted; deleting restores the always-correct reserved-scratch behavior.
- **PREFILL_LDS_PACK_WITHLOCAL_SKIP_SMALL_B_TILEKEY_BUF / _WITHLOCAL_B128_GROUP_ONLY / _REJECT_GLOBAL_DISCONTINUITY** — CLASSIFIED. Isolation/narrowing probes (skip generic pack for the 16KiB B buf / early-return None from the b128 store / reject pack on >64B global jumps). All default-off diagnostics; no route enables them; the real paths are gated elsewhere. Safe: isolation probes.

---

## Delete candidates held as KEEP for now (revisit)

- **PREFILL_CHUNKED / _CHUNKED_EXPERIMENTAL / _CHUNK_RESIDENT_BLOCKS** — the chunked per-layer fp16 overlay. Verdict banked at `tinygrad/llm/model.py` ("replays stale captured state → AMD MMU faults") + `docs/8b-prefill-generated-lifecycle-performance-integration-scope.md`; superseded by `PREFILL_ROUTE=direct_packed` for 14B/32B. Fail-loud disabled unless `_EXPERIMENTAL=1`. Kept as debug overlay for now; delete together if the overlay is formally abandoned.
