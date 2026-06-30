# Decode-Attention Two-Kernel Problem — Audit Result (2026-06-25)

## Update (2026-06-30): verdict reinforced; current state

Since this audit, two later results confirm the conclusion (the owned two-kernel tile+combine remains the shipped
default; do not spend broad search on attention):

- **Native generated attention exists, but is correct-not-fast.** The native AMD-ISA backend closed the primitive gaps
  this audit flagged as "a capability, not a speed lever" (vector dot, cross-lane reduce, LDS, barriers, grid
  parallelism, hardware exp, dynamic split count, register accumulators). The generated attention tile is token-correct
  and route-bound but lands ~60–68% of the owned tile's per-token speed, so it is not promoted.
- **Ceiling audit says attention is low-leverage.** A whole-decode ceiling audit found decode is weight-memory-bound:
  the attention KV-read floor is under 1% of the per-token weight-read floor, so even a perfect attention tile barely
  moves whole-decode tokens/s. Decode is near its practical ceiling.
- **The leverage the audit said was "outside attention" materialized in the weight path.** The Q4_K weight matrix-vector
  kernels are now search-generated and speed-equivalent to the owned hand-tuned ones (promoted); a Q6_K direct-route
  variant was tried and refuted (no speedup, stays default-off). The remaining decode headroom is structural, not a
  tile/combine lever.

The rest of this document (the original audit) stands as written.

## Decision: **`TWO_KERNEL_PROBLEM_EXHAUSTED_CURRENT_ROUTE`**

The decode-attention split-KV route (TILE `owned_flash_tile_gqa_whole` + a **separate** log-sum-exp COMBINE
`owned_flash_combine`) is the **shipped default** and works in-model (byte-identical, no `E_49152`, unknown-bucket
lockstep-PROVEN, +13–19 % W==D vs the slice comparator, **100.6–104 % of llama.cpp** across ctx 512–4096). Every
bounded lever for *this* route has been measured and closed:
- the cheaper/fused-combine lever the 06-21 economics audit pointed at was **built (B5-lite, hardware 2.4× faster) and
  W==D-REFUTED** (saturates ~+5.7 % @ctx4096 because the combine **overlaps in-graph**, off the serial critical path),
  directly falsifying the 06-21 free-combine +8.58 % projection;
- the aggressive non-search probe falls only ~0.7–1.5 %/ctx short and is explicitly `do_not_promote`;
- the ctx-slope audit returns `NO_ACTION` with the two-kernel residual "below action bar";
- the campaign synthesis declares attention exhausted and decode HBM-bandwidth-bound at parity+.

A truly **fused single-lifecycle** tile+combine kernel is **codegen-walled** (multi-granularity reduce→broadcast→reduce
producers are not expressible via the UOp store-group idiom — the q8-lifecycle `Q8L-2` precedent), not a bounded build.
**Open decode levers remain *outside* the two-kernel attention boundary** (weight-GEMV/FFN share — the aggressive probe;
small-op fusion; native codegen). This verdict is scoped to the two-kernel attention route; it is **not** a claim that
decode as a whole is closed (per the 2026-06-23 owner correction: frame remaining decode work as open levers).

Runner-ups rejected: `ACTIVE_BOUNDED_TARGET` — the named combine lever was tried and W==D-refuted, not untried;
`ACTIVE_BUT_NEEDS_NEW_PRIMITIVE` — a fused primitive is codegen-walled **and** projected sub-gate (overlap ⇒ even a free
combine ≈ +5.7 %), so it is not an actionable bounded target for this route; `BLOCKED_BY_MISSING_ATTRIBUTION` —
attribution is present and convergent (lockstep PROVEN, route_preserved, B5 measured, ctx-slope audit).

## Primitive boundary (per `structure/Development/performance-primitive-research-principles.md`, "Split-KV Reduction Economics Are Part Of The Decode Primitive")

The decode-attention primitive is **two components that must BOTH be measured before promotion** — a tile A/B alone is
not W==D-ready:

1. **TILE kernel** — Flash-Decoding split-KV tile (`owned_flash_tile_gqa_whole`): q·k / online-softmax / PV partial
   work, LDS staging, vector dot (`v_dot2`) + cross-lane (`ds_bpermute`), one online-softmax state `(m, l, PV[D])` per
   split. Grid `(Hkv,S,1) = (8,48,1)`, block `(128,1,1)` → 384 workgroups.
2. **COMBINE / integration path** — a **separate** log-sum-exp merge (`owned_flash_combine`) of the `S` partials, plus
   the K/V read ABI (whole-cache buffer identity vs sliced views), split-count/workgroup economics, dispatch count /
   graph-node integration, hidden copies/materialization (`E_49152`), and W==D transfer. Grid `(Hq,1,1) = (32,1,1)`,
   block `(32,1,1)` → 32 workgroups. *"That combine is part of the primitive — a separate lifecycle/economics tax a
   tile A/B never measures. It gives back part of the tile win every layer."*

Both are precompiled AMDGCN `.co` ELFs injected as separate `Ops.PROGRAM` JIT graph nodes via `Tensor.custom_kernel`
(`extra/qk_owned_flash_decode_graph_node.py`, "Two precompiled graph nodes"), sourced from
`extra/qk_owned_flash_decode.hip`.

## Timeline reconciliation (why 06-21 `COMBINE_TAX_DOMINATES` and 06-24 "shipped + exhausted" are consistent)

| date | state | meaning |
|---|---|---|
| **06-21** | B4 owned tile default-**OFF**, W==D **FAIL** (best +5.41 % @ctx4096 < +7 % gate), `COMBINE_TAX_DOMINATES`; Amdahl **estimate**: half-combine +6.97 %, free +8.58 % | looks like an active bounded combine lever |
| **06-22** | (a) the 06-21 fail was partly a **dtype-contract bug** (fp32 cache read as fp16) + over-conservative ctx guard, both fixed + native fp16 cache; (b) **buffer-identity whole-cache KV read** (`DECODE_ATTN_KV_IDENTITY`) removes the `E_49152` slice tax → **+13–19 %**, owner-authorized DEFAULT-ON; (c) **B5-lite** builds a 2.4× cheaper combine → W==D **saturates ~+5.7 %** (`B5_COMBINE_LOCAL_PASS_WD_FAIL`) | tile promoted via bug-fix + cache-ABI, **not** via a cheaper combine; the combine lever is refuted (overlaps in-graph) |
| **06-23** | campaign synthesis: decode **102–105 % of llama**, default-on, `POST_PARITY_HARDENING_COMPLETE`; attention exhausted, decode HBM-bound at parity+ | route shipped |
| **06-24** | lifecycle-recheck `PASS` (route fires, no `E_49152`, lockstep PROVEN, +13–19 % vs slice); aggressive-target-proof `UNPROVEN__THROUGHPUT` (~0.7–1.5 % short, `do_not_promote`); ctx-slope `NO_ACTION` | only a sub-threshold aggressive delta remains |

The 06-21 "W==D fail" was **not** purely combine economics: it was partly a bug (since fixed) plus a combine projection
(since refuted by B5). The route was made promotable by the **buffer-identity** lever, and the combine was proven to be
a non-issue at the whole-decode level.

## Answers to the 10 questions

- **Q1 — exact tile + combine kernels.** Tile `owned_flash_tile_gqa_whole` (`whole_cache=True` default; shipped tile
  constants `tk=16,vec=1,unroll=1` → plain symbol; the 06-24 recheck capture used a byte-identical Mode-B variant
  `…_tk32_v4_u2`). Combine `owned_flash_combine` (`combine='base'` default; `_hd/_hw/_sr` selectable via
  `DECODE_ATTN_AMDGCN_COMBINE`). Source `model.py:990-1006`, `qk_owned_flash_decode_graph_node.py`. **[high]**
- **Q2 — separate or fused?** **Separate.** Two distinct precompiled graph nodes (tile, then a second
  `Tensor.custom_kernel` combine dispatch). Not fused/hidden. Buffer-identity addressed cache slicing, not fusion.
  Confirmed in `route_fire.program_node_names` (both present). **[high]**
- **Q3 — tile_us / combine_us / fraction / bandwidth / wg by ctx** (split-KV economics, operative S=48; **06-21
  magnitudes — stale vs current route, structurally representative**):

  | ctx | tile_us | combine_us | combine frac | eff GB/s (% of 960) | tile wg | combine wg |
  |----:|----:|----:|----:|----:|----:|----:|
  | 512  | 15.92 | 12.64 | 0.443 | 64.5 (6.7 %) | 384 | 32 |
  | 1024 | 23.32 | 12.60 | 0.351 | 64.7 (6.7 %) | 384 | 32 |
  | 2048 | 36.70 | 12.64 | 0.256 | 64.5 (6.7 %) | 384 | 32 |
  | 4096 | 62.48 | 12.64 | 0.168 | 64.5 (6.7 %) | 384 | 32 |

  The **current-route** combine is fresher-measured at **~6.1 µs** (`b5-combine/latest.json`, base @S32), not 12.6 µs;
  the flat-floor / underoccupancy structure is identical. **[medium — magnitudes 06-21-pinned]**
- **Q4 — combine underoccupies?** **Yes, severely.** 32 workgroups on 96 CUs (`combine_underoccupied=true`, occupancy
  proxy 0.33 wg/CU vs the tile's 4.0). Grid `(32,1,1)` confirms the count in source. **[high]**
- **Q5 — fixed latency floor across ctx?** **Yes.** combine_us is ~constant across ctx (scales only with `S`, not KV
  length — it merges `S` partials per head); latency/occupancy-bound (~64 GB/s = 6.7 % of HBM peak), not bandwidth-bound.
  `combine_fraction` falls 0.443→0.168 only because tile_us grows. **[high]**
- **Q6 — does any route reintroduce `E_49152`?** **Default route: no** (`E_49152_present=false`,
  `buffer_identity_inputs=true`, regression-guard verified). Only the **non-default** fallback
  `DECODE_ATTN_KV_IDENTITY=0` slices `assigned_kv[0,0]/[1,0]` into views that callify materializes as `E_49152`
  (correct, slower, byte-identical) — this slice path is the deliberate "B" A/B comparator. The aggressive probe keeps
  `KV_IDENTITY=1` ⇒ no `E_49152`. **[high]**
- **Q7 — aggressive-probe improvement source?** There is **no realized improvement to attribute** — the aggressive lane
  **FAILS** target at all ctx (`UNPROVEN__THROUGHPUT`, 103.4/101.6/99.1/94.4 vs 104.0/102.1/99.6/95.1, max 0.7 % short
  @ctx4096). It runs the **same** owned route (`route_preserved=true`, owner kernel present, no `E_49152`, **no Q6K knob
  anywhere**). The only differing knob is **`Q4K_GEMV_WARP_PROJ` (=1 aggressive vs =0 candidate)** — a **weight-GEMV
  projection** lever, **not** the tile/combine path — and it does not lift the lane to target. Verdict: at/below the
  reproducibility band = effectively **noise on the same shipped owned route**; **not** a tile-speed, GEMV-flag,
  projection-route, or combine win. (The separate +13–19 % "A beats B" gain is from `DECODE_ATTN_KV_IDENTITY=1`
  buffer-identity vs the slice comparator — not GEMV-proj or combine.) **[high]**
- **Q8 — single-lifecycle / cheaper-combine feasible?** **No actionable bounded W==D win.** Cheaper combine = **built &
  refuted** (B5-lite 2.4× faster → W==D saturates +5.7 %; combine overlaps in-graph). Fused single-lifecycle merge =
  **codegen-walled** (UOp store-group idiom cannot express multi-granularity reduce→broadcast→reduce — `Q8L-2` precedent)
  **and** projected sub-gate (overlap ⇒ even a free/fused combine ≈ +5.7 %). **[high]**
- **Q9 — Amdahl projection vs the W==D gate.** Gate `= (ctx1024 ≥ +5 % OR ctx4096 ≥ +7 %) AND no ctx512/1024 regression`.
  Measured W==D never cleared it on 06-21 (best +5.41 %). The only projection that clears is the free-combine **estimate**
  (+8.58 % @ctx4096; half +6.97 % *misses* the +7 % number) — **explicitly flagged an estimate, then refuted** by the
  B5 +5.7 % measurement. The route as shipped cleared via the **tile-default + buffer-identity** fixes (+13–19 % vs
  slice, 100.6–104 % of llama), a different lever. **No remaining Amdahl path** shows a further *combine* improvement
  clearing the gate. **[high]**
- **Q10 — exhaustion evidence (five convergent).** (1) `B5_COMBINE_LOCAL_PASS_WD_FAIL` — 2.4× cheaper combine → +5.7 %
  saturation, refutes the free-combine projection. (2) aggressive probe `UNPROVEN__THROUGHPUT`, ~0.7–1.5 % short,
  `do_not_promote`, "keep current decode route as shipped default". (3) ctx-slope `DECODE_CTX_SLOPE_NO_ACTION_UNDER_8B_MAXC`,
  residual "below action bar", do_not = [broad_decode_search, default_change, kernel_change]. (4) Legacy capture path
  (`DECODE_ATTN_AMDGCN_TILE=0`) delta +0.00. (5) campaign synthesis: attention exhausted, decode HBM-bound at parity+
  (100.6–104 % of llama), no large structural lever. Unknown-bucket lockstep PROVEN pre+post ⇒ no hidden mis-attributed
  tax. **[high]**

## Acceptance checklist (a two-kernel route is promotable only if all hold)

| criterion | status | evidence |
|---|---|---|
| Correctness (greedy byte-identical) | **MET** | `token_byte_identical=true`; `POST_PARITY_REGRESSION_GUARD_PASS`; tile rmse 2.8e-07 |
| Intended route fires in-model (default, no flags) | **MET** | `model.py:990/999` default `TILE=1`,`KV_IDENTITY=1`; both owned nodes in `program_node_names`; `route_preserved=true` |
| No `E_49152` / hidden materialization on default | **MET** | `E_49152_present=false`, `full_maxc_copy_kernels=[]`, `buffer_identity_inputs=true`; only `KV_IDENTITY=0` fallback reintroduces it |
| Unknown bucket lockstep-closed | **MET** | `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN` pre+post; largest_unknown @ctx4096 ms pre 1.5844 / post 1.5841 / aggressive 1.5846 |
| W==D improves over baseline at all required ctx | **MET** | owned-whole vs **slice** comparator +18.97/17.55/16.11/13.33 % @512/1024/2048/4096; 100.6–104 % of llama |
| Clears promotion threshold + repro band | **PARTIAL** | shipped route cleared (default-on, +13–19 %, wd_spread ~0.3–0.4 %); the **aggressive** non-search envelope does **not** (FAIL all ctx, ~0.7–1.5 % short, `do_not_promote`) |
| Combine economics show tile win not erased | **PARTIAL** | as shipped the tile win survives net-positive (+13–19 %); but the separate combine remains a flat latency floor and cutting it adds ~0 W==D (B5 +5.7 % saturation) — capped, not eliminated |

The two PARTIALs decompose into **PASS for the shipped default route, FAIL only for the aggressive non-search target** —
different gates with **different comparators** (slice route for promotion W==D; `gqa_coop_vec` SSOT for the historical
+7 % B5 gate). **Do not arithmetically combine the +13–19 % and the +7 % figures** — they are against different
references.

## Required evidence (artifacts cited)

- `bench/qk-decode-lifecycle-recheck-bundle/latest.json` → `…/decode-lifecycle-recheck-20260624-200800/decision.json`
  (`DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS`; route_fire both owned nodes, slice absent; `E_49152_present=false`; lockstep
  PROVEN pre+post; W==D 103.0/101.3 @512/1024; A-beats-B +18.97/17.55/16.11/13.33). Snapshot hash-pinned
  `git_head=b814ca61c` (clean tree, ancestor of HEAD).
- `bench/qk-decode-aggressive-target-proof-20260624/decision.json` (`UNPROVEN__THROUGHPUT`, `do_not_promote`) +
  `artifact_snapshot.json` (`git_commit=dee1c19ff`).
- Route-fire: `program_node_names` contains `owned_flash_tile_gqa_whole` **and** `owned_flash_combine`,
  `candidate_kernel_present=true`, `slice_route_absent=true`.
- No-materialization: `E_49152_present=false`, `buffer_identity_inputs=true` (both pre+post in the recheck;
  regression-guard in `docs/decode-campaign-final-synthesis-20260623.md` §Regression guard).
- Unknown-bucket lockstep pre/post: `…/correctness/artifacts/decision.json` (`DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`).
- W==D ctx sweep: lifecycle-recheck `wd` + `current_ctx_A_beats_B`; aggressive vs baseline in
  `docs/decode-aggressive-target-proof-scope-20260624.md`.
- Split-KV economics: `extra/qk_split_kv_economics_audit.py`, `bench/qk-split-kv-economics-audit/latest.json`
  (`COMBINE_TAX_DOMINATES`), `bench/qk-decode-eval/binding_templates.json`.
- B4/B5 combine-tax provenance: `docs/archive/b4-split-kv-combine-tax-result-20260621.md`,
  `…-scope-20260621.md`, `docs/archive/split-kv-economics-audit-result-20260621.md`,
  **`bench/qk-decode-attention-route-b-b5-combine/{wd.json,latest.json,policy_or_wd.json}`**
  (`B5_COMBINE_LOCAL_PASS_WD_FAIL` — the decisive refutation), `structure/Development/session-handoff.md`.

## Provenance / staleness (evidence is CURRENT for HEAD)

HEAD `4bd54cef5` (2026-06-24 23:25). **No commit in `dee1c19ff..HEAD` modifies any of the three route-defining files**
(`tinygrad/llm/model.py`, `extra/qk_owned_flash_decode.hip`, `extra/qk_owned_flash_decode_graph_node.py`) — `git log`
on each over that range is empty (verified). The range contains `[repo]`/`[test]` commits (and removed dead probe
scripts that are **not** runtime route dependencies), so "all docs-only" would be imprecise, but the load-bearing claim
holds: the decode-attention route at HEAD is byte-identical to the route under which the lifecycle-recheck (hash-pinned
`b814ca61c`, clean) and aggressive-proof (`dee1c19ff`) evidence was collected. **Caveat:** the split-KV economics audit
(`2a872d2f1`, 2026-06-21, dirty tree, perf_state=auto) **predates** the current route series — its absolute per-ctx
`tile_us/combine_us` magnitudes (Q3) are stale-but-structurally-representative; the fresher current-route combine is
~6 µs (B5 `latest.json`).

## Adversarial verification

Three independent lenses, all `label_holds=true`:
- **Staleness/provenance** — route code byte-identical to artifacts (corrected the "all docs-only" overstatement to the
  precise per-file claim; confirmed the bundle is hash-pinned, not date-inferred).
- **Bounded-target-still-open** (devil's advocate for `ACTIVE_BOUNDED_TARGET`) — **conceded** after verifying in-repo
  that B5 was built (registered `owned_flash_combine_hw`, rmse ≤ 5e-7, byte-identical) and W==D-refuted; even a free
  combine extrapolates to ~+5.7 % < +7 %.
- **Promotion-gate/acceptance** — confirmed the audit does not conflate "fails aggressive target" with "fails the
  promotion gate vs baseline"; flagged the slice-vs-`gqa_coop_vec` comparator mismatch (folded in above).

## Recommendation / open levers (not part of the two-kernel route)

Keep the current owned two-kernel route as the shipped default; **do not** re-chase the combine (refuted) or attempt a
fused single-lifecycle merge as a speed play (codegen-walled and projected sub-gate). Per the owner correction, decode
work continues as **open levers outside the two-kernel attention boundary**: weight-GEMV/FFN share (the
`Q4K_GEMV_WARP_PROJ` aggressive probe, sub-threshold today), small-op fusion (HBM-bound, needs a W==D gate first), and
native codegen of `v_dot2`/cross-lane/LDS (a capability, not a speed lever). Re-open combine/primitive attribution only
if the throughput target is tightened (the aggressive-proof's recommended follow-up). Any future search row must name
the **full** primitive boundary: tile + combine + cache identity + dispatch lifecycle + correctness + W==D.
