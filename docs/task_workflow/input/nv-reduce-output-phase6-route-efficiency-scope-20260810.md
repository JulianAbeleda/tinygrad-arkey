# NV reduce-output Phase 6 route-efficiency scope (coalesce bodies + scope callify flags)

Date: 2026-08-10
Branch: `nvidia-bringup-20260731`, HEAD `554cbd3f7` (post
`25370bbad` emitter association fix and `554cbd3f7` NO-GO bracket record).
Status: **COMPLETE, campaign re-run NO-GO (2026-08-10, record
`nv-reduce-output-phase6-wall-bracket-record-20260810.md`). Workstreams A and B
landed (18 coalesced fused bodies, callify flags scoped to the reduce-output
route, residual E_32_32_4 flag-leak shifts eliminated), and all three gates
PASS (smoke, exact logits SHA identical to control, census with honest net
-22 programs). The reverse wall bracket still does NOT promote: candidate
5.420 vs control 5.401 ms/token median (-18.5 us, -0.63 tok/s). The fused
route's own materialization penalty outweighs the saved launches, so the
+495.330 us norms row stays unbooked. No policy promotion, no model wiring
change, no correctness contract weakening.**

## 1. Why this scope exists

The 2026-08-10 wall bracket (`nv-reduce-output-wall-bracket-record-20260809.md`)
proved the fused body is CORRECT on NV: exact-logits gate PASS (control and
candidate logits SHA identical `70838f52...`), census gate PASS (54 fused
bodies, rmsnorm_reduce 56 -> 38, epilogues removed 18, q/k reduces untouched,
net +1 program). But the reverse control/candidate/control bracket did NOT
promote:

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control A | 5.4086 | 184.89 |
| candidate | 5.5974 | 178.65 |
| control B | 5.4030 | 185.10 |

Candidate is -188.8 us vs control A and -194.4 us vs control B (negative =
slower), -6.33 tok/s. Promotion needs +50 us/token vs both. The +495.330 us
norms row stays unbooked.

The candidate's own per-program histogram exposes WHY it loses ~192 us:

- 54 fused bodies at 5.7 us median = ~308 us of fused-body kernel time. The
  ordinary graph ran 18 norms as 18 reduces (3.9 us) + 18 epilogues (2.3 us)
  = ~112 us. The fused route does ~2.7x the norm work because it emits ONE
  body per consuming call argument (3 consumers per norm) plus one weight
  materialization each.
- The callify Context flags (`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT`,
  `CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER`) are GLOBAL. They change
  `transform_precompiled_call` behavior for every precompiled function, not
  just the reduce-output route, which shifts the non-norms residual family:
  `E_32_32_4` attention/FFN residual programs move -36/+36/-71/+54 across
  program identities (see the census `callify_redirect_side_effects`).

So the route's overhead has two independent components, and both are
in-scope for this follow-up:

1. **Body multiplicity**: 54 fused bodies + 54 weight materializations where
   the ordinary graph had 36 norm programs (18 reduce + 18 epilogue). The
   hermetic single-consumer STORE/CALL form fuses 2->1 in isolation; the
   production CALL-input route must coalesce to the same 2->1 per norm.
2. **Callify flag scope**: the two ContextVars must gate ONLY the
   reduce-output route's transformations, leaving every other precompiled
   family byte-identical to the control graph.

## 2. Target end state

With both fixes, the candidate census should read:

- `reduce_output_rmsnorm_1_4096` bodies: 18 (one per norm value), not 54.
- Weight materializations: 18 (one per norm), not 54.
- Norms population: rmsnorm_reduce 56 -> 38 (drop 18, consistent with 18
  bodies), epilogues removed 18, q/k reduce roles untouched (36).
- Non-norms populations: byte-identical counts to control, INCLUDING the
  residual family (no `callify_redirect_side_effects`).
- Net program delta vs ordinary: 0 (18 bodies + 18 weight mats replace 36
  ordinary programs 1:1), and no +1/+72 residue.
- Exact-logits gate: unchanged PASS (logits SHA identical to control).

Wall expectation (to be measured, not claimed): eliminating 36 duplicate
fused bodies removes ~36 x (5.7 us body + weight mat + launch overhead) of
kernel time, and restoring the residual family removes the non-norms shift
cost. The bracket decides; the row books only if the candidate beats BOTH
controls by >= 50 us/token with identical token streams.

## 3. Workstream A: coalesce one fused body per norm

Mechanism today: `_lower_c6_call_input` in
`tinygrad/schedule/rangeify.py` rewrites each consumer CALL argument whose
carrier matches `CONTIGUOUS(RESHAPE(MS(REDUCE_OUTPUT)))` independently: each
matching argument gets its own fresh output buffer, its own fused body
(`lower_reduce_output_store`), and its own weight materialization
(`w_buffer.after(w_buffer.store(weight))`). The same norm value reaches
multiple consumers (q/k/v projections, or gate/up/down), so one norm becomes
3 bodies.

Fix: make the lowering graph-level instead of per-argument. Before the
per-CALL rewrite, collect every C6-chain CALL argument in the sink, group by
the underlying REDUCE_OUTPUT marker identity, and emit ONE fused body into
ONE fresh output buffer per unique marker. Every consumer argument in the
group rewrites to the same `out_buf.after(fused)` dependency-bearing view.
The weight materialization is emitted once per marker and shared by all
consumers of that norm.

Guards:

- Grouping is by marker UOp identity (same `REDUCE_OUTPUT` marker node), not
  by value; two distinct norm instances never share a body.
- The fused body's output buffer must be fresh (never the norm input), and
  the in-place aliasing guard from `14f0e3297` stays: consumers read the
  AFTER dependency, so the body cannot clobber a buffer other consumers read.
- Any argument that cannot prove the C6 chain still fails closed to the
  ordinary graph (no behavior change for non-matching args).
- The exact-output contract is unchanged: the emitted body is the same
  `emit_reduce_output` body already proven bitwise-equal, just emitted fewer
  times.

Acceptance for A (CPU, hermetic):

- A unit test builds a graph with one norm marker consumed by 3 CALLs and
  asserts exactly ONE fused body + ONE weight materialization in the
  lowered schedule, with all 3 consumers reading the same output buffer.
- Existing tripwires green: `test_generic_reduce_output.py`,
  `test_reduce_output_rmsnorm.py`, `test_nv_reduce_output_primitive_ab.py`.

## 4. Workstream B: scope callify flags to the reduce-output route

Mechanism today: `tinygrad/callify.py` consults the two ContextVars in
`transform_precompiled_call`, `_precompiled_output_redirect`,
`_collapse_owned_invocation_input_contiguous`,
`_bind_reduce_output_invocation_inputs`, and the typed-semantic input
producer path. With either flag on, `transform_to_call` also runs the
top-down `pm_precompile_function_boundary` pass over the whole graph. Every
precompiled family (including the residual `E_32_32_4` add/cast programs)
therefore transforms differently than the control arm, changing program
identity and launch counts.

Fix: gate the flag-sensitive behaviors on the presence of the reduce-output
route in the FUNCTION being transformed, instead of on the global
ContextVars. Concretely:

- `transform_precompiled_call` only applies owned-redirect/output-slot
  behavior when the FUNCTION body contains a `REDUCE_OUTPUT` marker with
  `owned_contiguous_candidate` (or a typed-semantic producer marker); all
  other precompiled functions take exactly the control-arm path.
- The top-down boundary pass and `pm_typed_semantic_call_input` skip
  functions/calls whose bodies contain no reduce-output marker; the
  bottom-up ordinary pass (which runs unconditionally) is the only pass
  that touches them.
- The `_require_candidate_callify_flags` harness check stays, but the flags
  now only alter reduce-output-bearing boundaries; the census
  `callify_redirect_side_effects` map must come back EMPTY.

Acceptance for B (CPU, hermetic):

- A unit test runs the candidate context on a graph with a precompiled
  residual function that has NO reduce-output marker and asserts its
  transformed identity equals the control context's identity.
- Census side effects map empty on CPU hermetic shapes; GPU census re-check
  under lock before any bracket arm.

## 5. Re-run campaign (gated on A + B green)

Same harness, same protocol as the 08-10 record:
`extra/llm_research/decode/nv_reduce_output_primitive_ab.py --mode ab`,
depth 512, count 32, reps 5, settled-continuous, fresh process per arm,
GPU bench lock held per child only (the harness self-serializes; do NOT wrap
the orchestrator in an outer flock). Gate order and hard stops unchanged:
smoke -> exact logits -> census -> reverse wall bracket. Promotion still
requires candidate >= +50 us/token vs BOTH controls with identical token
stream hashes.

## 6. HARD STOP and rules

- No policy promotion: `decode-reduce-output-rmsnorm-route-policy.json`
  stays `promoted_targets: []`; no model wiring change; no default flip.
- No M4/M5/Path-3/M3 record changes; no changes to `decode_routes.py`,
  `qk_primitives.py`, or the shared-Q8 promotion record.
- Exact-output contract never weakened: any candidate whose logits SHA
  differs from control is a hard stop, and the tripwire tests must catch it
  on CPU before any GPU arm.
- Every GPU arm a fresh process under lock; no outer flock around the
  orchestrator (the 08-10 campaign's smoke failure was the outer flock
  blocking child locks, confirmed empirically).
- `[nv]` prefix for code + tests in one commit; `[docs]` for records;
  commit and push after each milestone.
- Scratch in /tmp only (disk ~99% full); committed artifacts are small JSON.

## 7. Deliverables

1. Workstream A implementation + hermetic tests (one `[nv]` commit).
2. Workstream B implementation + hermetic tests (one `[nv]` commit).
3. CPU gate green; GPU census re-check under lock (candidate: 18 bodies,
   no non-norms side effects, exact logits PASS).
4. Full wall bracket campaign; record + artifacts committed and pushed
   (`[docs]`), verdict BOOKED or NO-GO with exact evidence.
