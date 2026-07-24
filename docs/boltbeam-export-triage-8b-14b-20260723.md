# BoltBeam Export Triage — 8B / 14B (2026-07-23)

## Corrected problem statement

The compile stack works — this is **not** a scheduler / kernel-affinity / register-pressure
bug. The production shared flash-attention kernel compiles clean (254 VGPR, 0 spills, 0 scratch)
for the capture geometries. The actual open failure is downstream: **the end-to-end BoltBeam
export path fails for production runs on 8B and 14B *after* compilation succeeds** — it does not
produce a valid artifact.

Perf work (occupancy / VGPR / roofline) is explicitly OUT OF SCOPE here and must stay separate.
Prior levers in that space are already measured dead ends: cutting VGPR regressed perf 1.4–2.5%
(`ATTENTION_COMPACT_VGPR_LEASE_NEGATIVE_20260723.md`), occupancy needs ≤128 VGPR not ≤192
(`SHARED_ATTENTION_LIVE_STATE_RESIDENCY_LEDGER_20260723.md`), and the rotating-PV probe that
chased that lever has been retired (see the rewrite handoff + primitive results docs).

## Three-way classifier

The failure is exactly one (or a different one per model) of these buckets. Prove which before
any fix.

1. **Numerics (functional correctness).** Kernel runs and produces values, but attention output
   is wrong/unstable before artifact write; BoltBeam drops it as invalid, so no artifact.
   - Check: compare pre-BoltBeam prefill logits/output against fp32 / known-good CPU reference;
     assert finite + bounded per-token error.

2. **Schema / format (artifact contract).** Compute may be fine, but the emitted JSON/KA blob is
   malformed / missing fields / wrong keys or types / stale schema version. BoltBeam parser
   silently rejects → "no artifact."
   - Check: capture the raw produced artifact bytes + metadata and strict JSON-schema-diff
     against the expected KA schema, per model.

3. **Geometry / routing (shape handling).** Compile says the kernel exists, but real 8B/14B
   input shapes don't traverse the same attention→BoltBeam branch the small cases do — a variant
   compiles but the right route is never fed into the BoltBeam runtime.
   - Check: log graph/rewrite route selection + shape/mask tags (batch, seq, nheads, hd, kv_len)
     at all capture points; confirm both models hit the same prefill-attn + artifact-emit flow
     with identical capture graph IDs and non-empty artifact-writer state.

## Per-model triage readout

### 8B (fp16-overlay route) — fails now
- **Numerics:** kernel produces values but attention output wrong before artifact write.
  Check: pre-BoltBeam prefill logits vs fp32/CPU known-good; assert finite + bounded error/token.
- **Schema/format:** KA payload/schema version or field layout for 8B malformed/incomplete.
  Check: dump emitted artifact blob + metadata, strict-diff vs expected KA schema.
- **Geometry:** 8B capture compiles but route doesn't use the intended BoltBeam attention path.
  Check: log route selection + shape tags (batch, seq, nheads, hd, kv_len) at capture time.

### 14B (bounded-packed route) — fails now
- **Numerics:** larger hidden/state shape amplifies reductions; wrong m/l/acc update or
  vectorization path → unstable outputs. Check: per-step prefill vs reference; probe NaN/Inf,
  scale mismatch, overflow around softmax reductions.
- **Schema/format:** 14B payload may miss expected large-model fields (e.g. slot/shape
  descriptors) → parser rejects. Check: serialize + validate model-specific metadata and KA
  schema keys/types before ingestion.
- **Geometry:** 14B shape path compiles but isn't end-to-end resolvable (longer sequence / tiled
  slot shape diverges after rewrite). Check: confirm both models hit the same prefill-attention
  rewrite + artifact-emit branch, identical capture graph IDs, non-empty artifact-writer state.

## Execution plan (Claude action form)

Keep perf work separate. Prove the bucket FIRST by running minimal repros for 8B and 14B, in
this order, **stop on the first confirmed failure class per model**:

1. **Numerics probe** — print pre-BoltBeam prefill checksum + max-error diff vs known-good
   reference. (Unblocked: reuse the harness's independent NumPy reference, `_numeric` in
   `extra/qk/generate_shared_attention_captures.py`.)
2. **Schema probe** — dump the serialized BoltBeam payload/metadata bytes; JSON-schema-diff vs
   the expected KA schema for each model. (Needs the export-path serialization point — being
   mapped.)
3. **Geometry probe** — dump the final attention-route decision + shape/mask tags at every
   capture point; confirm 8B and 14B hit the same prefill-attn→artifact branch. (Needs the
   route-decision point — being mapped.)

Sequencing constraint: run probe 1 only AFTER the rotating-PV retirement lands (it edits
`wmma.py`, which the production numeric path also uses — running mid-edit gives a muddy result).
The retirement's own gate (production still 254 VGPR / 0 spills) also confirms tree consistency.

## RESOLVED — verified diagnosis (2026-07-23)

The three-way classifier resolved to **bucket (c) geometry/routing — but at the POLICY-PLUMBING
layer, not shape support or numerics.** Evidence (all verified in code, not inferred):

- **Not numerics.** The v2 proof (`docs/artifacts/shared-attention-m10e1-20260723/shared_attention_proof.json`,
  schema `tinygrad.shared_attention_proof.v2`, status PASS) contains 4 correct captures —
  8B-first/prefix and 14B-first/prefix — max abs err ~3.5e-05. The kernel is numerically correct.
- **Not BoltBeam-side schema rejection.** BoltBeam never receives an artifact to reject.
  `grep -rl "shared_attention" BoltBeam/` is empty, and `boltbeam/policy/route_manifest.py` has
  **no route entry** for a prefill flash-attention kernel (only decode-attention and attn_qo/attn_kv
  GEMM roles). BoltBeam doesn't know this kernel exists.
- **Root cause 1 — the route switch is never turned on.** `tinygrad/llm/model.py:605-613` runs
  `shared_prefill_attention` only when `self.config.prefill_tc_attn` is True. That flag comes from
  `shared_attention_proven_eligible` (`tinygrad/llm/prefill_policy.py:23`), which requires a composite
  `shared_attention_proof` mapping: status PASS + `target` + `geometry` + an embedded v2 `artifact`
  (4 captures) + all 8 flags True (`correctness, score_resident, qk_wmma, pv_wmma, model_8b_prefill,
  model_14b_prefill, decode_nonregression_8b, decode_nonregression_14b`). **This composite object is
  assembled nowhere in production code** — only in `test/unit/test_shared_prefill_policy.py`. The real
  policy path (`select_prefill_runtime_policy` → memory-adaptive cache) never attaches it (confirmed
  against the on-disk cached policy: keys `selected_candidate_id`, `strategy` only). So
  `prefill_tc_attn` is always False and the kernel is **dead code on every real load** — hence no
  whole-prefill run and no artifact.
- **Root cause 2 — the admission gate's VGPR cap forbids the working kernel.**
  `extra/qk/shared_attention_promotion.py:53` requires `1 <= vgpr <= 192` (plus 0 spills/scratch).
  Production is **254 VGPR** — so even a fully-assembled proof would be rejected on VGPR. This ≤192 cap
  is the target the retired rotating-PV probe was chasing; it is unjustified per the residency ledger
  (occupancy needs ≤128) and the compact-lease negative (cutting VGPR regressed perf). The device
  ceiling is 256; 254 fits and runs correctly.
- **Same failure for 8B and 14B** — single all-or-nothing gate requiring both `model_*_prefill` and
  both `decode_nonregression_*` flags simultaneously by design.

**Smallest repro:** build a policy the way a real `from_gguf` load does and observe `prefill_tc_attn`
is False / `shared_attention_proof` absent — equivalently
`test/unit/test_shared_prefill_policy.py::test_shared_attention_is_disabled_without_complete_proof`.

### Real next step (plumbing + one policy decision — NOT kernel work)
1. **Decide the VGPR cap** (`shared_attention_promotion.py:53`). It blocks a correct kernel for no
   measured benefit. Raise it to the real device ceiling (254 fits under 256) or drop VGPR as a hard
   admission gate. This is a strategic call — it means accepting 254 VGPR as the shipping design — but
   all measured evidence supports it. **This unblocks the whole path.**
2. **Build the production collector** that assembles the composite `shared_attention_proof` (target +
   geometry + v2 artifact + the 8 flags) from real evidence and attaches it to the runtime policy, so
   `prefill_tc_attn` flips on. The decode-nonregression flags must be earned from a real decode run,
   not fabricated (the all-or-nothing gate is intentional).
3. **Add a BoltBeam route entry** in `boltbeam/policy/route_manifest.py` for the prefill
   flash-attention kernel so BoltBeam consumes the emitted artifact for 8B and 14B.

## Status (as of writing)
- Rotating-PV probe retirement: in progress (removes probe-exclusive ABI only; production
  untouched; gated on production still compiling 254/0 spills).
- Export-path scout: in progress (mapping the artifact schema + serialization point + route
  selection for probes 2 and 3).
- Next concrete action once both land: run probe 1 (numerics) for 8B and 14B, then 2 and 3.
