# WC1 failing baseline — severity triage and remediation scope

Date: 2026-07-31
Status: triage complete, remediation scoped, nothing implemented. Branch: `exp` @ `d5292d96b`.

Answers the question the reconciliation deliberately did not: which failures are high concern,
and what the fix is for each. Grounding: `output/wc1-failing-baseline-reconciliation-20260731.md`
(3-commit bisect), `output/wc0-carrier-census-20260731.md` (B1-B8 sites), both re-verified
standalone for every claim below that was not already measured.

## 1. Verdict (direct answer)

Nothing in this baseline is an NVIDIA-path emergency: the fp16 overlay admission work
(S1-S6) is committed and its tests pass. Every failing id is AMD/CPU/Metal-facing,
environmental, or test hygiene. Two items are high concern:

1. **The carrier family (46 new + 19 masked ids)** — blocks the AMD fused prefill attention
   render (FA-CTRL records the failure), which blocks the Metal fused-attention port, the
   largest measured prefill lever. Actionable now: WC2 is scoped and WC0/WC1 are done.
2. **The fixture goldens (3 subtests)** — stale at all three bisect commits; the emitter moved
   after `3e152b218` and nobody reviewed the delta. The campaign's silent-movement detector is
   currently blind.

Everything else is medium (3 real defects, each one test- or graph-level, each with a concrete
fix below) or low (environmental reds that should be skips).

## 2. Triage table (measured at HEAD, DEV=NV, this box)

| bucket | ids | severity | verdict |
| --- | ---: | --- | --- |
| B1 carrier defect (new since `0e41c260d`) | 46 | HIGH | blocks AMD render + Metal port; WC2 action |
| fixture SHA mismatch (stale at all 3 commits) | 1 test / 3 sub | HIGH | un-reviewed emitter movement since `3e152b218` |
| rangeify `find_bufs` false cycle | 10 | MEDIUM | real pre-existing defect; PARAM cannot cycle |
| devectorizer lane-store contract | 1 | MEDIUM | code/test shipped disagreeing in `90e93875c` |
| order-dependent flaky (TRACEMETA) | 1 | MEDIUM | test isolation; passes standalone (verified) |
| no-AMD device + admission-gate | 9 + 2 | LOW | cannot pass on this box; convert to skips |
| Metal env (`libSystem.dylib`) | 20 | LOW | cannot pass on this box; convert to skips |
| assertions (timing/env-sensitive) | 6 | LOW | triage individually; likely test-side |
| `ops.py:1691` validation + softmax/WMMA stragglers | 1 + 4 + 3 | LOW | pre-existing reds; most fold into carrier fix |

Baseline hygiene note: the task's "72" and the campaign's "~114" are not reproducible as stated;
measured HEAD = 116 failed (NV), of which 47 are carrier-verification. Diff failing-id sets,
never counts (already the campaign rule).

## 3. HIGH-1 — carrier family: execute WC2 (already scoped)

Root cause chain (probe-verified, WC0): `softmax.py:54` creates `AMD_ROW_SOFTMAX_SLOT`
typed scalar `float` with shape `(8,)`; `UOp.alu` (`uop/ops.py:625-633`) types the MUL by the
last operand's dtype (:631); `expand_native_row_softmax_repack`
(`renderer/isa/amd_attention_abi.py:189`) substitutes a `STACK float.vec(8)`; `spec.py:164`
rejects the scalar/vector mix.

Fix shape (per `input/wmma-carrier-shape-awareness-scope-20260731.md`):

- Make the rewrites shape-aware, not the paths. The slot must carry the carrier dtype (B2), the
  ALU typing must follow the carrier (B1), and the symbolic rules (B5/B6) must distinguish a
  complete carrier from a projected lane by shape, not dtype identity.
- Derive the width from `tc` — replay `binary_axis_count` (`codegen/opt/kernel_lds.py:58`) and
  `tc_upcast_axes` (`codegen/opt/postrange.py:505-559`). No new registry, no new fact source.
- Fail closed on any descriptor family the code cannot resolve.
- **No broadcasting** — the validation doc forbids it by name; the fix is shape-awareness.

Done signal (all three, else stop and report):

1. FA-CTRL flips from recorded failure to emitting hashes.
2. Six packed-WMMA hashes byte-identical (`pg2_amd_all_routes_rendered_source_equality.py`).
3. The failing-id set shrinks by exactly the predicted carrier ids (46 new + 19 masked, plus the
   4 softmax + 3 residency stragglers that fail the same dtype check). Any id that remains with
   a carrier signature after the fix means the scope missed a site.

Predicted remaining bucket-1 signature delta: zero. If the AMD 8B e2e first-token digits move
(not runnable here, but checkable on an AMD box), stop and report; RDNA3's carrier is 8 before
and after, so any numeric movement is a rewrite, not shape-awareness.

## 4. HIGH-2 — fixture goldens: attribute, review, then regenerate

Verified live: `tc_16x16x16_unrolled` records `instruction_count: 104` and the emitter now
emits 121 (mnemonic SHA `834da5...` vs `cfdb6c...`). The test file's goldens were last written
at `3e152b218`; the movement is later and unreviewed.

Fix (test-only, `[test]` commit):

1. Bisect the emitter movement between `3e152b218` and `0e41c260d^` (compile-only, cheap).
2. Review the diff that moved the output. If it was intentional (e.g. the manifest move),
   record it; if it was accidental, fix the emitter first.
3. Regenerate the goldens as a reviewed commit. Regenerating blind is the one banned action:
   it blesses the movement the golden exists to detect.

Do not fold this into the carrier commit; it is an independent verification-integrity fix.

## 5. MEDIUM — the three real defects outside the carrier

### 5.1 rangeify `find_bufs` false cycle (10 ids)

Verified live: `test_composite_scalar_loop.py::test_semantic_composite_scalar_loop_q16_kv16_hd64_cpu`
raises `cycle detected while indexing UOp(Ops.PARAM, dtypes.half, arg=ParamArg(4, ...))` at
`rangeify.py:859`.

Defect: the guard records the first producer op that indexes a buffer and raises when a second
INDEX has a different producer op. That conflates "consumed by two producers" with "cycle".
A `PARAM` is an input; it has no in-range definition, so two-producer indexing of a PARAM is
cycle-free by construction.

Fix (`[codegen]`, one commit):

- Restrict the conflict check to buffers that can be defined in-range (`Ops.BUFFER` / local), or
  replace the heuristic with a genuine dependency-cycle check on the buffer graph.
- Add a unit test: a `PARAM` indexed by two distinct producer ops in one graph must not raise.

Predicted delta: −10 ids, all CPU-side semantic-attention lowering; no AMD/NV effect.

### 5.2 devectorizer lane-store contract (1 id)

Verified live: `test_devectorizer_output_safety.py::test_distinct_gep_store_keeps_ordinary_lane_store_semantics`
fails at `:126`. The commit `90e93875c` changed `codegen/late/devectorizer.py` (+248 lines) and
added this test in the same commit; the test and the code disagree on the GEP-store inversion
contract: the test asserts the store keeps the original concrete INDEX untouched
(`lowered.src[0] is target.src[0]`) while the code now emits a GEP of a widened vector LOAD on
the store side.

Fix (`[codegen]` or `[test]`, one commit, adjudicate first):

- The test's comment states the intended contract: inversion moves the lane map to the value and
  must not widen the output buffer identity. If that contract is still load-bearing, fix
  `load_store_folding` (`devectorizer.py:296`).
- If the widened GEP-store is the new intended shape, update the test to the new contract and
  prove ordinary lane-store semantics are preserved by the surrounding devectorizer suite.

Predicted delta: −1 id either way.

### 5.3 TRACEMETA order-dependent flaky (1 id)

Verified live: `test_graph_admission.py::test_runtime_tracemeta_context_controls_explicit_metadata_without_import_time_mode`
passes standalone, fails in-suite. Global TRACEMETA context leaks across tests.

Fix (`[test]`): isolate the context in the test (fixture-scoped setup/cleanup) so suite order
cannot matter. No production code.

## 6. LOW — environmental reds become skips (29 + 2 ids)

The no-AMD (9), admission-gate (2), and Metal (20) failures cannot pass on this box. Per the
cross-platform model agreed earlier (skip on missing capability, CI matrix, xfail pinned for
known-expected):

- Convert to `skipif` with a capability reason using the repo's existing idioms:
  `test_lowering_baseline.py:6,125` (`_has_amd()`), `test_metal_graph.py:177`
  (`skipUnless(Device.DEFAULT == "METAL")`), `test_amd_isa_integer_wmma_hardware_correctness.py:26`
  (`/dev/kfd`).
- One `[test]` commit; the skip reason names the missing capability, never "fails on CI".

The 6 heterogeneous assertions and the `ops.py:1691` validation failure get individual triage
in a later pass; none block the carrier or the port.

## 7. Execution order

1. **WC2 carrier shape-awareness** (HIGH-1) — the only load-bearing work; unblocks AMD render.
2. **Fixture bisect + reviewed regeneration** (HIGH-2) — independent, can run in parallel.
3. **rangeify `find_bufs`** (5.1) — independent, CPU-side.
4. **devectorizer contract** (5.2) — independent, 1 test.
5. **TRACEMETA isolation** (5.3) — test-only.
6. **Env skip conversion** (6) — test-only, can land any time.

Each item is one commit, one owning prefix, no pushes without instruction.

## 8. Bans

- No broadcasting to paper over the carrier mismatch (named by the validation doc).
- No dtype authority cleanup (D1/D2/D3) interleaved — that scope is separate and unscheduled.
- No blind golden regeneration.
- No new registry or fact source; `tc` is the fact source.
- NFC claims must be byte-proven, never asserted.
- No mixing a functional commit with a test-only one.

