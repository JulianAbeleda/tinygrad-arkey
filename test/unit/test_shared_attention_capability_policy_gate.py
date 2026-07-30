"""TG8 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): split
`admission.py:21`'s old `_TC_ATTN_TARGET_REQUIREMENTS = {"backend":"AMD","architecture":"gfx1100"}` hardcoded
target-string gate into its two independent questions, following TG3's shape
(test_qk_capability_policy_gate.py / tinygrad/llm/qk_primitives.py QKPrimitiveCapability):

  (a) capability -- can the resolved target's renderer express tensor-core-based fused QK/PV attention at all?
      Read verbatim from Renderer.tensor_cores via DeviceFacts.capabilities.supports_tensor_cores.
  (b) policy -- is THIS measured shared_attention_proof / bounded_packed_projection_proof promoted for the
      resolved target? The proof records its own `target` and is only valid evidence for the exact
      (backend, architecture) it was measured against.

No AMD hardware is available on this machine (scope section 8): AMD is verified structurally, by feeding a
faked DeviceFacts (real HIPRenderer facts values, not a renderer instance) through the production eligibility
functions, the same pattern test_qk_capability_policy_gate.py already uses.
"""
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.admission import (
  SharedAttentionCapability, bounded_packed_projection_ineligibility_reason, bounded_packed_projection_proven_eligible,
  shared_attention_capability_from_scanned_facts, shared_attention_ineligibility_reason, shared_attention_proven_eligible,
)


def _facts(*, backend, architecture, supports_tensor_cores):
  probe = ProbeRecord("test", "2026-07-15T00:00:00+00:00")
  return DeviceFacts(f"{backend}:0", backend, architecture, None, None,
                     DeviceCapabilities(supports_tensor_cores=supports_tensor_cores), probe, probe)


def _amd_facts(): return _facts(backend="AMD", architecture="gfx1100", supports_tensor_cores=True)
def _metal_facts_with_tensor_cores(): return _facts(backend="METAL", architecture="Apple9", supports_tensor_cores=True)
def _metal_facts_without_tensor_cores(): return _facts(backend="METAL", architecture="Apple9", supports_tensor_cores=None)


def _artifact():
  return {"schema": "tinygrad.shared_attention_proof.v2", "status": "PASS", "passed": True, "captures": [{} for _ in range(4)]}


def _shared_attention_proof(*, backend, architecture):
  return {"status": "PASS", "target": {"backend": backend, "architecture": architecture}, "geometry": {"Bq": 16, "Bkv": 64},
          "correctness": True, "score_resident": True, "qk_wmma": True, "pv_wmma": True,
          "model_8b_prefill": True, "model_14b_prefill": True,
          "decode_nonregression_8b": True, "decode_nonregression_14b": True, "artifact": _artifact()}


def _bounded_packed_proof(*, backend, architecture):
  return {"status": "PASS", "target": {"backend": backend, "architecture": architecture},
          "q4_source_owner": "MODEL_PARAMETER", "fused_dequant_wmma": True, "fp16_qkv_outputs": True,
          "numeric_correctness": True, "memory_cap": True, "allocation_owner_identity": "q4k:selected"}


# ---- Fact: capability is a strict fact, never inferred from the backend/architecture strings ---------------

def test_capability_is_read_verbatim_never_inferred_from_backend_string():
  amd_capability = shared_attention_capability_from_scanned_facts(_amd_facts())
  assert amd_capability.satisfied
  # An AMD-shaped backend/architecture string alone must not satisfy capability -- only the actual
  # supports_tensor_cores fact does.
  amd_no_tc = shared_attention_capability_from_scanned_facts(
    _facts(backend="AMD", architecture="gfx1100", supports_tensor_cores=None))
  assert not amd_no_tc.satisfied
  assert not SharedAttentionCapability("AMD", "gfx1100", False).satisfied
  assert not SharedAttentionCapability("AMD", "gfx1100", None).satisfied  # unreported -- never truthy
  # Metal is not assumed ineligible either: MetalRenderer conditions tensor_cores on Apple7+ (cstyle.py), a
  # real per-renderer fact, so Metal WITH the fact reported satisfies capability just like AMD does.
  assert shared_attention_capability_from_scanned_facts(_metal_facts_with_tensor_cores()).satisfied
  assert not shared_attention_capability_from_scanned_facts(_metal_facts_without_tensor_cores()).satisfied


# ---- Fact: capability and policy are independently decidable, with distinct reasons -------------------------

def test_capability_missing_is_distinct_from_policy_target_not_promoted():
  # Capability holds (Metal reports tensor cores) but the proof was measured for a different target ->
  # policy_proof_target_not_promoted, never capability_missing.
  proof = _shared_attention_proof(backend="AMD", architecture="gfx1100")
  reason = shared_attention_ineligibility_reason({"shared_attention_proof": proof}, _metal_facts_with_tensor_cores())
  assert reason == "policy_proof_target_not_promoted"

  # Capability is missing (Metal did not report tensor cores) even though the proof's recorded target happens
  # to equal the resolved (backend, architecture) -> capability_missing, never a policy reason.
  proof_metal = _shared_attention_proof(backend="METAL", architecture="Apple9")
  reason2 = shared_attention_ineligibility_reason({"shared_attention_proof": proof_metal}, _metal_facts_without_tensor_cores())
  assert reason2 == "capability_missing"


def test_admitted_when_both_capability_and_promotion_hold_for_a_non_amd_target():
  """Proves the policy channel can affirmatively admit a non-AMD target once its own proof names it -- the
  proof's `target` field IS the promotion record for this route (there is no separate BoltBeam route-manifest
  lookup for this specific hand-measured roofline evidence)."""
  proof = _shared_attention_proof(backend="METAL", architecture="Apple9")
  assert shared_attention_ineligibility_reason({"shared_attention_proof": proof}, _metal_facts_with_tensor_cores()) is None
  assert shared_attention_proven_eligible({"shared_attention_proof": proof}, _metal_facts_with_tensor_cores())


def test_proof_incompleteness_reasons_are_distinct_from_capability_and_policy():
  proof = _shared_attention_proof(backend="AMD", architecture="gfx1100")
  proof["geometry"] = {}
  assert shared_attention_ineligibility_reason({"shared_attention_proof": proof}, _amd_facts()) == "proof_incomplete_geometry"

  proof2 = _shared_attention_proof(backend="AMD", architecture="gfx1100")
  proof2["artifact"] = {"schema": "wrong"}
  assert shared_attention_ineligibility_reason({"shared_attention_proof": proof2}, _amd_facts()) == "proof_incomplete_artifact"

  proof3 = _shared_attention_proof(backend="AMD", architecture="gfx1100")
  proof3["pv_wmma"] = False
  assert shared_attention_ineligibility_reason({"shared_attention_proof": proof3}, _amd_facts()) == "proof_incomplete_roofline_fields"

  assert shared_attention_ineligibility_reason({}, _amd_facts()) == "proof_missing_or_not_pass"


def test_bounded_packed_projection_capability_and_policy_are_independently_decidable():
  proof = _bounded_packed_proof(backend="AMD", architecture="gfx1100")
  assert bounded_packed_projection_ineligibility_reason({"bounded_packed_projection_proof": proof}, _amd_facts()) is None
  assert bounded_packed_projection_proven_eligible({"bounded_packed_projection_proof": proof}, _amd_facts())

  # capability missing (no tensor cores reported), policy (target) would otherwise match
  proof_metal = _bounded_packed_proof(backend="METAL", architecture="Apple9")
  reason = bounded_packed_projection_ineligibility_reason({"bounded_packed_projection_proof": proof_metal},
                                                          _metal_facts_without_tensor_cores())
  assert reason == "capability_missing"

  # capability satisfied, policy target mismatched (proof recorded for AMD, resolved target is Metal)
  proof_amd = _bounded_packed_proof(backend="AMD", architecture="gfx1100")
  reason2 = bounded_packed_projection_ineligibility_reason({"bounded_packed_projection_proof": proof_amd},
                                                           _metal_facts_with_tensor_cores())
  assert reason2 == "policy_proof_target_not_promoted"


# ---- AMD structural non-regression: today's real facts keep exactly today's admitted behaviour --------------

def test_amd_real_facts_admission_is_unchanged_by_the_tg8_split():
  """Structural-only (scope section 8: no AMD hardware here). Real AMD gfx1100 facts (tensor_cores reported
  True, matching tc.get_amd('gfx1100') being non-empty) plus a proof recorded for exactly that target are
  admitted identically to the pre-TG8 hardcoded-equality gate."""
  shared_proof = _shared_attention_proof(backend="AMD", architecture="gfx1100")
  bounded_proof = _bounded_packed_proof(backend="AMD", architecture="gfx1100")
  facts = _amd_facts()
  assert shared_attention_proven_eligible({"shared_attention_proof": shared_proof}, facts)
  assert bounded_packed_projection_proven_eligible({"bounded_packed_projection_proof": bounded_proof}, facts)

  # And today's real production default (no proof at all) still yields no admission, same as before.
  assert not shared_attention_proven_eligible({}, facts)
  assert not bounded_packed_projection_proven_eligible({}, facts)
