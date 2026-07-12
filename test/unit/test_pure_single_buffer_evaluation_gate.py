import pytest

from extra.qk.prefill.pure_single_buffer_evaluation_gate import (
  EvaluationAuthorities, canonical_candidate_hash, evaluate,
)

EXPECTED_BOLTBEAM_ANCHOR_HASH = "81c27275d1aad1bb8147c5c5cdaa8000e9375e81f3d085b49d62064a731313d6"


def _payload():
  # Exact wire descriptor emitted by BoltBeam data/full_kernel_candidates.json.
  # The fixed hash above makes cross-repository drift fail rather than silently
  # creating a second candidate contract.
  return {
    "schema_version": "boltbeam.full_kernel_candidate.v1",
    "workload": {"profile": "qwen3_8b_q4k_m_gfx1100", "role": "ffn_gate_up",
      "shape": {"m": 512, "n": 12288, "k": 4096},
      "dtypes": {"a": "fp16", "b": "fp16", "c": "fp16", "accumulator": "fp32"},
      "layout": {"a": "row_major", "b": "transposed_row_major", "c": "row_major"},
      "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}},
    "schedule": {"tile": {"m": 128, "n": 128, "k": 32}, "waves": {"m": 4, "n": 2}, "threads": 256,
      "lane_ownership": "rdna3_wmma_f32_16x16x16_f16_lds2_static", "cooperative_load": {
        "a": {"lane_mapping": "cooperative_row_stride_64_b128", "vector_width": 8, "alignment": 16},
        "b": {"lane_mapping": "cooperative_row_stride_64_b128", "vector_width": 8, "alignment": 16}},
      "lds": {"windows": {"a": [0, 10240], "b": [10240, 20480]}, "strides": {"a": 80, "b": 80},
              "padding": 16, "banks": 32, "store_vector_width": 8, "load_vector_width": 8},
      "pipeline": {"buffer_count": 1, "stage_count": 1, "epoch_graph": [
        {"epoch": "body", "slot": 0, "produce": ["a", "b"], "wait": ["global", "lds"],
         "barrier": "before_fragment_load", "consume": ["a", "b"]}]},
      "wmma": {"instruction_family": "wmma_f32_16x16x16_f16",
               "fragment_layout": "rdna3_wmma_f32_16x16x16_f16_lds2_static",
               "accumulator_ownership": "wmma_accum_wm_x_wn_8_vgprs"},
      "dependency_policy": {"waitcnt": {"vm": 0, "lgkm": 0},
                            "barriers": ["before_fragment_load", "after_wmma_before_slot_reuse"]},
      "residency": {"preload": ["a", "b"], "resident": ["accumulator"], "reuse": {"a": 4, "b": 2}},
      "epilogue": {"lane_mapping": "wmma_accumulator_scalar_b16", "vector_width": 1},
      "numerical_mode": "ieee_fp16_acc_fp32"},
    "static_constraints": {"max_lds_bytes": 65536, "max_vgpr_per_thread": 256, "allow_spill": False},
    "applicability": {"exact_shape": True, "profiles": ["qwen3_8b_q4k_m_gfx1100"],
      "roles": ["ffn_gate_up"], "targets": ["AMD:gfx1100:wave32"]}}


def _authorities(identity, calls=None):
  calls = [] if calls is None else calls
  def ordinary(name):
    def run(_payload, got): calls.append(name); return {"canonical_identity": got, "status": "pass"}
    return run
  def binding(_payload, got):
    calls.append("route_binding")
    return {"canonical_identity": got, "status": "pass", "route_binding_complete": True,
            "route_id": "pure.single_buffer.anchor", "selected_route_id": "pure.single_buffer.anchor",
            "runtime_binary_matches_candidate": True, "strict_pure": True, "fallback_used": False}
  return EvaluationAuthorities(ordinary("static_legality"), ordinary("compile_resources"), binding,
                               ordinary("full_output_correctness"), ordinary("kernel_timing"))


def test_complete_gate_composes_all_authorities_in_order():
  payload, calls = _payload(), []
  identity = canonical_candidate_hash(payload)
  assert identity == EXPECTED_BOLTBEAM_ANCHOR_HASH
  report = evaluate(payload, identity, _authorities(identity, calls))
  assert report["passed"] is True and report["blocked_at"] is None
  assert calls == ["static_legality", "compile_resources", "route_binding", "full_output_correctness", "kernel_timing"]


def test_gate_fails_closed_until_route_binding_exists_and_skips_expensive_stages():
  payload, calls = _payload(), []
  identity = canonical_candidate_hash(payload)
  auth = _authorities(identity, calls)
  auth = EvaluationAuthorities(auth.static_legality, auth.compile_resources, None,
                               auth.full_output_correctness, auth.kernel_timing)
  report = evaluate(payload, identity, auth)
  assert report["passed"] is False and report["blocked_at"] == "route_binding"
  assert calls == ["static_legality", "compile_resources"]


@pytest.mark.parametrize("mutation,error", [
  (lambda p: p["schedule"]["pipeline"].update(buffer_count=2), "single-buffer"),
  (lambda p: p["schedule"]["pipeline"].update(stage_count=2), "one pipeline stage"),
  (lambda p: p["schedule"]["lds"]["windows"].update(b=[10000, 20480]), "overlap"),
  (lambda p: p["schedule"]["lds"]["strides"].update(a=96), "strides"),
  (lambda p: p["schedule"]["lds"]["windows"].update(a=[0, 10000]), "tile dimensions"),
  (lambda p: p["static_constraints"].update(max_lds_bytes=20000), "exceeds"),
  (lambda p: p["workload"]["shape"].update(m=1024), "exact anchor shape"),
])
def test_gate_rejects_non_anchor_candidate_before_collectors(mutation, error):
  payload = _payload(); mutation(payload); calls = []
  report = evaluate(payload, canonical_candidate_hash(payload), _authorities("unused", calls))
  assert report["blocked_at"] == "candidate_contract" and error in report["blockers"][0]
  assert calls == []


def test_gate_rejects_hash_mismatch_and_cross_candidate_evidence():
  payload = _payload(); identity = canonical_candidate_hash(payload)
  with pytest.raises(ValueError, match="does not match"): evaluate(payload, "0" * 64, EvaluationAuthorities())
  bad = lambda _payload, _identity: {"canonical_identity": "1" * 64, "status": "pass"}
  report = evaluate(payload, identity, EvaluationAuthorities(static_legality=bad))
  assert report["blocked_at"] == "static_legality"


@pytest.mark.parametrize("field,value", [("route_binding_complete", False), ("runtime_binary_matches_candidate", False),
                                           ("strict_pure", False), ("fallback_used", True)])
def test_route_binding_contract_fails_closed(field, value):
  payload = _payload(); identity = canonical_candidate_hash(payload)
  auth = _authorities(identity)
  def binding(_payload, got):
    row = {"canonical_identity": got, "status": "pass", "route_binding_complete": True,
           "route_id": "pure.single_buffer.anchor", "selected_route_id": "pure.single_buffer.anchor",
           "runtime_binary_matches_candidate": True, "strict_pure": True, "fallback_used": False}
    row[field] = value
    return row
  report = evaluate(payload, identity, EvaluationAuthorities(auth.static_legality, auth.compile_resources, binding,
                                                               auth.full_output_correctness, auth.kernel_timing))
  assert report["blocked_at"] == "route_binding" and report["passed"] is False
