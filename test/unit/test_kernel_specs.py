"""LR-060: KernelSpec must be able to emit a manifest row that agrees with the real
extra/qk/route_manifest.json entry for at least one prefill and one decode route -- and the checker
must actually be able to catch a drifted spec (positive control), not just pass by construction."""
from __future__ import annotations

from extra.qk import route_manifest
from extra.qk.decode.flash_decode_attention_spec import describe_flash_decode_attention
from extra.qk.prefill.q4k_prefill_route_spec import describe_q4k_packed_prefill
from tinygrad.codegen.plan import TargetCapabilities
from tinygrad.llm.kernel_specs import AdmissionPredicate, CorrectnessEvidence, KernelSpec, PerformanceEvidence


def _q4k_prefill_spec() -> KernelSpec:
  route_id = "prefill_q4k_direct_tile4x4_default"
  entry = route_manifest.route(route_id)
  geometry = describe_q4k_packed_prefill(rows=512, k=4096, tokens=128, role="ffn_gate_up", output_layout="direct_out")
  shape_guard = entry["shape_guards"][0]  # referenced, not re-typed

  def admits() -> bool:
    geometry.validate()
    return geometry.rows == shape_guard["M"]

  return KernelSpec(
    route_id=route_id,
    target=TargetCapabilities(target=geometry.target),
    geometry=geometry,
    abi_roles=(geometry.role,),
    lowering_plan=geometry.opts,
    resource_contract=None,  # no PipelinePolicy/ResourcePlan is populated for this route yet; see report
    correctness_evidence=CorrectnessEvidence(gate=entry["authority_gate"]),
    performance_evidence=PerformanceEvidence(artifacts=tuple(entry["promotion_artifacts"])),
    admission=AdmissionPredicate(check=admits, shape_guard=shape_guard),
    emitter=lambda: geometry,
  )


def _decode_flash_spec() -> KernelSpec:
  route_id = "decode_flash_live_split_g4_kvboth"
  entry = route_manifest.route(route_id)
  geometry = describe_flash_decode_attention(Hq=32, Hd=128, Hkv=8, MAXC=4096, S=8)
  shape_guard = entry["shape_guards"][0]

  def admits() -> bool:
    geometry.validate()
    return (geometry.tile.Hq == shape_guard["Hq"] and geometry.tile.Hkv == shape_guard["Hkv"]
            and geometry.tile.Hd == shape_guard["Hd"] and geometry.tile.Hq // geometry.tile.Hkv == shape_guard["G"])

  return KernelSpec(
    route_id=route_id,
    target=TargetCapabilities(target=geometry.tile.target),
    geometry=geometry,
    abi_roles=tuple(entry["roles"]),  # not computed from geometry -- see report, field 4
    lowering_plan=(),
    resource_contract=None,
    correctness_evidence=CorrectnessEvidence(gate=entry["authority_gate"]),
    performance_evidence=PerformanceEvidence(artifacts=tuple(entry["promotion_artifacts"])),
    admission=AdmissionPredicate(check=admits, shape_guard=shape_guard),
    emitter=lambda: geometry,
  )


def test_q4k_prefill_kernel_spec_matches_manifest():
  spec = _q4k_prefill_spec()
  assert spec.check_against_manifest() == []


def test_decode_flash_kernel_spec_matches_manifest():
  spec = _decode_flash_spec()
  assert spec.check_against_manifest() == []


def test_checker_catches_a_drifted_correctness_gate():
  """Positive control: prove check_against_manifest actually detects drift, not just passes vacuously."""
  spec = _q4k_prefill_spec()
  drifted = spec.__class__(**{**spec.__dict__, "correctness_evidence": CorrectnessEvidence(gate="not/the/real/gate.py")})
  errors = drifted.check_against_manifest()
  assert any("authority_gate" in e for e in errors)


def test_checker_catches_a_role_not_in_the_manifest():
  spec = _q4k_prefill_spec()
  drifted = spec.__class__(**{**spec.__dict__, "abi_roles": ("not_a_real_role",)})
  errors = drifted.check_against_manifest()
  assert any("abi_roles" in e for e in errors)


def test_checker_rejects_unknown_route_id():
  spec = _q4k_prefill_spec()
  drifted = spec.__class__(**{**spec.__dict__, "route_id": "not_a_real_route_id"})
  errors = drifted.check_against_manifest()
  assert len(errors) == 1 and "not in route_manifest.json" in errors[0]


def test_emitted_kernel_names_reads_the_real_geometry_naming():
  spec = _decode_flash_spec()
  assert spec.emitted_kernel_names() == ("flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128",
                                          "flash_fused_gmax_combine_32_128")
