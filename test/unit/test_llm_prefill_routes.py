import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def clean_prefill_route_env():
  old = {k: os.environ.get(k) for k in ("PREFILL_ROUTE", "PREFILL_QK_DIRECT", "PREFILL_ROUTE_STRICT",
                                        "QK_GENERATED_POLICY_STRICT", "PREFILL_DIRECT_QUANTS",
                                        "PREFILL_DIRECT_TENSORS", "PREFILL_DIRECT_SKIP_TENSORS",
                                        "PREFILL_Q4K_PACKED_LOAD", "PREFILL_Q6K_PACKED_LOAD",
                                        "PREFILL_DIRECT_B_UPCAST", "PREFILL_DIRECT_OUT", "PREFILL_DIRECT_PARTS",
                                        "PREFILL_DIRECT_Q4K_PARTS", "PREFILL_DIRECT_Q6K_PARTS",
                                        "PREFILL_DIRECT_FFN_GATE_UP_PARTS", "PREFILL_DIRECT_FFN_DOWN_PARTS",
                                        "PREFILL_Q4K_Q8", "PREFILL_Q4K_REDUCE_OUT", "PREFILL_Q4K_DIRECT_OPTS",
                                        "PREFILL_Q4K_Q8_ROLES",
                                        "PREFILL_Q4K_DIRECT_EXTRA_OPTS", "PREFILL_Q6K_DIRECT_OPTS",
                                        "PREFILL_Q6K_DIRECT_EXTRA_OPTS", "PREFILL_DIRECT_FFN_GATE_UP_OPTS",
                                        "PREFILL_DIRECT_FFN_GATE_UP_EXTRA_OPTS", "PREFILL_Q4K_DIRECT_SCHEDULE",
                                        "PREFILL_Q4K_WMMA_N_TILE", "PREFILL_Q4K_WMMA_MAX_RAW_ELEMS",
                                        "PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION",
                                        "PREFILL_Q4K_WMMA_TILED_M_TILE", "PREFILL_Q4K_WMMA_TILED_N_TILE",
                                        "PREFILL_Q4K_WMMA_TILED_GROUP_TILE",
                                        "PREFILL_QK_GENERATED_TILE",
                                        "PREFILL_QK_GENERATED_TILE_ROLES", "PREFILL_QK_GENERATED_TILE_MODE",
                                        "PREFILL_QK_GENERATED_TILE_ROWS", "PREFILL_QK_GENERATED_TILE_TOKENS",
                                        "PREFILL_LM_HEAD_ROUTE", "PREFILL_LM_HEAD_DIRECT", "PREFILL_UBATCH",
                                        "PREFILL_COOPERATIVE_MMQ", "PREFILL_COOPERATIVE_MMQ_CANDIDATE",
                                        "PREFILL_COOPERATIVE_MMQ_EVIDENCE", "PREFILL_Q6K_WMMA")}
  for k in old: os.environ.pop(k, None)
  yield
  for k, v in old.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = v


def test_prefill_route_policy_defaults_auto():
  from tinygrad.llm.prefill_routes import prefill_route_policy
  assert prefill_route_policy() == "auto"


def test_prefill_qk_direct_alias_selects_direct_packed():
  from tinygrad.llm.prefill_routes import prefill_route_policy
  os.environ["PREFILL_QK_DIRECT"] = "1"
  assert prefill_route_policy() == "auto"
  assert prefill_route_policy("direct") == "direct_packed"


def test_prefill_q4k_q8_role_filter_is_explicit_and_default_broad():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_role_enabled
  assert prefill_q4k_q8_role_enabled("attn_qo")
  roles = frozenset(("attn_qo", "ffn_gate_up"))
  assert prefill_q4k_q8_role_enabled("attn_qo", roles)
  assert prefill_q4k_q8_role_enabled("ffn_gate_up", roles)
  assert not prefill_q4k_q8_role_enabled("ffn_down", roles)


def test_prefill_route_rejects_unknown_policy():
  from tinygrad.llm.prefill_routes import prefill_route_policy
  with pytest.raises(ValueError):
    prefill_route_policy("chunked")


def test_lm_head_prefill_defaults_lazy_and_keeps_direct_alias():
  from tinygrad.llm.prefill_routes import prefill_lm_head_route_policy
  assert prefill_lm_head_route_policy() == "lazy"
  assert prefill_lm_head_route_policy("direct_packed") == "direct_packed"


def test_lm_head_prefill_route_rejects_unknown_policy():
  from tinygrad.llm.prefill_routes import prefill_lm_head_route_policy
  with pytest.raises(ValueError, match="PREFILL_LM_HEAD_ROUTE"):
    prefill_lm_head_route_policy("eager_everything")


def test_auto_keeps_resident_fp16_when_it_fits():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  assert prefill_route_wants_resident_fp16(est_gb=12.0, budget_gb=18.0, has_direct_packed=True)


def test_auto_skips_resident_fp16_when_direct_packed_exists_and_fp16_exceeds_budget():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  assert not prefill_route_wants_resident_fp16(est_gb=24.0, budget_gb=18.0, has_direct_packed=True)


def test_fp16_policy_keeps_resident_fp16_even_over_budget():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  assert prefill_route_wants_resident_fp16(est_gb=24.0, budget_gb=18.0, has_direct_packed=True, route="fp16")


def test_direct_policy_skips_resident_fp16_for_8b_experiments_too():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  assert not prefill_route_wants_resident_fp16(est_gb=12.0, budget_gb=18.0, has_direct_packed=True, route="direct_packed")


def test_direct_packed_quant_selector():
  from tinygrad.llm.prefill_routes import _direct_packed_enabled_for
  lin = type("Lin", (), {"name": "blk.0.ffn_down.weight"})()
  assert not _direct_packed_enabled_for(lin, "Q4_K")


def test_direct_packed_tensor_selector():
  from tinygrad.llm.prefill_routes import _direct_packed_enabled_for
  lin = type("Lin", (), {"name": "blk.0.ffn_down.weight"})()
  assert not _direct_packed_enabled_for(lin, "Q4_K")


def test_direct_packed_parts_prefers_role_then_quant_then_global():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_parts
  lin = type("Lin", (), {"parts": 1})()
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_gate_up", 512, 17408, 5120)
  assert _direct_packed_parts(lin, spec) == 1


def test_direct_packed_q4_ffn_down_defaults_to_single_part():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_parts
  lin = type("Lin", (), {"parts": 4, "name": "blk.0.ffn_down.weight"})()
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "", 512, 5120, 17408)
  assert _direct_packed_parts(lin, spec) == 1
  assert _direct_packed_parts(lin, spec) == 1


def test_direct_packed_q6_defaults_to_single_part():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_parts
  lin = type("Lin", (), {"parts": 4, "name": "blk.0.ffn_down.weight"})()
  spec = PrefillLinearRouteSpec("direct_packed", "q6k", "", 512, 5120, 17408)
  assert _direct_packed_parts(lin, spec) == 1
  assert _direct_packed_parts(lin, spec) == 1


def test_direct_packed_q4_opts_override_and_extra():
  from tinygrad.codegen.opt import OptOps
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_opts
  lin = type("Lin", (), {"opts": ()})()
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_gate_up", 512, 17408, 5120)
  opts = _direct_packed_opts(lin, spec)
  assert [(x.op, x.axis, x.arg) for x in opts] == [
    (OptOps.LOCAL, 0, 16), (OptOps.LOCAL, 1, 16), (OptOps.UPCAST, 0, 4), (OptOps.UPCAST, 1, 4)]
  # Ambient tuning cannot mutate the promoted candidate descriptor schedule.
  os.environ["PREFILL_Q4K_DIRECT_SCHEDULE"] = "legacy"
  os.environ["PREFILL_Q4K_DIRECT_EXTRA_OPTS"] = "UPCAST:0:4"
  os.environ["PREFILL_Q4K_DIRECT_OPTS"] = "LOCAL:0:16,UPCAST:1:4"
  os.environ["PREFILL_DIRECT_FFN_GATE_UP_OPTS"] = "LOCAL:0:64,GROUP:0:10,UPCAST:1:4"
  opts = _direct_packed_opts(lin, spec)
  assert [(x.op, x.axis, x.arg) for x in opts] == [
    (OptOps.LOCAL, 0, 16), (OptOps.LOCAL, 1, 16), (OptOps.UPCAST, 0, 4), (OptOps.UPCAST, 1, 4)]


def test_prefill_q4k_q8_legacy_gemm_flag_is_rejected():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  with pytest.raises(ValueError, match="PREFILL_Q4K_Q8"):
    prefill_q4k_q8_mode("1")


def test_prefill_q4k_q8_wmma_flag_is_valid_route_env():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode, prefill_route_policy
  assert prefill_route_policy() == "auto"
  assert prefill_q4k_q8_mode("wmma") == "wmma"


def test_prefill_q4k_q8_mmq_direct_flag_is_rejected():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  with pytest.raises(ValueError, match="PREFILL_Q4K_Q8"):
    prefill_q4k_q8_mode("mmq_direct")


def test_prefill_q4k_q8_wmma_tiled_flag_is_valid_but_explicit():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode, prefill_route_policy
  assert prefill_route_policy() == "auto"
  assert prefill_q4k_q8_mode("wmma_tiled") == "wmma_tiled"


def _cooperative_workload(*, role="attn_qo", shape=None, profile=None):
  workload = {"phase": "prefill", "role": role, "quant_format": "Q4_K", "route_id": "fused-q4",
              "shape": shape or {"M": 512, "N": 4096, "K": 4096},
              "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32},
              "capability": "amd.gfx1100.q4k.cooperative.v1"}
  if profile is not None: workload["profile"] = profile
  return workload


def _cooperative_bundle(*, role="attn_qo", shape=None, profile=None):
  from tinygrad.llm.cooperative_mmq_gate import canonical_candidate_identity
  workload = _cooperative_workload(role=role, shape=shape, profile=profile)
  candidate = {"route_id": workload["route_id"], "provenance": "research", "rollback_route": "direct_packed",
               "descriptor": {"m_tile": 16, "n_tile": 16, "k_tile": 256}, "workload": workload}
  evidence = {name: {"passed": True} for name in ("compile", "correctness", "guard", "resources",
                                                    "dynamic_owner_compile", "dynamic_owner_correctness",
                                                    "dynamic_owner_instruction")}
  evidence.update(candidate_identity=canonical_candidate_identity(candidate), fallback_used=False,
                  fallback_status="not_used", emitter_proven=True, source_identity="source",
                  binary_identity="binary", workload=dict(workload))
  return candidate, evidence


def _cooperative_linear(**facts):
  target = {"backend": "AMD", "architecture": "gfx1100",
            "capabilities": {"wave_size": 32}, **facts}
  return SimpleNamespace(_prefill_device_facts=target)


def test_cooperative_q4k_binding_is_blocked_without_proven_emitter(monkeypatch):
  from tinygrad.llm import prefill_routes
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  candidate, evidence = _cooperative_bundle()
  evidence["emitter_proven"] = False
  assert prefill_routes._cooperative_q4k_binding(
    _cooperative_linear(), spec, candidate=candidate, evidence=evidence, enabled=True) is None


def test_cooperative_q4k_binding_requires_exact_runtime_shape(monkeypatch):
  from tinygrad.llm import prefill_routes
  candidate, evidence = _cooperative_bundle()
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 256, 4096, 4096)
  assert prefill_routes._cooperative_q4k_binding(
    _cooperative_linear(), spec, candidate=candidate, evidence=evidence, enabled=True) is None


def test_cooperative_q4k_binding_rejects_nested_fallback_for_generated_loop(monkeypatch):
  from tinygrad.llm import prefill_routes
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_down", 512, 4096, 4096)
  candidate, evidence = _cooperative_bundle(role="ffn_down")
  evidence["fallback"] = {"used": True, "policy": "fail_closed"}
  assert prefill_routes._cooperative_q4k_binding(
    _cooperative_linear(), spec, candidate=candidate, evidence=evidence, enabled=True) is None


def test_cooperative_q4k_binding_rejects_non_object_evidence(monkeypatch):
  from tinygrad.llm import prefill_routes
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  for candidate, evidence in ((None, {}), ([], {}), ({}, None), ({}, [])):
    assert prefill_routes._cooperative_q4k_binding(
      _cooperative_linear(), spec, candidate=candidate, evidence=evidence, enabled=True) is None


def test_cooperative_q4k_binding_without_profile_is_structurally_bound(monkeypatch):
  from tinygrad.llm import prefill_routes
  candidate, evidence = _cooperative_bundle()
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  assert prefill_routes._cooperative_q4k_binding(
    _cooperative_linear(), spec, candidate=candidate, evidence=evidence, enabled=True) == candidate


def test_cooperative_q4k_binding_fails_closed_without_attached_scanned_facts(monkeypatch):
  from tinygrad.llm import prefill_routes
  candidate, evidence = _cooperative_bundle()
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  assert prefill_routes._cooperative_q4k_binding(
    SimpleNamespace(), spec, candidate=candidate, evidence=evidence, enabled=True) is None


def test_cooperative_q4k_binding_compares_candidate_target_to_device_facts(monkeypatch):
  from tinygrad.llm import prefill_routes
  from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
  candidate, evidence = _cooperative_bundle()
  probe = ProbeRecord("test-scan", "2026-07-15T00:00:00+00:00")
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  matching = DeviceFacts("AMD:0", "AMD", "gfx1100", None, None, DeviceCapabilities(wave_size=32), probe, probe)
  wrong_wave = DeviceFacts("AMD:0", "AMD", "gfx1100", None, None, DeviceCapabilities(wave_size=64), probe, probe)
  assert prefill_routes._cooperative_q4k_binding(
    SimpleNamespace(_prefill_device_facts=matching), spec, candidate=candidate, evidence=evidence, enabled=True) == candidate
  assert prefill_routes._cooperative_q4k_binding(
    SimpleNamespace(_prefill_device_facts=wrong_wave), spec, candidate=candidate, evidence=evidence, enabled=True) is None


def test_wrong_profile_and_model_rename_provenance_do_not_block_structural_match():
  from tinygrad.llm import prefill_routes
  from tinygrad.llm.cooperative_mmq_gate import canonical_candidate_identity
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  candidate, evidence = _cooperative_bundle(profile="wrong-profile")
  candidate["workload"]["model_path"] = "/models/renamed-copy.gguf"
  evidence["workload"]["profile"] = "different-wrong-profile"
  evidence["workload"]["model_path"] = "/evidence/original-name.gguf"
  evidence["candidate_identity"] = canonical_candidate_identity(candidate)
  assert prefill_routes._cooperative_evidence_matches(_cooperative_linear(), spec, candidate, evidence)


@pytest.mark.parametrize("field,bad", [
  ("phase", "decode"), ("quant_format", "Q6_K"), ("role", "ffn_down"),
  ("shape", {"M": 512, "N": 4096, "K": 5120}),
  ("target", {"backend": "AMD", "arch": "gfx1200", "wave_size": 32}),
  ("capability", "amd.gfx1200.q4k.cooperative.v1"), ("route_id", "other-candidate")])
def test_wrong_profile_cannot_authorize_structural_mismatch(field, bad):
  from tinygrad.llm import prefill_routes
  spec = prefill_routes.PrefillLinearRouteSpec("direct_packed", "q4k", "attn_qo", 512, 4096, 4096)
  candidate, evidence = _cooperative_bundle(profile="formerly-compatible-profile")
  candidate["workload"][field] = bad
  assert not prefill_routes._cooperative_evidence_matches(_cooperative_linear(), spec, candidate, evidence)


def test_prefill_q4k_q8_packed_ds4_flag_is_valid_research_route():
  from extra.qk.mmq_logical_vocabulary import MMQCandidate
  from extra.qk.mmq_ds4_logical_emitter import packed_ds4_candidate
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  assert prefill_q4k_q8_mode("packed_ds4") == "packed_ds4"
  assert isinstance(packed_ds4_candidate(16, 16, 256, role="test"), MMQCandidate)


def test_prefill_q4k_q8_rejects_unknown_mode():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  with pytest.raises(ValueError, match="PREFILL_Q4K_Q8"):
    prefill_q4k_q8_mode("surprise_tensorcore")


def test_direct_packed_route_spec_exports_runtime_op_spec():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec
  q4 = PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_gate_up", 512, 17408, 5120).runtime_op_spec()
  assert q4.family == "QuantizedLinear"
  assert q4.phase == "prefill"
  assert q4.role == "ffn_gate_up"
  assert q4.weight.format == "Q4_K"
  assert q4.activation.format == "fp16"
  assert q4.shape == {"M": 512, "N": 17408, "K": 5120}
  q6 = PrefillLinearRouteSpec("direct_packed", "q6k", "", 512, 5120, 17408).runtime_op_spec()
  assert q6.role == "unknown"
  assert q6.weight.format == "Q6_K"


class _PrefillTensorStub:
  shape = (1, 512, 256)
  device = "CPU"

  def __getitem__(self, _idx):
    return self

  def cast(self, *_args, **_kwargs):
    return self

  def contiguous(self):
    return self

  def reshape(self, *_args, **_kwargs):
    return self

  def custom_kernel(self, *_args, **_kwargs):
    return (self,)

  def sum(self, *_args, **_kwargs):
    return self

  def transpose(self, *_args, **_kwargs):
    return self


def test_production_route_ignores_environment_when_attachment_selects_exact_baseline(monkeypatch):
  from tinygrad.llm import prefill_routes
  from tinygrad.llm.prefill_route_census import PrefillRouteAttachment
  lin = _q4_prefill_linear()
  lin._prefill_route_attachment = PrefillRouteAttachment(
    "blk.0", "direct_packed", "weight", {"candidate_id": "direct_packed"}, {"backend": "CPU"})
  monkeypatch.setenv("PREFILL_ROUTE", "fp16")
  monkeypatch.setenv("PREFILL_Q4K_Q8", "packed_fused")
  assert prefill_routes._attached_production_route(lin, _PrefillTensorStub()) == "direct_packed"


def test_production_route_missing_or_mismatched_attachment_fails_closed(monkeypatch):
  from tinygrad.llm import prefill_routes
  from tinygrad.llm.prefill_route_census import PrefillRouteAttachment
  x, lin = _PrefillTensorStub(), _q4_prefill_linear()
  assert prefill_routes._attached_production_route(lin, x) is None
  lin._prefill_route_attachment = PrefillRouteAttachment(
    "blk.0", "research-mmq", "weight", {"candidate_id": "direct_packed"}, {"backend": "CPU"})
  monkeypatch.setenv("PREFILL_ROUTE", "direct_packed")
  assert prefill_routes._attached_production_route(lin, x) is None


class _TensorFactoryStub(_PrefillTensorStub):
  @classmethod
  def empty(cls, *_args, **_kwargs):
    return cls()


class _SmallPrefillTensorStub(_PrefillTensorStub):
  shape = (1, 37, 384)


class _Q6PrefillWeight:
  def to(self, *_args, **_kwargs):
    return self


class _Q4PrefillWeight:
  def to(self, *_args, **_kwargs):
    return self


def _q4_prefill_linear(parts=1):
  return SimpleNamespace(
    bias=None, in_features=256, out_features=16, parts=parts, opts=(), name="blk.0.ffn_gate.weight",
    q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight())


def _q6_prefill_linear(parts=1):
  return SimpleNamespace(
    bias=None, in_features=256, out_features=16, parts=parts, opts=(), name="blk.0.ffn_down.weight",
    q6k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q6PrefillWeight())


def _attached_direct_baseline(lin):
  from tinygrad.llm.prefill_route_census import PrefillRouteAttachment
  lin._prefill_route_attachment = PrefillRouteAttachment(
    "invocation", "direct-packed-baseline", lin.name,
    {"candidate_id": "direct-packed-baseline", "strategy": "DIRECT_PACKED_FALLBACK"}, {"backend": "CPU"})
  return lin


def _research_config(**kwargs):
  from tinygrad.llm.prefill_routes import PrefillResearchRouteConfig
  return PrefillResearchRouteConfig(**kwargs)


def test_production_attachment_ignores_all_research_route_env(monkeypatch):
  from tinygrad.llm import prefill_routes
  for key, value in {
    "PREFILL_COOPERATIVE_MMQ": "1", "PREFILL_COOPERATIVE_MMQ_CANDIDATE": "not-json",
    "PREFILL_COOPERATIVE_MMQ_EVIDENCE": "not-json", "PREFILL_QK_GENERATED_TILE": "1",
    "PREFILL_QK_GENERATED_TILE_ROLES": "ffn_gate_up", "PREFILL_QK_GENERATED_TILE_MODE": "research",
    "PREFILL_QK_GENERATED_TILE_ROWS": "16", "PREFILL_QK_GENERATED_TILE_TOKENS": "16",
    "PREFILL_Q4K_Q8": "packed_fused", "PREFILL_Q4K_Q8_ROLES": "ffn_gate_up",
    "PREFILL_Q4K_WMMA_N_TILE": "512", "PREFILL_Q4K_WMMA_MAX_RAW_ELEMS": "1",
    "PREFILL_Q4K_WMMA_ALLOW_GRAPH_EXPLOSION": "1",
    "PREFILL_Q4K_WMMA_TILED_M_TILE": "32", "PREFILL_Q4K_WMMA_TILED_N_TILE": "32",
    "PREFILL_Q4K_WMMA_TILED_GROUP_TILE": "2", "PREFILL_Q6K_WMMA": "1",
  }.items(): monkeypatch.setenv(key, value)
  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes, "_cooperative_q4k_binding", lambda *_args: (_ for _ in ()).throw(
    AssertionError("production reached cooperative research dispatch")))
  monkeypatch.setattr(prefill_routes.qk_ops, "packed_fused_candidate", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("production reached packed-fused research dispatch")))
  calls = []
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated",
                      lambda *_args, **kwargs: calls.append(kwargs["output_layout"]) or SimpleNamespace(output_layout=kwargs["output_layout"]))
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("baseline", spec.output_layout))
  out = prefill_routes.route_prefill_linear(
    _attached_direct_baseline(_q4_prefill_linear()), _PrefillTensorStub(), prefill_graph_gemm=True)
  assert isinstance(out, _PrefillTensorStub)
  assert calls == ["direct_out"]


def test_production_direct_and_gate_up_fail_closed_without_promoted_attachment(monkeypatch):
  from tinygrad.llm import prefill_routes
  monkeypatch.setenv("PREFILL_Q4K_Q8", "packed_fused")
  monkeypatch.setenv("PREFILL_COOPERATIVE_MMQ", "1")
  assert prefill_routes.route_direct_packed_prefill(_q4_prefill_linear(), _PrefillTensorStub()) is None
  gate, up = _attached_direct_baseline(_q4_prefill_linear()), _attached_direct_baseline(_q4_prefill_linear())
  assert prefill_routes.route_prefill_q4k_gate_up(gate, up, _PrefillTensorStub()) is None


def test_direct_packed_q4_request_facts_are_built_from_fake_module():
  from tinygrad.llm.prefill_routes import build_direct_packed_prefill_request

  lin = SimpleNamespace(
    bias=None, in_features=384, out_features=96, parts=1, opts=(), name="custom.layers.7.ffn_gate_proj",
    _prefill_graph_role="mlp_expand", q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight())
  req = build_direct_packed_prefill_request(lin, _SmallPrefillTensorStub(), ubatch=123)
  assert req is not None
  assert req.route_facts == {
    "quant": "Q4_K", "role": "mlp_expand", "M": 37, "N": 96, "K": 384, "bias": False, "ubatch": 123}


def test_direct_packed_q6_shadow_request_facts_are_built_from_fake_module():
  from tinygrad.llm.prefill_routes import select_direct_packed_prefill_shadow_request

  lin = SimpleNamespace(
    bias=object(), in_features=384, out_features=80, parts=1, opts=(), name="toy.block.3.ffn_down",
    q6k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q6PrefillWeight())
  req = select_direct_packed_prefill_shadow_request(lin, _SmallPrefillTensorStub(), ubatch=64)
  assert req is not None
  assert req.route_facts == {
    "quant": "Q6_K", "role": "ffn_down", "M": 37, "N": 80, "K": 384, "bias": True, "ubatch": 64}


def test_direct_packed_request_prefers_carried_route_role_over_ambiguous_name():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_role, build_direct_packed_prefill_request

  lin = SimpleNamespace(
    bias=None, in_features=384, out_features=96, parts=1, opts=(), name="custom.layers.7.attn_k.weight",
    route_role="attn_qo", q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight())

  req = build_direct_packed_prefill_request(lin, _SmallPrefillTensorStub(), ubatch=123)
  assert req is not None
  assert req.role == "attn_qo"
  assert _direct_packed_role(lin, PrefillLinearRouteSpec("direct_packed", "q4k", "", 37, 96, 384)) == "attn_qo"


def test_q4_direct_packed_prefill_default_uses_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_direct_out_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K packed prefill default must not use q4k_gemm_packed_load_direct_out_kernel")))
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K packed prefill default must not use q4k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_attached_direct_baseline(_q4_prefill_linear()), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "direct_out"


def test_q4_direct_packed_prefill_partials_use_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  os.environ["PREFILL_DIRECT_Q4K_PARTS"] = "2"
  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K packed prefill partials must not use q4k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_attached_direct_baseline(_q4_prefill_linear(parts=2)), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "partials"


def test_q4_reduce_out_uses_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_reduce_out_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K reduce_out should use Q4KPrefillRouteSpec")))
  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_attached_direct_baseline(_q4_prefill_linear()), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "direct_out"


def test_q4_generated_tile_flag_is_retired(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  with pytest.raises(RuntimeError, match="PREFILL_QK_GENERATED_TILE was retired"):
    prefill_routes.route_direct_packed_prefill_research(
      _q4_prefill_linear(), _PrefillTensorStub(), config=_research_config(generated_tile=True))


def test_q4_wmma_tiled_small_multitile_uses_scheduler_owned_route(monkeypatch):
  from tinygrad.llm import prefill_routes

  calls = []
  monkeypatch.setattr(prefill_routes.qk_ops, "q8_1_quantize", lambda x: ("xq", "xscales"))

  def one_tile(*_args, **_kwargs):
    calls.append("one_tile")
    raise NotImplementedError("multi tile")

  def scheduler(*_args, **_kwargs):
    calls.append("scheduler")
    return _PrefillTensorStub()

  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_prefill_tensor", one_tile)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_scheduler_tensor", scheduler)
  out = prefill_routes.route_direct_packed_prefill_research(SimpleNamespace(
    bias=None, in_features=256, out_features=32, parts=1, opts=(), name="blk.0.ffn_gate.weight",
    q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight()), _PrefillTensorStub(),
    config=_research_config(q4k_q8_mode="wmma_tiled"))
  assert isinstance(out, _PrefillTensorStub)
  assert calls == ["one_tile", "scheduler"]


def test_q4_wmma_tiled_large_shape_uses_scheduler_owned_route(monkeypatch):
  from tinygrad.llm import prefill_routes

  class _LargePrefillTensorStub(_PrefillTensorStub):
    shape = (1, 512, 5120)

  calls = []
  monkeypatch.setattr(prefill_routes.qk_ops, "q8_1_quantize", lambda x: ("xq", "xscales"))
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_prefill_tensor",
                      lambda *_args, **_kwargs: (_ for _ in ()).throw(NotImplementedError("full role")))
  def scheduler(*_args, **_kwargs):
    calls.append("scheduler")
    return _PrefillTensorStub()
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_scheduler_tensor", scheduler)
  out = prefill_routes.route_direct_packed_prefill_research(SimpleNamespace(
    bias=None, in_features=5120, out_features=5120, parts=1, opts=(), name="blk.0.attn_q.weight",
    q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight()), _LargePrefillTensorStub(),
    config=_research_config(q4k_q8_mode="wmma_tiled"))
  assert isinstance(out, _PrefillTensorStub)
  assert calls == ["scheduler"]


def test_q4_packed_ds4_route_consumes_shared_candidate_and_packer(monkeypatch):
  from tinygrad.llm import prefill_routes
  calls = []
  monkeypatch.setattr(prefill_routes.qk_ops, "packed_ds4_candidate", lambda *args, **kwargs: calls.append(("candidate", args, kwargs)) or "candidate")
  monkeypatch.setattr(prefill_routes.qk_ops, "pack_q8_1_mmq_ds4", lambda *args, **kwargs: calls.append(("pack", args, kwargs)) or ("values", "scales", "sums"))
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_q8_mmq_ds4", lambda *args, **kwargs: calls.append(("emit", args, kwargs)) or _PrefillTensorStub())
  out = prefill_routes.route_direct_packed_prefill_research(
    _q4_prefill_linear(), _PrefillTensorStub(), config=_research_config(q4k_q8_mode="packed_ds4"))
  assert isinstance(out, _PrefillTensorStub)
  assert [entry[0] for entry in calls] == ["candidate", "pack", "emit"]


def test_q4_packed_ds4_reuses_only_the_immediately_shared_activation(monkeypatch):
  from tinygrad.llm import prefill_routes
  calls = []
  monkeypatch.setattr(prefill_routes.qk_ops, "packed_ds4_candidate", lambda *args, **kwargs: "candidate")
  monkeypatch.setattr(prefill_routes.qk_ops, "pack_q8_1_mmq_ds4", lambda *args, **kwargs: calls.append("pack") or ("values", "scales", "sums"))
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_q8_mmq_ds4", lambda *args, **kwargs: _PrefillTensorStub())
  x = _PrefillTensorStub()
  config = _research_config(q4k_q8_mode="packed_ds4")
  prefill_routes.route_direct_packed_prefill_research(_q4_prefill_linear(), x, config=config)
  prefill_routes.route_direct_packed_prefill_research(_q4_prefill_linear(), x, config=config)
  assert calls == ["pack"]


def test_q6_direct_packed_prefill_default_uses_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q6k_gemm_packed_load_direct_out_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q6_K packed prefill default must not use q6k_gemm_packed_load_direct_out_kernel")))
  monkeypatch.setattr(prefill_routes.qk_ops, "q6k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q6_K packed prefill default must not use q6k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q6k_packed_prefill", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q6k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_attached_direct_baseline(_q6_prefill_linear()), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "direct_out"


def test_direct_packed_role_infers_lm_head_for_output_weight():
  # LM-head prefill-route wiring: output.weight previously matched none of the ffn/attn name patterns and fell
  # through to "" (unrouted). A bare object with only `.name` set (no route_role/role carried) exercises the
  # name-based fallback added for the lm_head case.
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_role
  spec = PrefillLinearRouteSpec("direct_packed", "q6k", "", 512, 151936, 4096)
  lin = SimpleNamespace(name="output.weight")
  assert _direct_packed_role(lin, spec) == "lm_head"


def test_direct_packed_role_lm_head_does_not_shadow_attn_output():
  # "attn_output" also contains the substring "output" -- the lm_head fallback must sit after the attn_qo match
  # so blk.N.attn_output.weight keeps resolving to "attn_qo", not "lm_head".
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_role
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "", 512, 4096, 4096)
  lin = SimpleNamespace(name="blk.3.attn_output.weight")
  assert _direct_packed_role(lin, spec) == "attn_qo"


def test_direct_packed_module_role_infers_lm_head_for_output_weight():
  from tinygrad.llm.prefill_routes import _direct_packed_module_role
  lin = SimpleNamespace(name="output.weight")
  assert _direct_packed_module_role(lin) == "lm_head"


def test_direct_packed_module_role_prefers_prefill_graph_role_lm_head():
  # Transformer._prefill_v2_covered tags the installed lm-head primitive with _prefill_graph_role="lm_head"
  # directly (see tinygrad/llm/model.py); that should win over any carried route_role ("output" from the
  # generic model_route_plan install path) without needing the name fallback at all.
  from tinygrad.llm.prefill_routes import _direct_packed_module_role
  lin = SimpleNamespace(name="output.weight", route_role="output", _prefill_graph_role="lm_head")
  assert _direct_packed_module_role(lin) == "lm_head"


def test_q6_direct_packed_prefill_partials_use_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q6k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q6_K packed prefill partials must not use q6k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q6k_packed_prefill", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q6k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_attached_direct_baseline(_q6_prefill_linear(parts=2)), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "direct_out"
