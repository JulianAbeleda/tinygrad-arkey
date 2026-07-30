from types import SimpleNamespace
import ast
import pathlib
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.model_facts import model_facts_from_gguf_metadata
from tinygrad.llm.model_route_plan import build_model_route_plan, primitive_route_entry_for_tensor
from tinygrad.llm.qk_primitives import (
  _install_q4k_primitives, _install_q6k_primitives, Q4KPrimitiveLinear, Q6KPrimitiveLinear,
  QKConfig, QKPrimitiveBudget, QKPrimitiveEligibility, qk_primitive_eligibility_from_device_facts,
  QKPrimitiveCapability, QKPrimitiveRouteAdmission, qk_primitive_capability_from_device_facts,
)
from tinygrad.llm import qk_primitives


QWEN3_LIKE_PROFILES = (
  {"name": "8b-like", "hidden": 4096, "intermediate": 12288, "heads": 32, "kv_heads": 8, "head_dim": 128},
  {"name": "14b-like", "hidden": 5120, "intermediate": 17408, "heads": 40, "kv_heads": 8, "head_dim": 128},
)


def _tensor_rows(profile):
  h, i, kv = profile["hidden"], profile["intermediate"], profile["kv_heads"] * profile["head_dim"]
  return (
    ("blk.0.ffn_gate.weight", (h, i), 12, "ffn_gate_up"),
    ("blk.0.ffn_up.weight", (h, i), 12, "ffn_gate_up"),
    ("blk.0.ffn_down.weight", (i, h), 12, "ffn_down"),
    ("blk.0.attn_q.weight", (h, h), 12, "attn_qo"),
    ("blk.0.attn_output.weight", (h, h), 12, "attn_qo"),
    ("blk.0.attn_k.weight", (h, kv), 12, "attn_kv"),
    ("blk.0.attn_v.weight", (h, kv), 12, "attn_kv"),
    ("blk.1.ffn_down.weight", (i, h), 14, "ffn_down"),
    ("blk.1.attn_v.weight", (h, kv), 14, "attn_kv"),
    ("output.weight", (h, 151936), 14, "lm_head"),
  )


def _meta_for_profile(profile):
  offset = 0
  rows = []
  for name, dims, typ, _role in _tensor_rows(profile):
    rows.append((name, dims, typ, offset))
    offset += 4
  return {"data_start": 0, "tensor_infos": rows}


def _qwen_kv(profile):
  return {
    "general.architecture": "qwen3",
    "qwen3.embedding_length": profile["hidden"],
    "qwen3.feed_forward_length": profile["intermediate"],
    "qwen3.attention.head_count": profile["heads"],
    "qwen3.attention.head_count_kv": profile["kv_heads"],
    "qwen3.attention.key_length": profile["head_dim"],
  }


EXPECTED_PRIMITIVE_DEFAULTS = {
  "blk.0.ffn_gate.weight": (1, ("LOCAL:0:64",), "Q4_K"),
  "blk.0.ffn_up.weight": (1, ("LOCAL:0:64",), "Q4_K"),
  "blk.0.ffn_down.weight": (4, ("LOCAL:0:32",), "Q4_K"),
  "blk.0.attn_q.weight": (1, ("LOCAL:0:64",), "Q4_K"),
  "blk.0.attn_output.weight": (1, ("LOCAL:0:64",), "Q4_K"),
  "blk.0.attn_k.weight": (1, ("LOCAL:0:64",), "Q4_K"),
  "blk.0.attn_v.weight": (1, ("LOCAL:0:64",), "Q4_K"),
  "blk.1.ffn_down.weight": (1, ("LOCAL:0:64",), "Q6_K"),
  "blk.1.attn_v.weight": (4, ("LOCAL:0:32",), "Q6_K"),
  "output.weight": (1, ("LOCAL:0:64",), "Q6_K"),
}


def test_model_facts_route_plan_sets_q4_q6_defaults_from_tensor_facts_for_qwen3_like_tensors():
  for profile in QWEN3_LIKE_PROFILES:
    meta = _meta_for_profile(profile)
    facts = model_facts_from_gguf_metadata(_qwen_kv(profile), meta)
    plan = build_model_route_plan(meta, facts)
    expected_roles = {name: role for name, _dims, _typ, role in _tensor_rows(profile)}

    assert len(facts.tensors) == len(meta["tensor_infos"]), profile["name"]
    for name, dims, typ, _off in meta["tensor_infos"]:
      entry = plan.primitive(name)
      assert entry is not None
      expected_parts, expected_opts, expected_quant = EXPECTED_PRIMITIVE_DEFAULTS[name]
      assert (entry.rows, entry.cols) == tuple(reversed(dims))
      assert (entry.parts, entry.opts) == (expected_parts, expected_opts)
      assert entry.module_path == name.removesuffix(".weight")
      assert entry.role == expected_roles[name]
      assert entry.quant_label == expected_quant
      assert entry.quant_label == ("Q4_K" if typ == 12 else "Q6_K")


def test_route_plan_is_independent_of_legacy_environment_switches(monkeypatch):
  meta = _meta_for_profile(QWEN3_LIKE_PROFILES[0])
  monkeypatch.setenv("DECODE_ROUTE_ATTN_K", "0")
  monkeypatch.setenv("DECODE_ROUTE_ATTN_V", "0")
  monkeypatch.setenv("Q6K_COVER_MORE", "0")
  disabled = build_model_route_plan(meta)
  monkeypatch.setenv("DECODE_ROUTE_ATTN_K", "1")
  monkeypatch.setenv("DECODE_ROUTE_ATTN_V", "1")
  monkeypatch.setenv("Q6K_COVER_MORE", "1")
  enabled = build_model_route_plan(meta)
  assert list(disabled) == list(enabled)


def test_route_plan_rejects_non_block_aligned_quant_shape():
  assert primitive_route_entry_for_tensor("blk.0.ffn_down.weight", 12, 256, 255) is None


def _linear(rows, cols):
  return SimpleNamespace(weight=Tensor.empty(rows, cols, dtype=dtypes.float16), bias=None)


def _install_model():
  return SimpleNamespace(blk=[SimpleNamespace(
    ffn_gate=_linear(256, 256),
    ffn_down=_linear(256, 256),
  )])


def _device_facts(*, backend="AMD", architecture="gfx1100", wave_size=32, supports_warp_shfl_xor=True):
  probe = ProbeRecord("test", "2026-07-15T00:00:00+00:00")
  return DeviceFacts("AMD:0", backend, architecture, None, None,
                     DeviceCapabilities(wave_size=wave_size, supports_warp_shfl_xor=supports_warp_shfl_xor), probe, probe)


def test_qk_eligibility_requires_exact_structural_device_facts_match():
  assert qk_primitive_eligibility_from_device_facts(_device_facts()).eligible
  assert not qk_primitive_eligibility_from_device_facts(_device_facts(backend="amd")).eligible
  assert not qk_primitive_eligibility_from_device_facts(_device_facts(architecture="gfx1100:sramecc+")).eligible
  assert not qk_primitive_eligibility_from_device_facts(_device_facts(wave_size=64)).eligible
  assert not qk_primitive_eligibility_from_device_facts(None).eligible


def test_isolated_qk_construction_accepts_structural_route_admission_fixture():
  """TG3: the per-instance gate is now QKPrimitiveRouteAdmission (capability + target promotion), not the
  pre-TG3 QKPrimitiveEligibility single boolean -- see qk_primitives.py module docstring on the latter."""
  admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability("AMD", "gfx1100", 32, True), True)
  linear = Q4KPrimitiveLinear(None, None, Tensor.empty(8, dtype=dtypes.uint32), 1, 1, 1, (), "q4", 32, 0, "shared",
                              route_admission=admission)
  assert linear.route_admission is admission
  assert linear.route_admission.admitted


def test_q4k_shared_prefill_packed_weight_reuses_resident_view(monkeypatch):
  monkeypatch.delenv("PREFILL_PACKED_STREAM", raising=False)
  words = Tensor.empty(8, dtype=dtypes.uint32)
  linear = Q4KPrimitiveLinear(None, None, words, 1, 1, 1, (), "q4", 32, 0, "shared", shared_bytes=32)

  assert linear.prefill_packed_weight() is words
  assert linear.prefill_packed_weight() is linear.q4k_storage.words
  assert not hasattr(linear, "_prefill_q4k_words")


def test_q6k_shared_prefill_packed_weight_reuses_resident_view(monkeypatch):
  monkeypatch.delenv("PREFILL_PACKED_STREAM", raising=False)
  halfs = Tensor.empty(8, dtype=dtypes.uint16)
  linear = Q6KPrimitiveLinear(None, None, halfs, 1, 1, 1, (), "q6", 16, 0, "shared", shared_bytes=16)

  assert linear.prefill_packed_weight() is halfs
  assert linear.prefill_packed_weight() is linear.q6k_storage.halfs
  assert not hasattr(linear, "_prefill_q6k_halfs")


def test_q4k_ondemand_prefill_packed_weight_remains_distinct_and_cached(monkeypatch):
  monkeypatch.delenv("PREFILL_PACKED_STREAM", raising=False)
  words = Tensor.empty(8, dtype=dtypes.uint32)
  linear = Q4KPrimitiveLinear(None, None, words, 1, 1, 1, (), "q4", 32, 0, "q4_ondemand", nonpersistent_bytes=32)

  packed = linear.prefill_packed_weight()
  assert packed is not words
  assert linear.prefill_packed_weight() is packed


def test_q4k_install_uses_route_plan_without_direct_policy_call(tmp_path, monkeypatch):
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_gate.weight", (256, 256), 12, 0)]}
  plan = build_model_route_plan(meta)
  monkeypatch.delattr(qk_primitives, "_qk_generated_policy_entry")

  # TG3: installation is now gated on capability+promotion too (see qk_primitives.py), so a real target with
  # capability is supplied here -- this test's actual subject (route-plan-driven shape resolution without a
  # direct generated-policy call) is otherwise unaffected.
  installed = _install_q4k_primitives(_install_model(), gguf, meta, route_plan=plan, device_facts=_device_facts())

  assert len(installed) == 1
  assert isinstance(installed[0], Q4KPrimitiveLinear)
  assert installed[0].parts == 1
  assert installed[0].route_admission.admitted


def test_q4k_install_without_device_facts_is_not_admitted_but_still_constructed_from_route_plan(tmp_path, monkeypatch):
  """TG3: an unknown/absent target fails closed on capability (never treated as satisfied by omission), but
  route-plan shape resolution and construction proceed independently of whether the target is admitted --
  admission only decides `route_admission.admitted`, consulted at call time (see Q4KPrimitiveLinear.__call__)."""
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_gate.weight", (256, 256), 12, 0)]}
  plan = build_model_route_plan(meta)
  monkeypatch.delattr(qk_primitives, "_qk_generated_policy_entry")

  installed = _install_q4k_primitives(_install_model(), gguf, meta, route_plan=plan)

  assert installed == []


def test_explicit_qk_config_and_install_are_immune_to_environment(tmp_path, monkeypatch, capsys):
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_gate.weight", (256, 256), 12, 0)]}
  plan = build_model_route_plan(meta)
  cfg = QKConfig(False, None, "sidecar", "sidecar", False, False)
  monkeypatch.setenv("QK_PRIMITIVE_STORAGE", "q4_ondemand")
  monkeypatch.setenv("QK_PRIMITIVE_MAX_STORAGE_MB", "0")
  monkeypatch.setenv("Q4K_PRIMITIVE_DEBUG", "1")

  installed = _install_q4k_primitives(_install_model(), gguf, meta, route_plan=plan,
                                      budget=QKPrimitiveBudget(cfg.max_storage_bytes, cfg.generated_policy_strict),
                                      storage_mode=cfg.storage_mode, device_facts=_device_facts())

  assert len(installed) == 1
  assert installed[0].q4k_storage.mode == "sidecar"
  assert installed[0].q4k_storage.persistent_bytes > 0
  assert capsys.readouterr().out == ""


def test_qk_config_rejects_inconsistent_derived_storage_mode():
  with pytest.raises(ValueError, match="q6_storage_mode"):
    QKConfig(False, None, "shared", "sidecar", False, False)


def test_research_qk_mutation_apis_are_not_in_budgeted_core():
  assert not hasattr(qk_primitives, "Q4KFusedLinear")
  assert not hasattr(qk_primitives, "_install_q4k_fusions")
  assert not hasattr(qk_primitives, "_demote_q6k_to_q4")
  assert not {"demote_q6k_ffndown", "demote_targets", "fuse_q4k"} & QKConfig.__dataclass_fields__.keys()


def test_q4k_install_snapshots_load_entry_device_facts(tmp_path):
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_gate.weight", (256, 256), 12, 0)]}

  installed = _install_q4k_primitives(_install_model(), gguf, meta, route_plan=build_model_route_plan(meta),
                                      device_facts=_device_facts())

  # TG3: capability is read straight off the same DeviceFacts snapshot the pre-TG3 eligibility gate used, and
  # (with no promotion record loaded) target_promoted defaults to True -- so this AMD gfx1100 wave32 fixture
  # is admitted exactly as before, but via two separately-inspectable answers instead of one merged boolean.
  assert installed[0].route_admission.capability == QKPrimitiveCapability("AMD", "gfx1100", 32, True)
  assert installed[0].route_admission.target_promoted
  assert installed[0].route_admission.admitted


def test_q6k_install_uses_route_plan_without_direct_policy_call(tmp_path, monkeypatch):
  gguf = tmp_path / "q6.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 210))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_down.weight", (256, 256), 14, 0)]}
  plan = build_model_route_plan(meta)
  monkeypatch.delattr(qk_primitives, "_qk_generated_policy_entry")

  installed = _install_q6k_primitives(_install_model(), gguf, meta, route_plan=plan, device_facts=_device_facts())

  assert len(installed) == 1
  assert isinstance(installed[0], Q6KPrimitiveLinear)
  assert installed[0].parts == 1
  assert installed[0].route_admission.admitted


def test_legacy_q4_q6_install_policy_dispatchers_are_deleted():
  assert not hasattr(qk_primitives, "q4k_policy")
  assert not hasattr(qk_primitives, "q6k_policy")
  assert not hasattr(qk_primitives, "_q4k_policy")
  assert not hasattr(qk_primitives, "_q6k_policy")


def test_runtime_dispatch_install_selection_does_not_branch_on_model_size_or_name_literals():
  repo = pathlib.Path(__file__).resolve().parents[2]
  runtime_files = [
    repo / "tinygrad/llm/model.py",
    repo / "tinygrad/llm/model_route_plan.py",
    repo / "tinygrad/llm/qk_primitives.py",
  ]
  banned = ("8B", "14B", "32B", "8b", "14b", "32b")
  banned_ints = {8000, 8192, 14000, 14336, 32000, 32768}
  for path in runtime_files:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
      if not isinstance(node, (ast.If, ast.IfExp, ast.Match)): continue
      src = ast.get_source_segment(path.read_text(), node) or ""
      assert not any(token in src for token in banned), f"{path} branches on model name/size literal: {src}"
      for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, int):
          assert child.value not in banned_ints, f"{path} branches on model-size literal {child.value}: {src}"


def test_generated_policy_override_still_wins_over_route_plan(tmp_path):
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  name = "blk.0.ffn_gate.weight"
  meta = {"data_start": 0, "tensor_infos": [(name, (256, 256), 12, 0)]}
  generated_policy = {"by_shape": {}, "by_tensor": {
    (name, 12, 256, 256): {"winner": "generated", "parts": 1, "opts": ("LOCAL:0:32",),
                           "family": "q4_k_packed_u32_direct", "reduction": "direct_out"}
  }}

  class RoutePlanMustNotBeConsulted:
    def primitive(self, _name):
      raise AssertionError("generated policy path consulted route plan")

  installed = _install_q4k_primitives(_install_model(), gguf, meta, generated_policy=generated_policy,
                                      route_plan=RoutePlanMustNotBeConsulted())

  assert len(installed) == 1
  assert installed[0].kernel_mode == "direct_out"
  assert installed[0].parts == 1
