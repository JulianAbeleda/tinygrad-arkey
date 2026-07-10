from types import SimpleNamespace

from tinygrad import Tensor, dtypes
from tinygrad.llm import qk_primitives, route_policy
from tinygrad.llm.model_facts import model_facts_from_gguf_metadata
from tinygrad.llm.model_route_plan import build_model_route_plan
from tinygrad.llm.qk_primitives import _install_q4k_primitives, _install_q6k_primitives, Q4KPrimitiveLinear, Q6KPrimitiveLinear


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


def test_model_facts_route_plan_matches_legacy_q4_q6_policy_for_qwen3_like_tensors():
  for profile in QWEN3_LIKE_PROFILES:
    meta = _meta_for_profile(profile)
    facts = model_facts_from_gguf_metadata(_qwen_kv(profile), meta)
    plan = build_model_route_plan(meta, facts)
    expected_roles = {name: role for name, _dims, _typ, role in _tensor_rows(profile)}

    assert len(facts.tensors) == len(meta["tensor_infos"]), profile["name"]
    for name, dims, typ, _off in meta["tensor_infos"]:
      entry = plan.primitive(name)
      legacy = route_policy.q4k_policy(name) if typ == 12 else route_policy.q6k_policy(name)
      assert legacy is not None
      assert entry is not None
      assert (entry.rows, entry.cols) == tuple(reversed(dims))
      assert (entry.parts, entry.opts) == (legacy[0], tuple(legacy[1]))
      assert entry.module_path == name.removesuffix(".weight")
      assert entry.role == expected_roles[name]
      assert entry.quant_label == ("Q4_K" if typ == 12 else "Q6_K")


def _linear(rows, cols):
  return SimpleNamespace(weight=Tensor.empty(rows, cols, dtype=dtypes.float16), bias=None)


def _install_model():
  return SimpleNamespace(blk=[SimpleNamespace(
    ffn_gate=_linear(256, 256),
    ffn_down=_linear(256, 256),
  )])


def test_q4k_install_uses_route_plan_without_direct_policy_call(tmp_path, monkeypatch):
  gguf = tmp_path / "q4.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 144))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_gate.weight", (256, 256), 12, 0)]}
  plan = build_model_route_plan(meta)
  monkeypatch.setattr(qk_primitives, "_q4k_policy", lambda _name: (_ for _ in ()).throw(AssertionError("direct q4 policy called")))

  installed = _install_q4k_primitives(_install_model(), gguf, meta, route_plan=plan)

  assert len(installed) == 1
  assert isinstance(installed[0], Q4KPrimitiveLinear)
  assert installed[0].parts == route_policy.q4k_policy("blk.0.ffn_gate.weight")[0]


def test_q6k_install_uses_route_plan_without_direct_policy_call(tmp_path, monkeypatch):
  gguf = tmp_path / "q6.bin"
  gguf.write_bytes(bytes((256 * 256) // 256 * 210))
  meta = {"data_start": 0, "tensor_infos": [("blk.0.ffn_down.weight", (256, 256), 14, 0)]}
  plan = build_model_route_plan(meta)
  monkeypatch.setattr(route_policy, "q6k_policy", lambda _name: (_ for _ in ()).throw(AssertionError("direct q6 policy called")))

  installed = _install_q6k_primitives(_install_model(), gguf, meta, route_plan=plan)

  assert len(installed) == 1
  assert isinstance(installed[0], Q6KPrimitiveLinear)
  assert installed[0].parts == 1


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
