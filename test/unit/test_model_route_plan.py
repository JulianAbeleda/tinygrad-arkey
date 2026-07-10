from types import SimpleNamespace

from tinygrad import Tensor, dtypes
from tinygrad.llm import qk_primitives, route_policy
from tinygrad.llm.model_route_plan import build_model_route_plan
from tinygrad.llm.qk_primitives import _install_q4k_primitives, _install_q6k_primitives, Q4KPrimitiveLinear, Q6KPrimitiveLinear


QWEN3_LIKE_SHAPES = (
  {"hidden": 4096, "intermediate": 12288, "kv": 1024},
  {"hidden": 5120, "intermediate": 17408, "kv": 1024},
)


def _tensor_rows(profile):
  h, i, kv = profile["hidden"], profile["intermediate"], profile["kv"]
  return (
    ("blk.0.ffn_gate.weight", (h, i), 12),
    ("blk.0.ffn_up.weight", (h, i), 12),
    ("blk.0.ffn_down.weight", (i, h), 12),
    ("blk.0.attn_q.weight", (h, h), 12),
    ("blk.0.attn_output.weight", (h, h), 12),
    ("blk.0.attn_k.weight", (h, kv), 12),
    ("blk.0.attn_v.weight", (h, kv), 12),
    ("blk.1.ffn_down.weight", (i, h), 14),
    ("blk.1.attn_v.weight", (h, kv), 14),
    ("output.weight", (h, 151936), 14),
  )


def _meta_for_profile(profile):
  offset = 0
  rows = []
  for name, dims, typ in _tensor_rows(profile):
    rows.append((name, dims, typ, offset))
    offset += 4
  return {"data_start": 0, "tensor_infos": rows}


def test_default_model_route_plan_matches_legacy_policy_for_qwen3_like_tensors():
  for profile in QWEN3_LIKE_SHAPES:
    meta = _meta_for_profile(profile)
    plan = build_model_route_plan(meta)
    for name, dims, typ, _off in meta["tensor_infos"]:
      entry = plan.primitive(name)
      legacy = route_policy.q4k_policy(name) if typ == 12 else route_policy.q6k_policy(name)
      assert legacy is not None
      assert entry is not None
      assert (entry.rows, entry.cols) == tuple(reversed(dims))
      assert (entry.parts, entry.opts) == (legacy[0], tuple(legacy[1]))
      assert entry.module_path == name.removesuffix(".weight")


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

  installed = _install_q4k_primitives(_install_model(), gguf, meta, generated_policy=generated_policy,
                                      route_plan=build_model_route_plan(meta))

  assert len(installed) == 1
  assert installed[0].kernel_mode == "direct_out"
  assert installed[0].parts == 1
