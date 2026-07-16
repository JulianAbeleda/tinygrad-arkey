import json, subprocess
from types import SimpleNamespace

from extra.qk import q4k_q8_mmq_uop_role_compile_gate as gate


def _metadata(_path):
  kv = {"general.architecture":"qwen3", "qwen3.embedding_length":5120, "qwen3.feed_forward_length":17408,
        "qwen3.attention.head_count":40, "qwen3.attention.head_count_kv":8, "qwen3.attention.key_length":128}
  tensors = [("blk.0.attn_k.weight", (5120,1024), 12, 0), ("blk.0.attn_v.weight", (5120,1024), 12, 0),
             ("blk.1.attn_k.weight", (5120,1024), 14, 0),
             ("blk.0.attn_q.weight", (5120,5120), 12, 0), ("blk.0.attn_output.weight", (5120,5120), 12, 0),
             ("blk.0.ffn_down.weight", (17408,5120), 12, 0), ("blk.0.ffn_gate.weight", (5120,17408), 12, 0),
             ("blk.0.ffn_up.weight", (5120,17408), 12, 0)]
  return kv, {"tensor_infos": tensors}


def _program(shape):
  return {"programs":[{"name":shape.kernel_name, **shape.grid,
                        "signed_wmma":{"source_signed_integer_wmma":True, "linear_signed_wmma_count":1,
                                       "linear_count":1, "authored_wmma_count":0}}], "fallback_used":False}


def test_metadata_derives_exact_loaded_role_shapes_and_explicit_runtime_m():
  shapes = gate.derive_role_shapes("unused.gguf", loader=_metadata)
  assert [(x.role,x.m,x.n,x.k,x.quant) for x in shapes] == [
    ("attn_kv",512,1024,5120,"Q4_K"), ("attn_qo",512,5120,5120,"Q4_K"),
    ("ffn_down",512,5120,17408,"Q4_K"), ("ffn_gate_up",512,17408,5120,"Q4_K")]


def test_final_evidence_is_fail_closed_for_program_name_grid_wmma_and_fallback():
  shape = gate.RoleShape("attn_kv",512,1024,5120,"Q4_K")
  assert gate.validate_compile_evidence(shape, _program(shape))["passed"]
  mutations = [
    {"programs":[], "fallback_used":False},
    {"programs":[{**_program(shape)["programs"][0], "name":"fallback"}], "fallback_used":False},
    {"programs":[{**_program(shape)["programs"][0], "global_size":[1,1,1]}], "fallback_used":False},
    {"programs":[{**_program(shape)["programs"][0], "signed_wmma":{"source_signed_integer_wmma":False,
      "linear_signed_wmma_count":1}}], "fallback_used":False},
    {"programs":[{**_program(shape)["programs"][0], "signed_wmma":{"source_signed_integer_wmma":True,
      "linear_signed_wmma_count":0}}], "fallback_used":False},
    {**_program(shape), "fallback_used":True},
  ]
  assert all(not gate.validate_compile_evidence(shape, row)["passed"] for row in mutations)


def test_gate_compiles_serially_and_stops_at_exact_first_failure(monkeypatch):
  monkeypatch.setattr(gate, "derive_role_shapes", lambda *_a, **_k: tuple(
    gate.RoleShape(r,512,*gate.EXPECTED_NK[r],"Q4_K") for r in gate.ROLE_ORDER))
  calls = []
  def runner(cmd, **kwargs):
    role = cmd[cmd.index("--role")+1]; calls.append((role, kwargs["timeout"]))
    shape = gate.RoleShape(role,512,*gate.EXPECTED_NK[role],"Q4_K")
    row = _program(shape)
    if role == "attn_qo": row["programs"].append({"name":"fallback"})
    return SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr="")
  out = gate.run_gate("unused", timeout_seconds=7, runner=runner, env={})
  assert not out["passed"] and out["first_failure"] == "attn_qo: expected exactly one PROGRAM, got 2"
  assert calls == [("attn_kv",7), ("attn_qo",7)]


def test_timeout_is_typed_and_never_advances(monkeypatch):
  shape = gate.RoleShape("attn_kv",512,1024,5120,"Q4_K")
  monkeypatch.setattr(gate, "derive_role_shapes", lambda *_a, **_k: (shape,))
  def runner(*_a, **_k): raise subprocess.TimeoutExpired(["python"], 3)
  out = gate.run_gate("unused", timeout_seconds=3, runner=runner, env={})
  assert out["first_failure"] == "attn_kv: compile timed out after 3s"
