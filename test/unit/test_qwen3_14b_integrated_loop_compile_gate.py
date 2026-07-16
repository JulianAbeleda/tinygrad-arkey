from dataclasses import replace

import pytest

from extra.qk import model_profiles
from extra.qk.qwen3_14b_integrated_loop_compile_gate import CASES, run, workload_authority


def test_integrated_loop_gate_orders_smoke_first_and_stops_on_compile_failure(monkeypatch):
  seen = []

  def fake_run(cmd, **kwargs):
    class Result:
      stdout = '{"status":"COMPILE_FAILURE","error":"synthetic compile failure"}\n'
      stderr = ""
    seen.append(cmd)
    return Result()

  monkeypatch.setattr("extra.qk.qwen3_14b_integrated_loop_compile_gate.subprocess.run", fake_run)
  report = run(timeout=1)
  assert report["rows"][0]["role"] == "smoke_32x32x512"
  assert len(report["rows"]) == 1
  worker_calls = [cmd for cmd in seen if any("qwen3_14b_integrated_loop_compile_gate.py" in part for part in cmd)]
  assert len(worker_calls) == 1
  assert CASES == (("smoke_32x32x512", 32, 32, 512), ("ffn_down", 512, 5120, 17408))
  assert report["workload_authority"] == {
    "profile_id": "qwen3_14b_q4k_m_gfx1100", "family": "qwen3", "size_label": "14B",
    "device_profile": "gfx1100", "family_quant": "Q4_K_M", "candidate_quant": "Q4_K",
    "role": "ffn_down", "phase": "prefill", "shape": {"M": 512, "N": 5120, "K": 17408},
    "shape_source": "model_profile", "profile_shape_is_loaded_tensor_quant_authority": False,
    "loaded_tensor_quant_authority": {"status": "NOT_LOADED", "quant": None,
      "required_source": "loaded_GGUF_tensor_type", "runtime_binding": "fail_closed"}, "passed": True}
  assert report["rows"][0]["candidate_quant"] == "Q4_K"


@pytest.mark.parametrize("field,value", [("role", "ffn_gate_up"), ("phase", "decode"), ("quant", "Q8_0"), ("K", 0)])
def test_workload_authority_rejects_bad_role_quant_or_shape_evidence(monkeypatch, field, value):
  profile = model_profiles.qwen3_14b_q4k_m_gfx1100_profile()
  ffn_down = profile.role_shape("ffn_down")
  bad_role = replace(ffn_down, **{field: value})
  bad_profile = replace(profile, roles=tuple(bad_role if role is ffn_down else role for role in profile.roles))
  monkeypatch.setattr(model_profiles, "qwen3_14b_q4k_m_gfx1100_profile", lambda: bad_profile)
  with pytest.raises(ValueError): workload_authority()
