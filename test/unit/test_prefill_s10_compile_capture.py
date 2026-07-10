import os

from extra.qk.prefill import s10_compile_capture as cap


def test_capture_amd_compile_sources_records_failing_source(monkeypatch, tmp_path):
  from tinygrad.runtime.support import compiler_amd

  def fake_compile(self, src):
    raise RuntimeError("boom")

  monkeypatch.setattr(compiler_amd.HIPCompiler, "compile", fake_compile)

  with cap.capture_amd_compile_sources(tmp_path) as failures:
    try:
      compiler_amd.HIPCompiler.__new__(compiler_amd.HIPCompiler).compile("extern \"C\" __global__ void k() {}")
    except RuntimeError:
      pass

  assert len(failures) == 1
  assert failures[0]["compiler"] == "HIPCompiler"
  assert failures[0]["source_path"].endswith(".cpp")
  assert "extern" in (tmp_path / failures[0]["source_path"].split("/")[-1]).read_text()


def test_analyze_amd_source_classifies_attn_kv_lds_overflow():
  src = """
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) r_test(half* data0_524288, half* data1_2097152, half* data2_4194304) {
  __attribute__((shared, aligned(16)))half buf0[2048];
  float buf1[32];
  __attribute__((shared, aligned(16)))half buf2[32768];
  float8 x = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
"""
  out = cap.analyze_amd_source(src)

  assert out["kernel_name"] == "r_test"
  assert out["inferred_prefill_shape"] == {"m": 512, "n": 1024, "k": 4096}
  assert out["inferred_prefill_role"] == "attn_kv"
  assert out["inferred_route_family"] == "pipe"
  assert out["shared_bytes"] == 69632
  assert out["shared_over_limit"] is True
  assert out["contains_wmma_builtin"] is True


def test_run_capture_restores_s10_env_and_records_success(monkeypatch, tmp_path):
  old = {k: os.environ.get(k) for k in cap.S10_ROUTE_ENV}

  def fake_profile(mode, max_context):
    class Profile:
      K = 1
      warmups = 0
      rounds = 1
      start_positions = (0,)
      whole_lengths = (512,)
      chunk_n = 512
      max_context = 1024
      mode = "smoke"
    return Profile()

  def fake_authority(**kwargs):
    assert kwargs["require_route"] == "prefill_wmma_pipe_lds_dbuf_primitive_generated"
    assert os.environ["PREFILL_WMMA_LDS_PRIMITIVE"] == "1"
    return {"whole_tok_s": {"512": 1.0}}

  monkeypatch.setattr(cap, "prefill_run_profile", fake_profile)
  import extra.qk.prefill_whole_synced as whole
  monkeypatch.setattr(whole, "prefill_authority", fake_authority)

  out = cap.run_capture(out_dir=tmp_path)

  assert out["status"] == "ok"
  assert out["captured_failures"] == []
  assert out["whole_prefill_report"] == {"whole_tok_s": {"512": 1.0}}
  for key, value in old.items():
    assert os.environ.get(key) == value


def test_run_capture_records_error(monkeypatch, tmp_path):
  def fake_profile(mode, max_context):
    class Profile:
      K = 1
      warmups = 0
      rounds = 1
      start_positions = (0,)
      whole_lengths = (512,)
      chunk_n = 512
      max_context = 1024
      mode = "smoke"
    return Profile()

  def fake_authority(**kwargs):
    raise RuntimeError("compile failed")

  monkeypatch.setattr(cap, "prefill_run_profile", fake_profile)
  import extra.qk.prefill_whole_synced as whole
  monkeypatch.setattr(whole, "prefill_authority", fake_authority)

  out = cap.run_capture(out_dir=tmp_path)

  assert out["status"] == "compile_or_runtime_error"
  assert out["error"]["type"] == "RuntimeError"
  assert "compile failed" in out["error"]["message"]
  assert out["pre_route_blocker_note"] is None


def test_run_capture_explains_amd_isa_q4k_prefill_weight_blocker(monkeypatch, tmp_path):
  def fake_profile(mode, max_context):
    class Profile:
      K = 1
      warmups = 0
      rounds = 1
      start_positions = (0,)
      whole_lengths = (512,)
      chunk_n = 512
      max_context = 1024
      mode = "smoke"
    return Profile()

  def fake_authority(**kwargs):
    raise NotImplementedError("AMD:ISA CAST dtypes.char -> dtypes.float unsupported")

  monkeypatch.setenv("DEV", "AMD:ISA")
  monkeypatch.setattr(cap, "prefill_run_profile", fake_profile)
  import extra.qk.prefill_whole_synced as whole
  monkeypatch.setattr(whole, "prefill_authority", fake_authority)

  out = cap.run_capture(out_dir=tmp_path, scenario="lds-only")

  assert out["status"] == "compile_or_runtime_error"
  assert out["device_env"] == "AMD:ISA"
  assert "before the S10 route is entered" in out["pre_route_blocker_note"]


def test_run_capture_supports_decoupled_lds_only_scenario(monkeypatch, tmp_path):
  def fake_profile(mode, max_context):
    class Profile:
      K = 1
      warmups = 0
      rounds = 1
      start_positions = (0,)
      whole_lengths = (512,)
      chunk_n = 512
      max_context = 1024
      mode = "smoke"
    return Profile()

  def fake_authority(**kwargs):
    assert kwargs["require_route"] == "prefill_wmma_lds_dbuf_primitive_mixed"
    assert os.environ["PREFILL_WMMA_LDS_PRIMITIVE"] == "1"
    assert os.environ["PREFILL_DBUF"] == "1"
    assert "PREFILL_WMMA_PIPE_PRIMITIVE" not in os.environ
    return {"whole_tok_s": {"512": 1.0}}

  monkeypatch.setattr(cap, "prefill_run_profile", fake_profile)
  import extra.qk.prefill_whole_synced as whole
  monkeypatch.setattr(whole, "prefill_authority", fake_authority)

  out = cap.run_capture(out_dir=tmp_path, scenario="lds-only")

  assert out["status"] == "ok"
  assert out["scenario"] == "lds-only"
  assert out["required_route"] == "prefill_wmma_lds_dbuf_primitive_mixed"


def test_summarize_gate_ab_reports_pass(tmp_path):
  gate_on = tmp_path / "on.json"
  gate_off = tmp_path / "off.json"
  gate_on.write_text("""{
    "status": "ok",
    "captured_failures": [],
    "whole_prefill_report": {
      "prefill_role_routes": {"attn_kv": "generated_pipe_no_local_stage"},
      "whole_tok_s": {"512": 218.0},
      "prefill_route_binding_gate": {"verdict": "PREFILL_ROUTE_BINDING_PASS"}
    }
  }""")
  gate_off.write_text("""{
    "status": "compile_or_runtime_error",
    "captured_failures": [{
      "source_analysis": {
        "inferred_prefill_role": "attn_kv",
        "shared_over_limit": true,
        "shared_bytes": 69632,
        "shared_limit_bytes": 65536
      }
    }]
  }""")

  out = cap.summarize_gate_ab(gate_on, gate_off)

  assert out["verdict"] == "S10_ATTN_KV_NO_LOCAL_STAGE_PASS"
  assert out["gate_on"]["captured_failures"] == 0
  assert out["gate_off"]["source_analysis"]["shared_bytes"] == 69632
