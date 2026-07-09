import json
import sys
import types

import pytest

from extra.qk.prefill_pipe_mvp_artifact import SCHEMA, build_report, validate_report


def test_prefill_pipe_mvp_artifact_schema_snapshot_is_reproducible():
  report = build_report(artifact=False)
  assert report["schema"] == SCHEMA
  assert report["role"] == "attn_qo"
  assert report["shape"] == {"m": 512, "n": 4096, "k": 4096}
  assert report["prefill_gemm_schedule_spec"]["route_family"] == "pipe"
  assert report["wmma_pipe_spec"]["wait_policy"] == "targeted_vmcnt"
  assert report["route_attribution"]["generated_pipe_selected"] is False
  assert report["route_attribution"]["uses_hand_pipe_oracle"] is True
  assert report["correctness"]["status"] == "not_run"
  assert report["trace_counters"]["generated_route_attribution"] is False
  assert report["timing"]["status"] == "not_run"
  assert report["per_role_timing"]["schema"] == "prefill-per-role-timing-attribution.v1"
  assert report["per_role_timing"]["status"] == "not_run"
  assert list(report["per_role_timing"]["roles"]) == ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"]
  assert report["per_role_timing"]["roles"]["attn_qo"]["timing"] == {
    "status": "not_run", "samples": [], "median_ms": None, "tflops": None,
  }
  assert report["per_role_timing"]["roles"]["ffn_gate_up"]["route_attribution"]["route_family"] == "lds"
  assert validate_report(report) == []
  json.dumps(report)


def test_prefill_pipe_mvp_artifact_rejects_false_generated_claim():
  report = build_report(artifact=False)
  report["route_attribution"]["generated_pipe_selected"] = True
  with pytest.raises(ValueError, match="hand pipe oracle"):
    from extra.qk.prefill_pipe_mvp_artifact import write_report
    write_report(report)


def test_prefill_pipe_mvp_artifact_per_role_timing_requires_explicit_entries():
  report = build_report(artifact=False)
  del report["per_role_timing"]["roles"]["ffn_gate_up"]["timing"]

  errors = validate_report(report)

  assert "$.per_role_timing.roles.ffn_gate_up.timing missing" in errors


def test_prefill_pipe_mvp_artifact_per_role_timing_can_use_existing_role_reports(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_build_report(*, role, m, n, k, **kwargs):
    return {
      "env": {"flags": {"PREFILL_WMMA_PIPE_PRIMITIVE": "1"}},
      "role": role,
      "shape": {"m": m, "n": n, "k": k},
      "route_attribution": {
        "selected_route": "prefill_wmma_pipe_primitive_generated",
        "route_family": "pipe",
        "generated_pipe_selected": True,
        "uses_hand_pipe_oracle": False,
      },
      "timing": {"status": "compile_included_sample", "samples": [3.0], "median_ms": 3.0, "tflops": 4.0},
      "correctness": {"status": "pass"},
    }

  def fake_lds_report(**kwargs):
    return {
      "env": {"flags": {"PREFILL_WMMA_LDS_PRIMITIVE": "1", "PREFILL_DBUF": "1"}},
      "role": "ffn_gate_up",
      "shape": {"m": 512, "n": 12288, "k": 4096},
      "route_attribution": {
        "selected_route": "prefill_pipe_role_selective_generated",
        "route_family": "lds",
        "generated_lds_selected": True,
        "uses_hand_lds_oracle": False,
        "uses_hand_pipe_oracle": False,
      },
      "timing": {"status": "compile_included_sample", "samples": [7.0], "median_ms": 7.0, "tflops": 8.0},
      "correctness": {"status": "pass"},
    }

  monkeypatch.setattr(art, "build_report", fake_build_report)
  monkeypatch.setattr(art, "build_lds_primitive_report", fake_lds_report)

  timing = art.build_per_role_timing_report(measure=True, sample_cols=8)

  assert timing["status"] == "measured"
  assert timing["roles"]["attn_kv"]["route_attribution"]["selected_route"] == "prefill_wmma_pipe_primitive_generated"
  assert timing["roles"]["attn_kv"]["route_flags"]["PREFILL_WMMA_PIPE_PRIMITIVE"] == "1"
  assert timing["roles"]["ffn_gate_up"]["timing"]["median_ms"] == 7.0
  assert timing["roles"]["ffn_gate_up"]["hand_reference"]["status"] == "not_available"


def test_prefill_pipe_mvp_artifact_can_embed_bounded_diagnostic_lowering():
  report = build_report(artifact=False, diagnostic_lowering=True, diagnostic_shape=(64, 64, 64))
  diag = report["diagnostic_lowering"]

  assert diag["schema"] == "wmma-pipe-diagnostic-lowering.v1"
  assert diag["transport"] == "generated_program_diagnostic"
  assert diag["route_bound"] is False
  assert diag["uses_hand_pipe_oracle"] is False
  assert diag["track_counts"]["global_load_b128"] > 0
  assert diag["track_counts"]["v_wmma_f32_16x16x16_f16"] > 0
  assert diag["waitcnt_summary"]["count"] > 0
  assert diag["mvp_core_structure_ok"] is True
  assert diag["mvp_pipe_wait_ok"] is True
  assert diag["mvp_structure_ok"] is True
  assert diag["waitcnt_summary"]["has_expected_pipe_vmcnt"] is True
  assert validate_report(report) == []


def test_prefill_pipe_mvp_artifact_diagnostic_correctness_flag_calls_helper(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_correctness(spec):
    return {"schema": "wmma-pipe-diagnostic-correctness.v1", "passed": True, "spec": spec.to_json()}

  monkeypatch.setattr(art, "run_wmma_pipe_diagnostic_correctness", fake_correctness)
  report = art.build_report(artifact=False, diagnostic_lowering=True, diagnostic_correctness=True,
                            diagnostic_shape=(64, 64, 64))
  assert report["diagnostic_correctness"]["schema"] == "wmma-pipe-diagnostic-correctness.v1"
  assert report["diagnostic_correctness"]["passed"] is True
  assert validate_report(report) == []


def test_prefill_pipe_mvp_artifact_route_sample_correctness_updates_promotion_fields(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_route_sample(**kwargs):
    return {
      "schema": "prefill-pipe-route-sample-correctness.v1",
      "passed": True,
      "finite": True,
      "threshold": 2e-2,
      "max_abs_error": 1e-4,
      "rel_rmse": 2e-4,
      "elapsed_ms_compile_included": 12.0,
      "tflops_compile_included": 1.0,
      "uses_hand_pipe_oracle": False,
    }

  monkeypatch.setattr(art, "run_route_sample_correctness", fake_route_sample)
  report = art.build_report(artifact=False, route_sample_correctness=True, sample_cols=8)

  assert report["route_sample_correctness"]["schema"] == "prefill-pipe-route-sample-correctness.v1"
  assert report["route_sample_correctness"]["passed"] is True
  assert report["route_attribution"]["generated_pipe_selected"] is True
  assert report["route_attribution"]["uses_hand_pipe_oracle"] is False
  assert report["correctness"]["status"] == "pass"
  assert report["correctness"]["max_rel_error"] == 2e-4
  assert report["timing"]["status"] == "compile_included_sample"
  assert validate_report(report) == []


def test_prefill_pipe_mvp_artifact_lifecycle_trace_updates_structural_counters(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_trace(**kwargs):
    return {
      "schema": "prefill-pipe-lifecycle-trace-summary.v1",
      "track_counts": {"global_load_b128": 16, "v_wmma_f32_16x16x16_f16": 8},
      "waitcnt_summary": {"count": 7, "nonfull_count": 7},
      "generated_route_attribution": True,
      "ok": True,
    }

  monkeypatch.setattr(art, "run_lifecycle_trace_summary", fake_trace)
  report = art.build_report(artifact=False, lifecycle_trace=True)

  assert report["lifecycle_trace"]["schema"] == "prefill-pipe-lifecycle-trace-summary.v1"
  assert report["trace_counters"] == {
    "b128_global_loads": 16,
    "wmma": 8,
    "targeted_waitcnt": 7,
    "full_waitcnt": 0,
    "generated_route_attribution": True,
  }
  assert validate_report(report) == []


def test_prefill_pipe_mvp_all_pipe_roles_report_passes_when_each_role_passes(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_build_report(*, role, m, n, k, **kwargs):
    return {
      "role": role,
      "shape": {"m": m, "n": n, "k": k},
      "prefill_gemm_schedule_spec": {"route_family": "pipe"},
      "route_attribution": {
        "selected_route": "prefill_wmma_pipe_primitive_generated",
        "uses_hand_pipe_oracle": False,
      },
      "correctness": {"status": "pass"},
      "trace_counters": {
        "generated_route_attribution": True,
        "wmma": 8,
        "b128_global_loads": 32,
        "full_waitcnt": 0,
      },
    }

  monkeypatch.setattr(art, "build_report", fake_build_report)
  report = art.build_all_pipe_roles_report(artifact=False)

  assert report["verdict"] == "PATH1_PIPE_ALL_ROLES_PASS"
  assert report["roles"] == ["attn_qo", "attn_kv", "ffn_down"]
  assert report["excluded_roles"] == ["ffn_gate_up"]
  assert report["failures"] == []


def test_prefill_pipe_mvp_all_pipe_roles_report_fails_on_oracle_use(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_build_report(*, role, m, n, k, **kwargs):
    selected = "prefill_pipe_role_selective_generated" if role == "ffn_down" else "prefill_wmma_pipe_primitive_generated"
    return {
      "role": role,
      "shape": {"m": m, "n": n, "k": k},
      "prefill_gemm_schedule_spec": {"route_family": "pipe"},
      "route_attribution": {
        "selected_route": selected,
        "uses_hand_pipe_oracle": role == "ffn_down",
      },
      "correctness": {"status": "pass"},
      "trace_counters": {
        "generated_route_attribution": True,
        "wmma": 8,
        "b128_global_loads": 32,
        "full_waitcnt": 0,
      },
    }

  monkeypatch.setattr(art, "build_report", fake_build_report)
  report = art.build_all_pipe_roles_report(artifact=False)

  assert report["verdict"] == "PATH1_PIPE_ALL_ROLES_FAIL"
  assert any("ffn_down" in failure for failure in report["failures"])


def test_prefill_lds_primitive_report_records_generated_transport_compile(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_trace(**kwargs):
    return {
      "schema": "prefill-lds-oracle-trace-summary.v1",
      "track_counts": {
        "global_load_b128": 16,
        "ds_store_b128": 16,
        "ds_load_b128": 96,
        "v_wmma_f32_16x16x16_f16": 64,
        "s_barrier": 4,
      },
      "waitcnt_summary": {"nonfull_count": 18},
      "packed_global_to_lds_to_wmma_visible": True,
      "scalar_lds_fallback_total": 0,
      "ok": True,
    }

  def fake_generated_compile(**kwargs):
    return {
      "schema": "prefill-lds-generated-transport-compile.v1",
      "status": "ok",
      "transport": "ordinary_generated_matmul",
      "uses_hand_lds_oracle": False,
      "structural_ok": True,
      "track_counts": {
        "global_load_b128": 24,
        "global_load_u16": 0,
        "ds_store_b128": 24,
        "ds_store_b32": 0,
        "ds_store_b16": 0,
        "ds_load_b128": 64,
        "v_wmma_f32_16x16x16_f16": 16,
      },
    }

  monkeypatch.setattr(art, "run_lds_oracle_trace_summary", fake_trace)
  monkeypatch.setattr(art, "run_generated_lds_transport_compile_summary", fake_generated_compile)
  monkeypatch.setattr(art, "run_generated_lds_dbuf_cadence_probe", lambda **kwargs: {
    "schema": "prefill-lds-dbuf-cadence-probe.v1",
    "status": "ok",
    "candidate_ok": True,
    "promoted": False,
    "next_blocker": "strict dynamic D2 two-operand slot identity",
  })
  report = art.build_lds_primitive_report(artifact=False)

  assert report["schema"] == "prefill-lds-primitive-result.v1"
  assert report["verdict"] == "PREFILL_LDS_PRIMITIVE_GENERATED_TRANSPORT_COMPILES_BLOCKED_ON_CORRECTNESS_PERF"
  assert report["role"] == "ffn_gate_up"
  assert report["shape"] == {"m": 512, "n": 12288, "k": 4096}
  assert report["prefill_gemm_schedule_spec"]["route_family"] == "lds"
  assert report["lds_primitive_spec"]["lds_total_bytes"] == 40960
  assert report["route_attribution"]["generated_lds_selected"] is True
  assert report["route_attribution"]["uses_hand_lds_oracle"] is False
  assert report["route_attribution"]["uses_hand_pipe_oracle"] is False
  assert report["trace_counters"]["ds_store_b128"] == 16
  assert report["trace_counters"]["ds_load_b128"] == 96
  assert report["trace_counters"]["wmma"] == 64
  assert report["trace_counters"]["barriers"] == 4
  assert report["trace_counters"]["scalar_lds_fallback_total"] == 0
  assert report["generated_transport_compile"]["structural_ok"] is True
  assert report["generated_transport_compile"]["track_counts"]["ds_store_b128"] == 24
  assert report["lds_slot_identity_proof"]["ok"] is True
  assert report["lds_slot_identity_proof"]["active_buffers"] == 1
  assert report["lds_slot_identity_proof"]["dbuf_cadence_proven"] is False
  assert report["dbuf_slot_identity_proof"]["ok"] is True
  assert report["dbuf_slot_identity_proof"]["active_buffers"] == 2
  assert report["dbuf_slot_identity_proof"]["dbuf_slot_identity_proven"] is True
  assert report["generated_dbuf_cadence_probe"]["status"] == "ok"
  assert report["generated_dbuf_cadence_probe"]["candidate_ok"] is True
  assert report["generated_dbuf_cadence_probe"]["promoted"] is False
  assert report["generated_lowerer"]["status"] == "route_transport_wired_lowerer_contract_still_fail_closed"


def test_prefill_lds_primitive_report_records_oracle_trace_failures(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  def fake_trace(**kwargs):
    return {
      "schema": "prefill-lds-oracle-trace-summary.v1",
      "track_counts": {},
      "waitcnt_summary": {},
      "packed_global_to_lds_to_wmma_visible": False,
      "scalar_lds_fallback_total": 2,
      "ok": True,
    }

  monkeypatch.setattr(art, "run_lds_oracle_trace_summary", fake_trace)
  monkeypatch.setattr(art, "run_generated_lds_transport_compile_summary", lambda **kwargs: {"status": "ok", "structural_ok": True})
  monkeypatch.setattr(art, "run_generated_lds_dbuf_cadence_probe", lambda **kwargs: {"status": "ok", "candidate_ok": False, "promoted": False})
  report = art.build_lds_primitive_report(artifact=False)

  failures = report["generated_lowerer"]["failures"]
  assert any("global->LDS->WMMA" in failure for failure in failures)
  assert any("scalar LDS fallback" in failure for failure in failures)


def test_prefill_lds_primitive_report_can_embed_route_sample_correctness(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art

  monkeypatch.setattr(art, "run_lds_oracle_trace_summary", lambda **kwargs: {
    "schema": "prefill-lds-oracle-trace-summary.v1",
    "track_counts": {
      "global_load_b128": 16, "ds_store_b128": 16, "ds_load_b128": 96,
      "v_wmma_f32_16x16x16_f16": 64, "s_barrier": 4,
    },
    "waitcnt_summary": {"nonfull_count": 18},
    "packed_global_to_lds_to_wmma_visible": True,
    "scalar_lds_fallback_total": 0,
    "ok": True,
  })
  monkeypatch.setattr(art, "run_generated_lds_transport_compile_summary", lambda **kwargs: {
    "status": "ok", "schema": "prefill-lds-generated-transport-compile.v1",
    "structural_ok": True, "uses_hand_lds_oracle": False,
  })
  monkeypatch.setattr(art, "run_generated_lds_dbuf_cadence_probe", lambda **kwargs: {
    "status": "ok", "schema": "prefill-lds-dbuf-cadence-probe.v1", "candidate_ok": True, "promoted": False,
  })
  monkeypatch.setattr(art, "run_lds_route_sample_correctness", lambda **kwargs: {
    "schema": "prefill-lds-route-sample-correctness.v1",
    "passed": True,
    "finite": True,
    "threshold": 2e-2,
    "max_abs_error": 1e-4,
    "rel_rmse": 2e-4,
    "elapsed_ms_compile_included": 20.0,
    "tflops_compile_included": 2.5,
    "uses_hand_lds_oracle": False,
    "uses_hand_pipe_oracle": False,
    "dbuf_enabled": bool(kwargs.get("dbuf", False)),
  })

  report = art.build_lds_primitive_report(artifact=False, lds_sample_correctness=True, sample_cols=8)

  assert report["lds_route_sample_correctness"]["schema"] == "prefill-lds-route-sample-correctness.v1"
  assert report["lds_route_sample_correctness"]["dbuf_enabled"] is False
  assert report["lds_dbuf_route_sample_correctness"]["dbuf_enabled"] is True
  assert report["correctness"]["status"] == "pass"
  assert report["correctness"]["max_rel_error"] == 2e-4
  assert report["timing"]["status"] == "compile_included_sample"
  assert report["timing"]["median_ms"] == 20.0
  assert report["route_attribution"]["uses_hand_lds_oracle"] is False
  assert report["route_attribution"]["uses_hand_pipe_oracle"] is False


def test_prefill_lds_dbuf_env_bundle_keeps_primitive_flags():
  from extra.qk import prefill_pipe_mvp_artifact as art
  from extra.qk.prefill_schedule_spec import describe_prefill_schedule
  from extra.qk.wmma_lds_spec import extract_wmma_lds_spec

  lds = extract_wmma_lds_spec(describe_prefill_schedule(12288, 4096, role="ffn_gate_up"))
  env = art._wmma_lds_dbuf_env_defaults(lds)

  assert env["PREFILL_DBUF"] == "1"
  assert env["PREFILL_DBUF_NBUF"] == "2"
  assert env["PREFILL_DBUF_D3A_POST"] == "1"
  assert env["PREFILL_DBUF_LDS_INDEX_SPLIT"] == "1"
  assert env["REGALLOC_ADDR_REMAT"] == "1"
  assert env["PREFILL_TC_LOCAL_STAGE"] == "both"
  assert env["PREFILL_LDS_PACK_WITHLOCAL_B128"] == "1"


def test_run_lds_route_sample_correctness_restores_env_and_warmstart(monkeypatch):
  from extra.qk import prefill_pipe_mvp_artifact as art
  import numpy as np

  class FakeTensor:
    def __init__(self, data, dtype=None):
      self.data = np.array(data)
    def realize(self):
      return self
    def float(self):
      return self
    def numpy(self):
      return self.data

  class FakeDevice:
    DEFAULT = "FAKE"
    def __getitem__(self, key):
      return self
    def synchronize(self):
      return None

  class FakeContext:
    def __init__(self, **kwargs):
      self.kwargs = kwargs
    def __enter__(self):
      return self
    def __exit__(self, exc_type, exc, tb):
      return False

  class FakeCache:
    def __init__(self):
      self.clears = 0
    def clear(self):
      self.clears += 1

  class FakeGetenv:
    def __init__(self):
      self.clears = 0
    def cache_clear(self):
      self.clears += 1

  fake_getenv = FakeGetenv()
  fake_program_cache = FakeCache()
  fake_postrange = types.ModuleType("tinygrad.codegen.opt.postrange")
  fake_postrange._WARMSTART_OPTS = {"old": "warmstart"}
  fake_postrange._warmstart_stats = {"matched": 1}
  fake_tinygrad = types.ModuleType("tinygrad")
  fake_tinygrad.__path__ = []
  fake_tinygrad.Device = FakeDevice()
  fake_tinygrad.Tensor = FakeTensor
  fake_tinygrad.dtypes = types.SimpleNamespace(half="half")
  fake_codegen = types.ModuleType("tinygrad.codegen")
  fake_codegen.__path__ = []
  fake_codegen.to_program_cache = fake_program_cache
  fake_helpers = types.ModuleType("tinygrad.helpers")
  fake_helpers.Context = FakeContext
  fake_helpers.getenv = fake_getenv
  fake_codegen_opt = types.ModuleType("tinygrad.codegen.opt")
  fake_codegen_opt.__path__ = []
  fake_codegen_opt.postrange = fake_postrange
  fake_tinygrad.codegen = fake_codegen
  fake_codegen.opt = fake_codegen_opt
  fake_route_mod = types.ModuleType("extra.qk.prefill_graph_gemm_route")

  def fake_route(lin, x):
    assert lin._prefill_graph_role == "ffn_gate_up"
    assert "PREFILL_DBUF" not in art.os.environ
    assert art.os.environ["PREFILL_WMMA_LDS_PRIMITIVE"] == "1"
    fake_postrange._WARMSTART_OPTS[(frozenset({512, 4}), 3)] = ("lds",)
    return FakeTensor(x.data.reshape(512, 3).astype(np.float32) @ lin._pf16_w.data.astype(np.float32).T)

  fake_route_mod.route_pf16_graph_gemm = fake_route
  monkeypatch.setitem(sys.modules, "tinygrad", fake_tinygrad)
  monkeypatch.setitem(sys.modules, "tinygrad.codegen", fake_codegen)
  monkeypatch.setitem(sys.modules, "tinygrad.helpers", fake_helpers)
  monkeypatch.setitem(sys.modules, "tinygrad.codegen.opt", fake_codegen_opt)
  monkeypatch.setitem(sys.modules, "tinygrad.codegen.opt.postrange", fake_postrange)
  monkeypatch.setitem(sys.modules, "extra.qk.prefill_graph_gemm_route", fake_route_mod)
  monkeypatch.setattr(art, "describe_prefill_schedule", lambda n, k, role: types.SimpleNamespace(route_family="lds"))
  monkeypatch.setattr(art, "extract_wmma_lds_spec", lambda spec: object())
  monkeypatch.setattr(art, "wmma_lds_generated_env_defaults", lambda spec: {"PREFILL_TC_LOCAL_STAGE": "both"})
  monkeypatch.setenv("PREFILL_DBUF", "1")
  monkeypatch.setenv("PREFILL_WMMA_LDS_PRIMITIVE", "old")
  monkeypatch.setenv("PREFILL_TC_LOCAL_STAGE", "old-stage")

  result = art.run_lds_route_sample_correctness(m=512, n=4, k=3, sample_cols=2)

  assert result["passed"] is True
  assert result["dbuf_enabled"] is False
  assert result["uses_hand_lds_oracle"] is False
  assert result["warmstart_key_present_after_route"] is True
  assert art.os.environ["PREFILL_DBUF"] == "1"
  assert art.os.environ["PREFILL_WMMA_LDS_PRIMITIVE"] == "old"
  assert art.os.environ["PREFILL_TC_LOCAL_STAGE"] == "old-stage"
  assert fake_postrange._WARMSTART_OPTS == {"old": "warmstart"}
