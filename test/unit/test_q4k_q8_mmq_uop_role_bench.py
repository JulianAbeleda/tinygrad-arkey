import json, subprocess
from types import SimpleNamespace

import numpy as np

from extra.qk import q4k_q8_mmq_uop_role_bench as bench
from extra.qk.q4k_q8_mmq_uop_role_compile_gate import EXPECTED_NK, ROLE_ORDER, RoleShape


def _shape(role="attn_kv"):
  return RoleShape(role,512,*EXPECTED_NK[role],"Q4_K")


def _row(shape, warmups=2, samples=3):
  return {"role":shape.role,"shape":shape.to_json(),
    "target":{"program_count":1,"program_name":shape.kernel_name,"fallback_used":False,"warmup_launch_count":warmups,
              "timing":{"samples":samples,"measured_launch_counts":[1]*samples,"device_time_trustworthy":True}},
    "scalar_authority":{"program_count":1,"launch_count":1,"fallback_used":False},
    "full_output_correctness":{"reference":"scalar_direct_uop_same_operands_full_output","allclose":False,
      "rel_rmse_threshold":bench.REL_RMSE_THRESHOLD,"rel_rmse_pass":True,"finite":True,"nan_count":0,
      "mismatch_count":349,"max_abs":0.009},
    "independent_subset_correctness":{"reference":"independent_packed_byte_reference","shape":[8,8],
      "rel_rmse_threshold":bench.REL_RMSE_THRESHOLD,"wmma_rel_rmse_pass":True,"scalar_rel_rmse_pass":True,
      "all_finite":True,"nan_count":0},
    "provenance":{"independent_random_byte_bounded_authority":
      "extra.qk.q4k_q8_mmq_uop_validation:independent_packed_byte_reference"}}


def test_fixture_is_deterministic_and_has_loaded_role_sizes():
  shape=_shape(); a=bench.deterministic_fixture(shape); b=bench.deterministic_fixture(shape)
  assert all(np.array_equal(x,y) for x,y in zip(a,b))
  assert a[0].shape == (shape.n*(shape.k//256)*36,)
  assert a[1].shape == (512,5120) and a[2].shape == (512,160)


def test_summary_requires_one_measured_launch_and_reports_tflops():
  row=bench.summarize_samples([.002,.001,.003],[.0015,.001,.002],[1,1,1],2_000_000_000)
  assert row["wall_median_ms"] == 2 and row["device_median_ms"] == 1.5
  assert row["logical_wall_tflops"] == 1 and row["device_time_trustworthy"]
  assert not bench.summarize_samples([.001],[.0008],[2],1000)["device_time_trustworthy"]


def test_worker_contract_uses_rel_rmse_but_preserves_strict_diagnostics():
  shape=_shape(); good=_row(shape)
  assert bench.validate_worker(shape,good,warmups=2,samples=3) == (True,None)
  assert good["full_output_correctness"]["allclose"] is False
  assert good["full_output_correctness"]["mismatch_count"] == 349
  mutations=[]
  for key,value in (("program_count",2),("fallback_used",True)):
    row=json.loads(json.dumps(good)); row["target"][key]=value; mutations.append(row)
  row=json.loads(json.dumps(good)); row["target"]["timing"]["measured_launch_counts"]=[1,2,1]; mutations.append(row)
  row=json.loads(json.dumps(good)); row["full_output_correctness"]["rel_rmse_pass"]=False; mutations.append(row)
  row=json.loads(json.dumps(good)); row["full_output_correctness"]["nan_count"]=1; mutations.append(row)
  row=json.loads(json.dumps(good)); row["independent_subset_correctness"]["wmma_rel_rmse_pass"]=False; mutations.append(row)
  row=json.loads(json.dumps(good)); row["independent_subset_correctness"]["scalar_rel_rmse_pass"]=False; mutations.append(row)
  row=json.loads(json.dumps(good)); row["provenance"]={}; mutations.append(row)
  assert all(not bench.validate_worker(shape,row,warmups=2,samples=3)[0] for row in mutations)


def test_parent_starts_attn_kv_and_stops_at_first_failure(monkeypatch):
  shapes=tuple(_shape(r) for r in ROLE_ORDER); monkeypatch.setattr(bench,"derive_role_shapes",lambda *_a,**_k:shapes)
  calls=[]
  def runner(cmd,**kwargs):
    role=cmd[cmd.index("--role")+1]; calls.append(role); row=_row(_shape(role))
    if role == "attn_qo": row["full_output_correctness"]["rel_rmse_pass"]=False
    return SimpleNamespace(returncode=0,stdout=json.dumps(row),stderr="")
  health=lambda *_a,**_k:SimpleNamespace(returncode=0,stdout="GPU use ok",stderr="")
  out=bench.run_gate("unused",warmups=2,samples=3,timeout_seconds=7,runner=runner,health_checker=health,env={})
  assert not out["passed"] and calls == ["attn_kv","attn_qo"]
  assert out["first_failure"] == "attn_qo: full-output WMMA/scalar relative-RMSE correctness failed"


def test_canonical_rel_rmse_threshold_is_not_redeclared():
  assert bench.REL_RMSE_THRESHOLD == 6e-3
  got=np.array([1.001,-2.002],dtype=np.float32); ref=np.array([1.,-2.],dtype=np.float32)
  assert bench._rel_rmse(got,ref) < bench.REL_RMSE_THRESHOLD


def test_timeout_and_gpu_health_never_advance(monkeypatch):
  monkeypatch.setattr(bench,"derive_role_shapes",lambda *_a,**_k:(_shape(),_shape("attn_qo")))
  healthy=lambda *_a,**_k:SimpleNamespace(returncode=0,stdout="ok",stderr="")
  def timeout(*_a,**_k): raise subprocess.TimeoutExpired(["python"],3)
  out=bench.run_gate("unused",runner=timeout,health_checker=healthy,env={},timeout_seconds=3)
  assert out["first_failure"] == "attn_kv: timed out after 3s"
  unhealthy=lambda *_a,**_k:SimpleNamespace(returncode=1,stdout="",stderr="bad")
  out=bench.run_gate("unused",runner=lambda *_a,**_k:None,health_checker=unhealthy,env={})
  assert out["first_failure"] == "GPU health preflight failed"
