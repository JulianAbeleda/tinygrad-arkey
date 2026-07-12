import json, subprocess, sys

from extra.qk.prefill.full_kernel_candidate_worker import PROTOCOL
from test.unit.test_runtime_specs import _single_buffer_anchor_candidate, _strict_full_kernel_candidate

def _candidate(second=False):
  payload=_single_buffer_anchor_candidate().full_kernel_candidate
  assert payload is not None
  if second:
    payload["schedule"]["tile"]={"m":64,"n":64,"k":32};payload["schedule"]["waves"]={"m":2,"n":2};payload["schedule"]["threads"]=128
    payload["schedule"]["lds"]["windows"]={"a":[0,5120],"b":[5120,10240]};payload["schedule"]["lds"]["strides"]={"a":80,"b":80}
  return _strict_full_kernel_candidate(full_kernel_candidate=payload)

def _request(candidate,action="admit"):
  return {"protocol":PROTOCOL,"request_id":"req-1","action":action,
    "candidate":{"payload":candidate.full_kernel_candidate,"canonical_identity":candidate.canonical_identity},
    "workload":{"profile":"qwen3_8b_q4k_m_gfx1100","role":"ffn_gate_up","shape":[512,12288,4096],
                "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}}}

def _run(row):
  proc=subprocess.run([sys.executable,"-m","extra.qk.prefill.full_kernel_candidate_worker"],input=json.dumps(row),
    text=True,capture_output=True,check=True)
  assert proc.stderr == "" and len(proc.stdout.strip().splitlines()) == 1
  return json.loads(proc.stdout)

def test_worker_admits_dynamic_second_candidate_with_structured_plan():
  candidate=_candidate(True); out=_run(_request(candidate))
  assert out["ok"] and out["request_id"]=="req-1" and out["result"]["canonical_identity"]==candidate.canonical_identity
  assert out["result"]["capability_id"]=="amd.gfx1100.prefill.wmma_lds.single_buffer.v1"
  assert out["environment"]["capability"]["capability_id"]==out["result"]["capability_id"]
  assert out["result"]["plan"]=={"tile":[64,64,32],"waves":[2,2],"threads":128,"active_lds_bytes":10240,"subtiles":[2,2],"k_substeps":2}

def test_worker_rejects_unsupported_action_hash_and_malformed_requests():
  assert _run(_request(_candidate(),"evaluate"))["error"]["code"]=="unsupported_action"
  bad=_request(_candidate());bad["candidate"]["canonical_identity"]="0"*64
  out=_run(bad);assert out["error"]["class"]=="admission" and out["error"]["code"]=="identity_mismatch"
  assert _run({"protocol":PROTOCOL,"request_id":"x","action":"admit"})["error"]["code"]=="malformed_candidate"

def test_worker_classifies_unsupported_candidate_protocol_and_request_id():
  payload=_candidate().full_kernel_candidate;payload["schedule"]["pipeline"]["stage_count"]=2
  unsupported=_strict_full_kernel_candidate(full_kernel_candidate=payload)
  out=_run(_request(unsupported));assert out["error"]["class"]=="admission" and out["error"]["code"]=="capability_pipeline"
  row=_request(_candidate());row["protocol"]="worker.v0"
  assert _run(row)["error"]["code"]=="protocol_mismatch"
  row=_request(_candidate());row["request_id"]=""
  assert _run(row)["error"]["code"]=="invalid_request_id"

def test_worker_malformed_json_stdout_is_one_json_response():
  proc=subprocess.run([sys.executable,"-m","extra.qk.prefill.full_kernel_candidate_worker"],input="{",text=True,capture_output=True,check=True)
  out=json.loads(proc.stdout);assert out["ok"] is False and out["error"]["code"]=="malformed_json" and proc.stderr==""
