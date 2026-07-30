import io, json
from types import SimpleNamespace

from extra.llm_research import search_provider as provider


def request(action="describe", payload=None, request_id="r1"):
  return {"protocol": provider.PROTOCOL, "request_id": request_id, "action": action, "payload": payload or {}}


class MockAdapter:
  def describe(self, payload): return {"kind": "mock", "payload": dict(payload)}
  def admit(self, payload): return {"phase": "admit", "candidate": payload["candidate"]}
  def compile(self, payload): return {"phase": "compile", "source_sha256": "a" * 64}
  def check(self, payload): return {"phase": "check", "correct": True}
  def measure(self, payload): return {"phase": "measure", "samples_ns": [10, 11],
                                      "work_bytes": {"status": "estimated", "provenance": "mock model", "operations": 42, "bytes": 24}}


def test_each_canonical_action_uses_the_same_target_neutral_adapter_seam():
  adapter = MockAdapter()
  for action in provider.ACTIONS:
    payload = {"candidate": {"schema": "candidate.v2"}} if action != "describe" else {"target": {"backend": "mock"}}
    out = provider.process(request(action, payload), adapter=adapter)
    assert out["status"] == "ok"
    assert out["action"] == action
    assert out["result"]


def test_work_and_byte_evidence_is_orthogonal_to_timing_and_rejects_roofline_policy_fields():
  out = provider.process(request("measure", {"candidate": {}}), adapter=MockAdapter())
  assert out["result"]["samples_ns"] == [10, 11]
  assert out["result"]["work_bytes"] == {"status": "estimated", "provenance": "mock model", "operations": 42, "bytes": 24}

  class BadEvidence(MockAdapter):
    def measure(self, payload): return {"work_bytes": {"status": "exact", "provenance": "bad", "bytes": 1, "peak_bandwidth": 2}}
  blocked = provider.process(request("measure", {"candidate": {}}), adapter=BadEvidence())
  assert blocked["error"]["code"] == "invalid_evidence"


def test_default_shell_describes_but_fails_closed_for_unimplemented_gpu_actions():
  assert provider.process(request())["status"] == "ok"
  out = provider.process(request("compile", {"candidate": {}}))
  assert out["status"] == "blocked"
  assert out["error"]["code"] == "backend_unavailable"
  assert out["error"]["details"]["action"] == "compile"


def test_malformed_envelope_unknown_action_and_adapter_crash_are_structured_errors():
  malformed = provider.process({"protocol": provider.PROTOCOL})
  unknown = provider.process(request("promote"))
  class Broken:
    def describe(self, payload): raise RuntimeError("boom")
  crashed = provider.process(request(), adapter=Broken())
  assert malformed["error"]["code"] == "invalid_envelope"
  assert unknown["error"]["code"] == "unsupported_action"
  assert crashed["error"] == {"code": "provider_failure", "message": "provider adapter raised an unexpected exception",
                               "retryable": False, "details": {"exception_type": "RuntimeError"}}


def test_json_lines_keeps_rows_independent_and_never_emits_non_json():
  source = io.StringIO(json.dumps(request()) + "\nnot-json\n" + json.dumps(request("measure", {"candidate": {}}, "r3")) + "\n")
  sink = io.StringIO()
  assert provider.serve(source, sink) == 0
  rows = [json.loads(line) for line in sink.getvalue().splitlines()]
  assert [(row["request_id"], row["status"]) for row in rows] == [("r1", "ok"), (None, "blocked"), ("r3", "blocked")]
  assert rows[1]["error"]["code"] == "invalid_json"


def _metal_candidate(identity, *, threads=32, op="UPCAST", quant="q4_k", shape=None, heuristic=False):
  candidate = {"schema_version": "boltbeam.full_kernel_candidate.v2", "workload": {
    "profile": "qwen3_8b_q4k_m_apple_m4", "role": "decode_q4k_gemv", "operation": "matmul", "shape": shape or {"m": 1, "n": 256, "k": 256},
    "operands": {"a": {"dtype": "fp16"}, "b": {"dtype": quant, "quantization": quant}, "c": {"dtype": "fp16"}}, "target": identity},
    "schedule": {"plan_kind": "tinygrad_heuristic.v1" if heuristic else "tinygrad_opt_sequence.v1", "transforms": [] if heuristic else [{"op": op, "axis": 0, "arg": 4}], "launch": {"threads": threads}}}
  return candidate

def _metal_payload(identity, **kwargs):
  candidate = _metal_candidate(identity, **kwargs)
  return {"candidate": candidate, "candidate_hash": provider._canonical_candidate_hash(candidate), "target_identity": identity}


def test_metal_adapter_admits_only_live_exact_target_and_compiler_owned_plan(monkeypatch):
  facts = {"backend": "METAL", "architecture": "Apple9", "name": "test", "max_threads_per_threadgroup": 64}
  monkeypatch.setattr(provider, "_metal_device", lambda: object())
  monkeypatch.setattr(provider, "_metal_facts", lambda _device: facts)
  adapter = provider.MetalAdapter()
  identity = {"target_id": "apple_test", "backend": "METAL", "arch": "Apple9", "subgroup_size": 32}
  identity["resolved_target_hash"] = "boltbeam-owned-hash"
  out = provider.process(request("admit", _metal_payload(identity)), adapter=adapter)
  assert out["status"] == "ok"
  assert out["result"]["compiler_opts"] == [{"op": "UPCAST", "axis": 0, "arg": 4}]

  too_large = provider.process(request("admit", _metal_payload(identity, threads=65)), adapter=adapter)
  assert too_large["error"]["code"] == "resource_limit"
  bad_op = provider.process(request("admit", _metal_payload(identity, op="MSL")), adapter=adapter)
  assert bad_op["error"]["code"] == "unsupported_plan"


def test_metal_target_facts_expose_replay_strategy_and_counters_without_hardware(monkeypatch):
  from tinygrad.runtime import ops_metal
  from tinygrad.runtime.graph import metal as metal_graph
  monkeypatch.setattr(ops_metal, "from_ns_str", lambda _value: "mock-metal")
  monkeypatch.setattr(metal_graph, "metal_replay_facts", lambda _device: {
    "schema":"tinygrad.metal_replay.v1", "configured_strategy":"hybrid_icb_direct", "experimental_ab":True,
    "icb_offset_bits":32, "last_graph":{"graph_calls":323, "icb_calls":256, "direct_encoded_calls":67, "committed":True}})
  sysdevice = SimpleNamespace(name=lambda:object(), registryID=lambda:1, maxThreadsPerThreadgroup=lambda:SimpleNamespace(width=64),
    maxThreadgroupMemoryLength=lambda:32768, recommendedMaxWorkingSetSize=lambda:1 << 30, currentAllocatedSize=lambda:1 << 20)
  facts = provider._metal_facts(SimpleNamespace(sysdevice=sysdevice, arch="Apple9"))
  assert facts["replay"]["configured_strategy"] == "hybrid_icb_direct"
  assert facts["replay"]["last_graph"] == {"graph_calls":323, "icb_calls":256, "direct_encoded_calls":67, "committed":True}


def test_metal_adapter_hardware_compile_is_explicitly_opt_in():
  # Kept out of normal CI: this reaches the local Apple compiler/runtime.
  import os
  import pytest
  if os.getenv("RUN_METAL_PROVIDER_TESTS") != "1": pytest.skip("set RUN_METAL_PROVIDER_TESTS=1 on a Metal host")
  adapter = provider.MetalAdapter()
  facts = adapter.describe({})["target"]
  identity = {"target_id":"apple_m4_10c", "backend":"METAL", "arch":facts["architecture"], "subgroup_size":facts["subgroup_width"], "resolved_target_hash":"boltbeam-live"}
  out = provider.process(request("compile", _metal_payload(identity)), adapter=adapter)
  assert out["status"] == "ok"
  assert len(out["result"]["source_sha256"]) == len(out["result"]["mtlb_sha256"]) == 64
  for quant in ("q4_k", "q6_k"):
    checked = provider.process(request("check", _metal_payload(identity, quant=quant)), adapter=adapter)
    assert checked["status"] == "ok" and checked["result"]["correct"] is True
  for heuristic in (False, True):
    real = _metal_payload(identity, shape={"m": 1, "n": 4096, "k": 4096}, heuristic=heuristic)
    assert provider.process(request("compile", real), adapter=adapter)["status"] == "ok"
  streamed = provider.process(request("measure", {"measurement_kind": "practical_streaming", "target_identity": identity,
                                                     "source_bytes": 4 << 20, "samples": 2, "warmups": 0}), adapter=adapter)
  assert streamed["status"] == "ok"
  assert streamed["result"]["cache_capacity_bytes"] is None and streamed["result"]["timing_mode"].endswith("wait_true")

  import pathlib, subprocess, sys
  worker = pathlib.Path(provider.__file__)
  line = json.dumps(request("describe")) + "\n"
  response = subprocess.run([sys.executable, str(worker), "--backend", "METAL"], input=line, text=True,
                            stdout=subprocess.PIPE, check=True).stdout
  assert json.loads(response)["result"]["target"]["backend"] == "METAL"
