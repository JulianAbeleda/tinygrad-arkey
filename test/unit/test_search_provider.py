import io, json, subprocess, sys
import pytest
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


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal objc bridge requires macOS libSystem.dylib")
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

def test_exact_model_cache_reuses_stat_identity_reloads_on_drift_and_rejects_hash_mismatch(monkeypatch, tmp_path):
  path = tmp_path / "model.gguf"; path.write_bytes(b"first")
  import extra.llm_research.semantic_workload as semantic
  loads = []
  def load(workload, source):
    if workload["model_hash"] == "wrong": raise ValueError("GGUF content hash mismatch")
    loads.append((workload["model_hash"], str(source)))
    return SimpleNamespace(model_hash=workload["model_hash"])
  def bind(_workload, _path, *, model): return SimpleNamespace(model_hash=model.model_hash)
  monkeypatch.setattr(semantic, "load_exact_gguf_model", load)
  monkeypatch.setattr(semantic, "bind_exact_gguf_workload", bind)
  adapter = provider.MetalAdapter()
  exact = {"workload": {"model_hash": "good", "fixture_shape_substitution": "forbidden"}, "path": str(path)}
  assert adapter._exact_binding(exact).model_hash == "good"
  assert adapter._exact_binding(exact).model_hash == "good" and len(loads) == 1
  adapter._prepared_cache["old"] = object(); adapter._compile_cache["old"] = {"old": True}
  path.write_bytes(b"changed-size")
  assert adapter._exact_binding(exact).model_hash == "good" and len(loads) == 2
  assert not adapter._prepared_cache and not adapter._compile_cache
  with pytest.raises(ValueError, match="content hash mismatch"):
    adapter._exact_binding({"workload": {"model_hash": "wrong", "fixture_shape_substitution": "forbidden"}, "path": str(path)})

@pytest.mark.parametrize("field", ["model_sha256", "phase", "role", "profile"])
def test_exact_semantic_admission_rejects_candidate_identity_mismatches(monkeypatch, field):
  facts = {"backend":"METAL","architecture":"Apple9","name":"test","max_threads_per_threadgroup":64}
  monkeypatch.setattr(provider, "_metal_device", lambda: object()); monkeypatch.setattr(provider, "_metal_facts", lambda _d:facts)
  identity={"target_id":"x","backend":"METAL","arch":"Apple9","subgroup_size":32,"resolved_target_hash":"h"}
  candidate=_metal_candidate(identity); candidate["workload"].update({"model_sha256":"m","phase":"decode","profile":"mod","accumulator_dtype":"float32"}); candidate["correctness"]={"atol":1,"rtol":1}
  semantic={"model_hash":"m","target":identity,"fixture_shape_substitution":"forbidden","tolerance":{"atol":1,"rtol":1},"semantic_identity":{"phase":"decode","role":"decode_q4k_gemv","module_path":"mod","logical_m":1,"logical_n":256,"logical_k":256,"source_quant_storage":"Q4_K","source_layout":None,"accumulator_dtype":"float32"}}
  candidate["workload"][field]="bad"
  payload={"candidate":candidate,"candidate_hash":provider._canonical_candidate_hash(candidate),"target_identity":identity,"exact_gguf":{"workload":semantic,"path":"x"}}
  assert provider.process(request("admit",payload),adapter=provider.MetalAdapter())["error"]["code"]=="identity_mismatch"

def test_prepared_cache_key_is_stable_then_invalidates_exact_file_identity(tmp_path):
  path = tmp_path / "model.gguf"; path.write_bytes(b"one")
  identity={"target_id":"x","backend":"METAL","arch":"a","subgroup_size":32,"resolved_target_hash":"h"}
  payload=_metal_payload(identity); payload["exact_gguf"]={"workload":{"model_hash":"a"*64},"path":str(path)}
  adapter=provider.MetalAdapter(); first=adapter._prepared_key(payload); assert first==adapter._prepared_key(payload)
  path.write_bytes(b"two-size")
  assert first != adapter._prepared_key(payload)

def test_prepared_cache_builds_once_and_reuses_same_object(monkeypatch):
  identity={"target_id":"x","backend":"METAL","arch":"a","subgroup_size":32,"resolved_target_hash":"h"}
  payload=_metal_payload(identity); adapter=provider.MetalAdapter(); builds=[]; fake=object()
  monkeypatch.setattr(provider.MetalAdapter, "_build_prepared", lambda _self, _payload: (builds.append(1) or fake))
  assert adapter._prepared(payload) is fake and adapter._prepared(payload) is fake and len(builds)==1


# CUDA/NV flash adapter ------------------------------------------------------
def _flash_descriptor(**tile_kwargs):
  from extra.llm_research import flash_candidate_schema as schema
  defaults = {"Hq": 8, "Hd": 128, "Hkv": 8, "MAXC": 4096, "split_count": 1}
  defaults.update(tile_kwargs)
  # Build without validating so invalid-geometry fixtures reach the adapter.
  return {"schema_version": schema.SCHEMA_VERSION, "tile": schema.tile_fields(**defaults), "combine": None}


def _nv_facts(**overrides):
  facts = {"subgroup_size": 32, "max_threads_per_threadgroup": 1024,
           "max_threadgroup_memory_bytes": 227 * 1024, "shuffle_supported": True, "fdot2_supported": True}
  facts.update(overrides)
  return facts


def _flash_payload(descriptor=None, facts=None, **tile_kwargs):
  from extra.llm_research import flash_candidate_schema as schema
  descriptor = descriptor if descriptor is not None else _flash_descriptor(**tile_kwargs)
  envelope = schema.candidate_envelope(descriptor)
  return {"candidate": envelope, "candidate_hash": envelope["candidate_hash"],
          "target_facts": facts if facts is not None else _nv_facts()}


def test_cuda_describe_supplies_caller_supplied_target_facts():
  facts = _nv_facts(architecture="sm_120", backend="CUDA")
  out = provider.process(request("describe", {"target_facts": facts}), adapter=provider.CudaAdapter())
  assert out["status"] == "ok"
  target = out["result"]["target"]
  assert target["subgroup_size"] == 32
  assert target["max_threads_per_threadgroup"] == 1024
  assert target["max_threadgroup_memory_bytes"] == 227 * 1024
  assert target["shuffle_supported"] is True and target["fdot2_supported"] is True
  assert target["architecture"] == "sm_120"
  assert out["result"]["backend_live"] is False


def test_cuda_describe_and_admit_fail_closed_without_facts():
  adapter = provider.CudaAdapter()
  blocked = provider.process(request("describe"), adapter=adapter)
  assert blocked["error"]["code"] == "hardware_absent"
  payload = _flash_payload()
  payload.pop("target_facts")
  admitted = provider.process(request("admit", payload), adapter=adapter)
  assert admitted["error"]["code"] == "hardware_absent"
  malformed = provider.process(request("admit", _flash_payload(facts={"subgroup_size": "wide"})), adapter=adapter)
  assert malformed["error"]["code"] == "admission_rejected"


def test_cuda_admit_validates_geometry_against_supplied_facts():
  adapter = provider.CudaAdapter()
  out = provider.process(request("admit", _flash_payload()), adapter=adapter)
  assert out["status"] == "ok"
  geometry = out["result"]["flash_geometry"]
  assert geometry["lane_width"] == 32 and geometry["threads"] == 32
  assert geometry["local_memory_bytes"] == 2 * 16 * 128 * 2
  assert len(out["result"]["candidate_hash"]) == 64
  assert out["result"]["plan_hash"] == out["result"]["candidate_hash"]


def test_cuda_admit_rejects_over_threads_over_local_memory_and_cross_subgroup_lanes():
  adapter = provider.CudaAdapter()
  over_threads = provider.process(request("admit", _flash_payload(warps=64)), adapter=adapter)
  assert over_threads["error"]["code"] == "resource_limit"
  over_local = provider.process(request("admit", _flash_payload(token_block=512)), adapter=adapter)
  assert over_local["error"]["code"] == "resource_limit"
  cross_subgroup = provider.process(request("admit", _flash_payload(lane_width=64)), adapter=adapter)
  assert cross_subgroup["error"]["code"] == "resource_limit"


def test_cuda_admit_rejects_bad_geometry_and_hash_mismatch_with_classified_codes():
  adapter = provider.CudaAdapter()
  invalid = provider.process(request("admit", _flash_payload(reduce_structure="recursive")), adapter=adapter)
  assert invalid["error"]["code"] == "admission_rejected"
  payload = _flash_payload()
  payload["candidate_hash"] = "f" * 64
  forged = provider.process(request("admit", payload), adapter=adapter)
  assert forged["error"]["code"] == "identity_mismatch"
  wrong_schema = provider.process(request("admit", {"candidate": {"schema_version": "other.v1"},
                                                    "candidate_hash": "f" * 64, "target_facts": _nv_facts()}), adapter=adapter)
  assert wrong_schema["error"]["code"] == "admission_rejected"


@pytest.mark.parametrize("action", ["compile", "check", "measure"])
def test_cuda_compile_check_measure_fail_closed_without_live_backend(action):
  adapter = provider.CudaAdapter()
  out = provider.process(request(action, _flash_payload()), adapter=adapter)
  assert out["status"] == "blocked"
  assert out["error"]["code"] == "backend_unavailable"
  assert out["error"]["details"]["action"] == action


def test_cuda_canned_success_path_is_deterministic_and_needs_no_gpu():
  adapter = provider.CudaAdapter(live_backend=True)
  payload = _flash_payload()
  first = provider.process(request("compile", payload), adapter=adapter)
  second = provider.process(request("compile", payload), adapter=adapter)
  assert first["status"] == "ok" and second["status"] == "ok"
  assert first["result"]["compiler"] == "canned_fake_backend"
  assert first["result"]["source_sha256"] == second["result"]["source_sha256"]
  assert first["result"]["launch"] == second["result"]["launch"]
  measured = provider.process(request("measure", payload), adapter=adapter)
  again = provider.process(request("measure", payload), adapter=adapter)
  assert measured["result"]["samples_ns"] == again["result"]["samples_ns"]
  assert measured["result"]["timing_mode"] == "canned_fake_backend_deterministic_no_gpu"
  assert measured["result"]["work_bytes"]["status"] == "estimated"
  checked = provider.process(request("check", payload), adapter=adapter)
  assert checked["status"] == "ok" and checked["result"]["correct"] is True
  assert checked["result"]["oracle"].startswith("canned flash reference")


def test_cuda_adapter_accepts_injected_backend_hooks_and_preserves_envelope_errors():
  def fake_compile(_payload, descriptor):
    return {"compiler": "fake_cubin", "source_sha256": descriptor["tile"]["Hd"] * "d"}
  def fake_measure(_payload, descriptor):
    return {"timing_mode": "fake", "samples_ns": [11, 22, 33], "summary_ns": {"min": 11},
            "work_bytes": {"status": "exact", "provenance": "fake hook", "operations": 1, "bytes": 2}}
  adapter = provider.CudaAdapter(live_backend=True, compile_fn=fake_compile, measure_fn=fake_measure)
  compiled = provider.process(request("compile", _flash_payload()), adapter=adapter)
  assert compiled["result"]["compiler"] == "fake_cubin"
  assert compiled["result"]["source_sha256"] == "d" * 128
  measured = provider.process(request("measure", _flash_payload()), adapter=adapter)
  assert measured["result"]["timing_mode"] == "fake" and measured["result"]["samples_ns"] == [11, 22, 33]

  class Exploding:
    def compile(self, _payload):
      raise RuntimeError("fake compiler crashed")
  crashed = provider.process(request("compile", _flash_payload()), adapter=Exploding())
  assert crashed["error"] == {"code": "provider_failure",
                              "message": "provider adapter raised an unexpected exception",
                              "retryable": False, "details": {"exception_type": "RuntimeError"}}


def test_cuda_worker_registers_backend_via_main_without_any_gpu():
  import pathlib
  worker = pathlib.Path(provider.__file__)
  facts = _nv_facts()
  lines = json.dumps(request("describe", {"target_facts": facts})) + "\n" + \
          json.dumps(request("compile", _flash_payload())) + "\n"
  response = subprocess.run([sys.executable, str(worker), "--backend", "CUDA"], input=lines, text=True,
                            stdout=subprocess.PIPE, check=True).stdout
  rows = [json.loads(line) for line in response.splitlines()]
  assert rows[0]["status"] == "ok"
  assert rows[0]["result"]["target"]["subgroup_size"] == 32
  assert rows[0]["result"]["target"]["fdot2_supported"] is True
  assert rows[1]["status"] == "blocked"
  assert rows[1]["error"]["code"] == "backend_unavailable"
