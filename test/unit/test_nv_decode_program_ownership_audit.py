from extra.llm_research.decode.nv_decode_program_ownership_audit import _native_marker_sha, audit


def _row(name, source):
  return {"ordinal":0, "program_hash":"ab"*32, "program_name":name, "source_sha256":source,
          "binary_sha256":"cd"*32, "global_size":[1,1,1], "local_size":[32,1,1]}


def test_nonmarker_source_transport_remains_unknown():
  row=_row("rendered", "ef"*32)
  result=audit({"capture":{"selected_jits":["rollout"], "programs_by_jit":{"rollout":1},
                           "program_evidence_by_jit":{"rollout":[row]}}})
  assert result["no_known_native_precompiled_marker_in_selected_graph"]
  assert result["programs"][0]["transport_provenance"] == "unknown_source_transport"


def test_native_program_marker_is_detected():
  row=_row("oracle", _native_marker_sha("oracle"))
  result=audit({"capture":{"selected_jits":["rollout"], "programs_by_jit":{"rollout":1},
                           "program_evidence_by_jit":{"rollout":[row]}}})
  assert not result["no_known_native_precompiled_marker_in_selected_graph"]
  assert result["native_precompiled_programs"][0]["program_name"] == "oracle"


def test_exact_source_recompile_positively_proves_transport():
  import hashlib
  source="rendered source"; binary=b"compiled binary"
  row=_row("rendered",hashlib.sha256(source.encode()).hexdigest())
  row["binary_sha256"]=hashlib.sha256(binary).hexdigest()
  class Compiler:
    def compile(self, actual): assert actual == source; return binary
  result=audit({"capture":{"selected_jits":["rollout"],"programs_by_jit":{"rollout":1},
    "program_evidence_by_jit":{"rollout":[row]},"source_text_by_sha256":{row["source_sha256"]:source}}},Compiler())
  assert result["all_unique_programs_source_recompiled"]
  assert result["programs"][0]["transport_provenance"] == "source_binary_recompiled_match"


def test_source_recompile_mismatch_fails_closed():
  import hashlib
  source="source"; row=_row("rendered",hashlib.sha256(source.encode()).hexdigest())
  class Compiler:
    def compile(self, actual): return b"different"
  result=audit({"capture":{"selected_jits":["rollout"],"programs_by_jit":{"rollout":1},
    "program_evidence_by_jit":{"rollout":[row]},"source_text_by_sha256":{row["source_sha256"]:source}}},Compiler())
  assert not result["all_unique_programs_source_recompiled"]
  assert result["programs"][0]["transport_provenance"] == "source_recompile_mismatch"


def test_tampered_source_table_is_rejected_before_compile():
  import pytest
  row=_row("rendered","ef"*32)
  class Compiler:
    def compile(self, actual): raise AssertionError("must not compile tampered source")
  with pytest.raises(ValueError,match="does not match"):
    audit({"capture":{"selected_jits":["rollout"],"programs_by_jit":{"rollout":1},
      "program_evidence_by_jit":{"rollout":[row]},"source_text_by_sha256":{row["source_sha256"]:"tampered"}}},Compiler())
