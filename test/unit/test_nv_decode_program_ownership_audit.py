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
