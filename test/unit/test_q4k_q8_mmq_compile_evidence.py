from types import SimpleNamespace

from extra.qk.q4k_q8_mmq_compile_evidence import build_q4k_q8_mmq_compile_evidence
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec


def spec():
  return Q4KQ8MMQPrefillSpec("ffn", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows", 16, 16, 256)


def test_descriptor_emitted_metadata_evidence_contains_identity_abi_geometry_and_resources():
  program = SimpleNamespace(arg=SimpleNamespace(function_name="generated_mmq", global_size=(16, 16, 1), local_size=(64, 1, 1)))
  row = build_q4k_q8_mmq_compile_evidence(spec(), program, metadata={k: 0 for k in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills", "wavefront_size")} | {"dynamic_stack": False, "lowering": "generated", "backend": "AMD"}, source="kernel", binary=b"code", instruction_summary={"instruction_count": 3}, candidate_id="candidate")
  assert row["status"] == "pass" and len(row["canonical_identity"]) == 64
  assert row["identity"]["source_sha256"] and row["identity"]["binary_sha256"]
  assert row["geometry"]["local_size"] == [64, 1, 1] and row["abi"]["arguments"]
  assert row["resources"]["vgpr_spills"] == 0


def test_descriptor_evidence_fails_closed_when_compiler_facts_are_missing():
  program = SimpleNamespace(arg=SimpleNamespace(function_name="generated_mmq", global_size=(1, 1, 1), local_size=(64, 1, 1)))
  row = build_q4k_q8_mmq_compile_evidence(spec(), program)
  assert row["status"] == "blocked"
  assert row["identity"]["binary_sha256"] is None
  assert row["resources"]["vgpr"] is None
