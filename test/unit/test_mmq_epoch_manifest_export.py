import json

import pytest

import extra.qk.mmq_epoch_manifest_export as manifest_export
from extra.qk.mmq_epoch_manifest_export import (
  DEFAULT_MAX_ROWS, ROW_SCHEMA, SCHEMA, build_amd_isa_proof_manifest_bundle,
  export_current_amd_isa_proof_manifest_bundle, summarize_amd_isa_proof_rows, validate_amd_isa_proof_rows)


def _row(**overrides):
  return {
    "schema": ROW_SCHEMA,
    "kind": "wmma",
    "logical_op": "V_WMMA",
    "emitted": "v_wmma_f32_16x16x16_f16 ...",
    "dest_reg": 10,
    "source_regs": [1, 2, 3],
    **overrides,
  }


def test_build_bundle_packages_rows_and_optional_hashes():
  source_sha = "a" * 64
  binary_sha = "0" * 64

  bundle = build_amd_isa_proof_manifest_bundle(
    candidate_id="mmq.q4k.q8.epoch0",
    kernel_name="mmq_q4k_q8",
    source_sha256=source_sha,
    binary_sha256=binary_sha,
    rows=[_row(kind="global_load", logical_op="GLOBAL_LOAD"), _row(kind="waitcnt", logical_op="WAITCNT")],
  )

  assert bundle["schema"] == SCHEMA
  assert bundle["candidate_id"] == "mmq.q4k.q8.epoch0"
  assert bundle["kernel_name"] == "mmq_q4k_q8"
  assert bundle["source_sha256"] == source_sha
  assert bundle["binary_sha256"] == binary_sha
  assert [row["kind"] for row in bundle["rows"]] == ["global_load", "waitcnt"]
  assert json.loads(json.dumps(bundle))["rows"][0]["logical_op"] == "GLOBAL_LOAD"


def test_build_bundle_omits_absent_optional_hashes_and_copies_rows():
  row = _row()

  bundle = build_amd_isa_proof_manifest_bundle(candidate_id="c0", kernel_name="kernel", rows=[row])
  row["kind"] = "mutated"

  assert "source_sha256" not in bundle
  assert "binary_sha256" not in bundle
  assert bundle["rows"][0]["kind"] == "wmma"


@pytest.mark.parametrize("missing", ["schema", "kind", "logical_op", "emitted"])
def test_validate_rows_rejects_missing_required_fields(missing):
  row = _row()
  del row[missing]

  with pytest.raises(ValueError, match=rf"rows\[0\].{missing} missing"):
    validate_amd_isa_proof_rows([row])


def test_validate_rows_rejects_wrong_renderer_row_schema():
  with pytest.raises(ValueError, match="schema must be"):
    validate_amd_isa_proof_rows([_row(schema="other")])


def test_validate_rows_rejects_empty_required_strings():
  with pytest.raises(ValueError, match=r"rows\[0\].kind must be a non-empty string"):
    validate_amd_isa_proof_rows([_row(kind="")])


def test_build_bundle_rejects_bad_identity_and_hashes():
  with pytest.raises(ValueError, match="candidate_id must be a non-empty string"):
    build_amd_isa_proof_manifest_bundle(candidate_id="", kernel_name="kernel", rows=[])
  with pytest.raises(ValueError, match="source_sha256 must be a lowercase hex sha256 string"):
    build_amd_isa_proof_manifest_bundle(candidate_id="c0", kernel_name="kernel", source_sha256="A" * 64, rows=[])


def test_validate_rows_enforces_explicit_bound():
  with pytest.raises(ValueError, match="rows exceeds max_rows=1"):
    validate_amd_isa_proof_rows([_row(kind="a"), _row(kind="b")], max_rows=1)


def test_default_bound_is_finite():
  assert isinstance(DEFAULT_MAX_ROWS, int)
  assert DEFAULT_MAX_ROWS > 0


def test_export_current_uses_renderer_helpers_with_synthetic_rows(monkeypatch):
  reset_calls = []
  monkeypatch.setattr(manifest_export, "amd_isa_proof_manifest", lambda: (_row(kind="barrier", logical_op="BARRIER"),))
  monkeypatch.setattr(manifest_export, "reset_amd_isa_proof_manifest", lambda: reset_calls.append("reset"))

  bundle = export_current_amd_isa_proof_manifest_bundle(candidate_id="c0", kernel_name="kernel", reset_after=True)

  assert bundle["rows"][0]["kind"] == "barrier"
  assert reset_calls == ["reset"]


def test_bundle_carries_and_validates_explicit_abi_ownership_and_digests():
  digest = "b" * 64
  bundle = build_amd_isa_proof_manifest_bundle(
    candidate_id="c0", kernel_name="kernel",
    rows=[_row(operand_id="A", source_operand_id="arg0", fetch_group="k0", width_bytes=4,
               vector_width_bytes=16, retained_fragment={"bytes": 32}, semantic_owner={"role": "lhs"})],
    abi_metadata={"arguments": {"arg0": {"offset": 0}}}, ownership_metadata={"A": "arg0"},
    digest_metadata={"candidate_sha256": digest, "semantic_schedule_sha256": "c" * 64})

  assert bundle["abi_metadata"]["arguments"]["arg0"]["offset"] == 0
  assert bundle["ownership_metadata"] == {"A": "arg0"}
  assert bundle["digest_metadata"]["candidate_sha256"] == digest


@pytest.mark.parametrize("kwargs, match", [
  ({"abi_metadata": []}, "abi_metadata must be a mapping"),
  ({"ownership_metadata": {1: "arg0"}}, "ownership_metadata key must be a non-empty string"),
  ({"digest_metadata": {"candidate_sha256": "bad"}}, "lowercase hex sha256"),
])
def test_bundle_rejects_invalid_identity_metadata(kwargs, match):
  with pytest.raises(ValueError, match=match):
    build_amd_isa_proof_manifest_bundle(candidate_id="c0", kernel_name="kernel", rows=[], **kwargs)


def test_rows_reject_invalid_explicit_operand_path_metadata():
  with pytest.raises(ValueError, match="width_bytes must be a positive integer"):
    validate_amd_isa_proof_rows([_row(operand_id="A", width_bytes=0)])


def test_structure_summary_counts_final_rows_and_only_exports_explicit_operand_metadata():
  summary = summarize_amd_isa_proof_rows([
    _row(kind="global_load", emitted="global_load_b128", source_regs=[6, 7]),
    _row(kind="ds_load_b128", emitted="ds_load_b128", operand_id="B", source_operand_id="arg1"),
    _row(kind="waitcnt", emitted="s_waitcnt vmcnt(0)"), _row(kind="barrier", emitted="s_barrier"),
    _row(kind="wmma", emitted="v_wmma", semantic_ownership={"A": "lhs", "B": "rhs"}),
  ])
  assert summary["counts"] == {"global_load": 1, "ds_load": 1, "ds_store": 0, "wait": 1, "barrier": 1, "wmma": 1}
  assert len(summary["operand_paths"]) == 2
  assert summary["operand_paths"][0]["row_index"] == 1
  assert all("source_regs" not in row for row in summary["operand_paths"])
