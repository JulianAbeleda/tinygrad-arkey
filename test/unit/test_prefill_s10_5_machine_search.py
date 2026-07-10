import json

from extra.qk.prefill.s10_5_machine_search import (
  CLASSIFICATION, DEFAULT_OUTPUT, SCHEMA, build_authority_gate, build_ffn_gate_up_candidate, build_final_report,
  build_search_report, main)


def test_ffn_gate_up_candidate_schema_and_route_family():
  record = build_ffn_gate_up_candidate()

  assert record["schema"] == SCHEMA
  assert record["role"] == "ffn_gate_up"
  assert record["shape"] == {"M": 512, "N": 12288, "K": 4096}
  assert record["schedule_spec"]["route_family"] == "lds"
  assert record["lds_spec"]["ownership_classification"] == CLASSIFICATION
  assert record["legality_errors"] == []


def test_ffn_gate_up_candidate_backend_atom_classification():
  record = build_ffn_gate_up_candidate()

  assert record["classification"] == "compiler_primitive_spec_owned__asm_backend_atom"
  assert record["selected_backend_atom"]["name"] == "asm_backend_atom"
  assert record["selected_backend_atom"]["classification"] == "compiler_primitive_spec_owned__asm_backend_atom"
  assert record["selected_backend_atom"]["runtime_emission_changed"] is False
  assert record["dbuf_epoch_primitive"]["classification"] == "hand_coded_dbuf_epoch_primitive"


def test_ffn_gate_up_candidate_active_buffers_two_proof_ok():
  record = build_ffn_gate_up_candidate()
  proof = record["slot_identity_proof"]

  assert proof["active_buffers"] == 2
  assert proof["ok"] is True
  assert proof["dbuf_slot_identity_proven"] is True
  assert proof["dbuf_cadence_proven"] is False
  assert record["dbuf_checker_metadata"]["ok"] is True
  assert record["dbuf_checker_metadata"]["active_buffers"] == 2


def test_ffn_gate_up_candidate_makes_no_pure_generated_claim():
  record = build_ffn_gate_up_candidate()

  assert record["pure_generated"] is False
  assert "pure_generated" not in record["promotion_status"]
  assert "pure generated" not in record["promotion_reason"].lower()
  assert "ASM backend atom" in record["not_pure_generated_reason"]


def test_cli_writes_default_named_candidate_file(tmp_path, monkeypatch, capsys):
  monkeypatch.chdir(tmp_path)

  record = main(["--json"])
  printed = json.loads(capsys.readouterr().out)
  written = json.loads((tmp_path / DEFAULT_OUTPUT).read_text())

  assert printed["schema"] == SCHEMA
  assert written == record
  assert written["role"] == "ffn_gate_up"
  assert written["schedule_spec"]["route_family"] == "lds"


def test_authority_gate_accepts_hybrid_route_above_4k_even_if_generic_binding_failed():
  gate = build_authority_gate({
    "pin_clock": True,
    "whole_tok_s": {"512": 4410.25, "4096": 3233.46},
    "route_attribution": {"prefill_route_family": "prefill_pipe_role_selective_generated"},
    "prefill_route_binding_gate": {"verdict": "PREFILL_ROUTE_BINDING_FAIL"},
  })

  assert gate["ok"] is True
  assert gate["route_ok"] is True
  assert gate["perf_ok"] is True
  assert gate["binding_gate_verdict"] == "PREFILL_ROUTE_BINDING_FAIL"
  assert "hybrid/backend-atom" in gate["binding_gate_note"]


def test_final_report_ready_when_candidate_and_authority_gate_pass():
  report = build_final_report(authority={
    "pin_clock": True,
    "whole_tok_s": {"512": 4410.25, "4096": 3233.46},
    "route_attribution": {"prefill_route_family": "prefill_pipe_role_selective_generated"},
    "prefill_route_binding_gate": {"verdict": "PREFILL_ROUTE_BINDING_FAIL"},
  })

  assert report["verdict"] == "S10_5_HYBRID_SEARCH_OWNED_BACKEND_ATOM_READY"
  assert report["classification"] == "compiler_primitive_spec_owned__asm_backend_atom"
  assert report["pure_generated"] is False
  assert report["full_fine_tuned_hand_kernel"] is False
  assert report["candidate_ok"] is True
  assert report["authority_gate"]["ok"] is True


def test_search_report_enumerates_s9_safe_wait_policy_candidates():
  report = build_search_report()

  assert report["schema"] == "prefill-s10.5-machine-search-report.v1"
  assert report["search_space"] == "s9_safe_wait_policy_over_backend_atom"
  assert report["classification"] == "compiler_primitive_spec_owned__asm_backend_atom"
  assert report["pure_generated"] is False
  assert report["candidate_count"] == 3
  assert [c["candidate_id"] for c in report["candidates"]] == [
    "wait-default",
    "wait-lgkm-coop-store-2",
    "wait-lgkm-frag-load-2",
  ]
  assert all((c["slot_identity_proof"] or {})["ok"] for c in report["candidates"])
  assert all(c["search_knobs"]["runtime_emission_changed"] is False for c in report["candidates"])


def test_search_report_recommends_candidates_with_prior_4k_authority_band():
  report = build_search_report()
  summary = {row["candidate_id"]: row for row in report["summary"]}

  assert report["verdict"] == "S10_5_SEARCH_READY_FOR_AUTHORITY"
  assert summary["wait-default"]["recommended_for_authority"] is True
  assert summary["wait-lgkm-coop-store-2"]["recommended_for_authority"] is True
  assert summary["wait-lgkm-frag-load-2"]["recommended_for_authority"] is True
  assert summary["wait-lgkm-coop-store-2"]["env_overrides"] == {"PREFILL_LDS2_WAIT_LGKM_COOP_STORE": "2"}
  assert summary["wait-lgkm-frag-load-2"]["env_overrides"] == {"PREFILL_LDS2_WAIT_LGKM_FRAG_LOAD": "2"}
  assert float(summary["wait-default"]["prior_pp512"]) >= 4000
