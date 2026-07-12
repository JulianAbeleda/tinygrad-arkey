from extra.qk.prefill.ffn_gate_up_epoch_graph import build_epoch_graph


def test_exact_structural_graph_covers_all_gate_up_epochs_and_slots():
  report = build_epoch_graph()
  assert report["workload"] == {"role": "ffn_gate_up", "m": 512, "n": 12288, "k": 4096,
                                "tile_k": 32, "epoch_count": 128, "active_buffers": 2}
  assert report["dbuf_checker"]["ok"] is True
  reaching = [edge for edge in report["edges"] if edge["kind"] == "structural_reaching_definition"]
  assert len(reaching) == 256
  assert report["claims"]["structural_reaching_definitions_complete"] is True


def test_graph_exposes_value_and_lowered_identity_loss():
  report = build_epoch_graph()
  assert report["claims"]["complete"] is False
  assert report["claims"]["value_reaching_definitions_complete"] is False
  assert report["claims"]["lowered_instruction_correlation_complete"] is False
  assert report["identity_loss"]["count"] == 512
  assert all(row["field"] == "value_key" for row in report["identity_loss"]["records"])


def test_graph_composes_existing_audit_summaries_without_reinterpreting_them():
  stage = {"summary": {"stage_owner_ready": False}}
  trace = {"lds_reaching_def_map": {"key_strength": "addr_family", "limitation": "identity lost",
                                    "load_count": 4, "covered_load_count": 3, "missing_load_count": 1,
                                    "wmma_missing_a_count": 1, "wmma_missing_b_count": 0}}
  report = build_epoch_graph(stage_owner_audit=stage, lifecycle_trace=trace)
  assert report["stage_owner_audit_summary"] == stage["summary"]
  assert report["lowered_reaching_def_summary"]["missing_load_count"] == 1
  assert report["claims"]["lowered_instruction_correlation_complete"] is False
