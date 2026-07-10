import json

import pytest

from extra.qk.mmq_lifecycle import (
  COUNTER_NAMES, DEFAULT_ROUTE_ID, EPOCH_COUNTERS, MMQLifecycleRow, SCHEMA,
  aggregate_lifecycle_rows, build_lifecycle_report, validate_lifecycle_rows, zero_counters)


def _row(role: str, tile_id: str, **overrides: int) -> MMQLifecycleRow:
  return MMQLifecycleRow(role=role, tile_id=tile_id, counters=zero_counters(**overrides))


def test_mmq_lifecycle_counter_schema_matches_m5_scope_names():
  assert COUNTER_NAMES == (
    "activation_quant_epochs",
    "activation_q8_1_global_writes",
    "activation_q8_1_reads",
    "packed_weight_global_loads",
    "scale_min_metadata_loads",
    "dot_accumulation_epochs",
    "dot_ops_or_packed_dot_insts",
    "barriers",
    "intermediate_global_writes",
    "output_store_epochs",
    "output_stores",
    "duplicate_quant_work",
    "duplicate_dequant_or_scale_work",
    "split_k_reductions",
  )
  assert EPOCH_COUNTERS == (
    "activation_quant_epochs",
    "packed_weight_global_loads",
    "scale_min_metadata_loads",
    "dot_accumulation_epochs",
    "output_store_epochs",
  )


def test_mmq_lifecycle_report_aggregates_by_role_and_tile_and_serializes():
  rows = [
    _row("ffn_gate_up", "m0_n0_k0", activation_quant_epochs=1, activation_q8_1_global_writes=8,
         activation_q8_1_reads=16, packed_weight_global_loads=32, scale_min_metadata_loads=4,
         dot_accumulation_epochs=1, dot_ops_or_packed_dot_insts=64, barriers=2,
         output_store_epochs=1, output_stores=8),
    _row("ffn_gate_up", "m0_n0_k1", activation_q8_1_reads=16, packed_weight_global_loads=32,
         scale_min_metadata_loads=4, dot_accumulation_epochs=1, dot_ops_or_packed_dot_insts=64,
         barriers=1, duplicate_quant_work=1, duplicate_dequant_or_scale_work=2, split_k_reductions=1),
    _row("attn_qo", "m0_n1_k0", activation_quant_epochs=1, activation_q8_1_global_writes=8,
         activation_q8_1_reads=8, packed_weight_global_loads=16, scale_min_metadata_loads=2,
         dot_accumulation_epochs=1, dot_ops_or_packed_dot_insts=32, output_store_epochs=1,
         output_stores=8),
  ]

  report = build_lifecycle_report(rows)

  assert report["schema"] == SCHEMA
  assert report["route_id"] == DEFAULT_ROUTE_ID
  assert report["ok"] is True
  assert report["errors"] == []
  assert report["aggregation"]["total"]["dot_ops_or_packed_dot_insts"] == 160
  assert report["aggregation"]["total"]["duplicate_quant_work"] == 1
  assert report["aggregation"]["total"]["duplicate_dequant_or_scale_work"] == 2
  assert report["aggregation"]["by_role"]["ffn_gate_up"]["packed_weight_global_loads"] == 64
  assert report["aggregation"]["by_role"]["attn_qo"]["output_stores"] == 8
  assert report["aggregation"]["by_tile"]["m0_n0_k1"]["split_k_reductions"] == 1
  assert json.loads(json.dumps(report))["counter_names"] == list(COUNTER_NAMES)


def test_mmq_lifecycle_validation_reports_missing_and_negative_counters_without_raising():
  counters = zero_counters()
  del counters["output_stores"]
  counters["barriers"] = -1
  row = MMQLifecycleRow(role="ffn_gate_up", tile_id="m0_n0_k0", counters=counters)

  errors = validate_lifecycle_rows([row])
  report = build_lifecycle_report([row], validate=False)

  assert "$.tiles[0].counters.output_stores missing" in errors
  assert "$.tiles[0].counters.barriers must be non-negative" in errors
  assert report["ok"] is False
  assert report["aggregation"] is None
  assert report["errors"] == errors


def test_mmq_lifecycle_build_report_raises_on_invalid_rows_by_default():
  counters = zero_counters(activation_quant_epochs=-1)
  row = MMQLifecycleRow(role="ffn_gate_up", tile_id="m0_n0_k0", counters=counters)

  with pytest.raises(ValueError, match="activation_quant_epochs must be non-negative"):
    build_lifecycle_report([row])


def test_mmq_lifecycle_validation_reports_non_integer_counters():
  row = MMQLifecycleRow(
    role="ffn_gate_up", tile_id="m0_n0_k0",
    counters={**zero_counters(), "activation_quant_epochs": "one"})

  errors = validate_lifecycle_rows([row])

  assert "$.tiles[0].counters.activation_quant_epochs must be an integer" in errors


def test_mmq_lifecycle_rejects_unknown_counter_override():
  with pytest.raises(KeyError, match="unknown MMQ lifecycle counter"):
    zero_counters(not_a_counter=1)


def test_mmq_lifecycle_aggregate_rejects_malformed_identity():
  row = {"role": "", "tile_id": "m0_n0_k0", "counters": zero_counters()}

  with pytest.raises(ValueError, match="role must be a non-empty string"):
    aggregate_lifecycle_rows([row])
