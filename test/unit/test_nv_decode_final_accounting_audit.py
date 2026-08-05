import json
from pathlib import Path


def test_final_accounting_audit_equation_closes():
  payload = json.loads((Path(__file__).parents[2] / "docs/task_workflow/output/nv-decode-final-accounting-audit-20260805.json").read_text())
  e = payload["equation_us"]
  device = (e["support_exposure_delta"] + e["quantized_core_delta"] + e["llama_internal_gaps_credit"] +
            e["profile_to_unprofiled_device_reconciliation"])
  wall = device + e["outside_window_delta"] + e["outer_reconciliation"]
  assert abs(device - e["device_delta"]) < 1e-9
  assert abs(wall - e["authority_wall_gap"]) < 1e-9
  assert abs(wall - e["reconciled_wall_gap"]) < 1e-9
  assert abs(e["closure_error"]) < 1e-9


def test_predispatch_credit_is_combined_only():
  payload = json.loads((Path(__file__).parents[2] / "docs/task_workflow/output/nv-decode-final-accounting-audit-20260805.json").read_text())
  assert payload["provisional_engineering_remainder_if_token_contract_is_accepted_us"] == 1646.17 - 69.1655
