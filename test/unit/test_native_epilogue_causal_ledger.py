import json
from pathlib import Path

from extra.llm_research.decode.native_epilogue_causal_ledger import build


def test_epilogue_ownership_is_exact_and_not_a_credit():
  root=Path(__file__).resolve().parents[2]
  p=build(json.loads((root/"docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json").read_text()))
  assert p["ownership"]["native_serialized_us"] == 240.762
  assert p["ownership"]["llama_exposed_shapley_us"] == 0.443
  assert p["ownership"]["difference_us"] == 240.319
  assert p["native_credit_us"] == 0.0
  assert p["status"] == "CLOSED_NO_ADMISSIBLE_GENERIC_EPILOGUE_AB"


def test_closed_evidence_cannot_be_treated_as_additive_recovery():
  root=Path(__file__).resolve().parents[2]
  p=build(json.loads((root/"docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json").read_text()))
  assert any(e["result"] == "NO_GO" and e["route"] == "NV" for e in p["evidence"])
  assert any(e["result"] == "EPILOGUE_NEUTRAL" for e in p["evidence"])
  assert "without a custom-program boundary" in p["next_blocker"]
