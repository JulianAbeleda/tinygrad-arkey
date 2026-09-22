import json
from pathlib import Path

from extra.llm_research.decode.native_flash_causal_ledger import build

ROOT=Path(__file__).resolve().parents[2]

def test_settled_flash_ledger_proves_overlap_is_sufficient():
  out=build(*[json.loads((ROOT/p).read_text()) for p in (
    "docs/task_workflow/output/nv-decode-native-semantic-profile-ledger-20260805.json",
    "docs/task_workflow/output/nv-decode-llama-d512-timeline-ledger-20260804.json",
    "docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json")])
  assert out["status"] == "PASS_OVERLAP_SUFFICIENT_BODY_PARITY_UNPROVEN"
  assert out["comparisons"]["raw_native_minus_llama_us"] < 0
  assert abs(out["comparisons"]["exposed_native_minus_llama_us"]-247.989) < .01
