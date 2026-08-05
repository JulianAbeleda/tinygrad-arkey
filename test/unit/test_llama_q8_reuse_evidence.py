import json
from pathlib import Path


def test_pinned_llama_d512_trace_has_one_q8_node_per_mmvq_consumer():
  root = Path(__file__).resolve().parents[2]
  row = json.loads((root / "docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json").read_text())
  summary = row["summary"]
  assert summary["llama_quantize_q8_1"] == 217
  assert summary["llama_mmvq"] == 217
  assert summary["llama_observed_cross_mmv_q8_reuse"] is False
  assert {x["activation"]["observed_reuse_consumers"] for x in row["rows"]} == {1}
