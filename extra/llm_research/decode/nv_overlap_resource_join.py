"""CPU-only fail-closed resource join for the NV decode-overlap question."""
from __future__ import annotations
import json
from pathlib import Path

REQUIRED=("grid","block","registers_per_thread","static_smem_bytes","dynamic_smem_bytes","local_mem_bytes")

def resource_complete(row:dict) -> bool:
  return all(isinstance(row.get(k),int if k.endswith("bytes") or k == "registers_per_thread" else list) for k in REQUIRED)

def build(manifest:dict, timeline:dict, tinygrad_census:dict) -> dict:
  llama=[r for r in manifest["rows"] if r["model_role"] != "vocab"]
  llama_complete=sum(resource_complete({**r["llama"]["mmvq"],"local_mem_bytes":None}) for r in llama)
  # The captured tinygrad logical census has symbols/durations but deliberately
  # no launch-resource tuple. Do not invent it from source or PTX virtual regs.
  tg_complete=0
  hidden=timeline["classes"]["quantize_q8_1"]["hidden_behind_mmq_us"]
  return {"schema":"tinygrad.nv_overlap_resource_join.v1","status":"INCONCLUSIVE_FAIL_CLOSED",
    "llama_overlap_mass_us":{"quantize_q8_1_behind_mmvq":hidden},
    "llama_mmvq_rows":len(llama),"llama_complete_resource_rows":llama_complete,
    "tinygrad_complete_resource_rows":tg_complete,
    "tinygrad_logical_nodes":tinygrad_census["kernels_per_token_prime"],
    "finding":"No pair can be ranked resource-compatible: the pinned llama trace lacks local-memory metadata and the tinygrad logical-ready/census artifact lacks grid/block/register/shared/local tuples. The premise that a pair is resource-compatible only in llama is not established.",
    "cheapest_decisive_probe":"CPU-only first: augment one aligned tinygrad capture with per-call compiled resource metadata and join it to the existing logical-ready pairs; only if a positive independent pair has complementary CTA limits, run one native two-queue A/B span probe."}

def main():
  root=Path(__file__).parents[3]
  out=build(json.loads((root/'docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json').read_text()),
    json.loads((root/'docs/task_workflow/output/nv-decode-llama-d512-timeline-ledger-20260804.json').read_text()),
    json.loads((root/'docs/task_workflow/output/nv-decode-overlap-b3-0-census-cuda-20260804.json').read_text()))
  print(json.dumps(out,indent=2))
if __name__ == '__main__': main()
