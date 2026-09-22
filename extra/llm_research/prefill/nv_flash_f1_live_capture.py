#!/usr/bin/env python3
"""Fail-closed F1.1 live graph capture adapter.

The route runner may hand this adapter a finalized TinyJit graph dump.  It
never executes a reconstructed/full-T512 Flash launch; only existing PROGRAM
nodes and their graph-owned bindings are serialized.
"""
import argparse, hashlib, json
from pathlib import Path

PROGRAM = "nv_sm120_q16_grid_hd128_loop_attention"

def capture(graph_path, out_path):
  try:
    doc = json.loads(Path(graph_path).read_text())
    rows = doc.get("graph_calls", doc.get("calls", []))
    selected = []
    for i, row in enumerate(rows):
      if row.get("name", row.get("program")) != PROGRAM: continue
      # Buffer metadata must come from the finalized graph object.
      binding=row.get("graph_owned_slice", row.get("binding", {}))
      if binding.get("source") not in ("graph", "graph-owned"):
        raise RuntimeError(f"call {i}: finalized graph has no graph-owned binding")
      required=("layer","q_head","kv_head","causal_length","shapes","strides","dtypes","buffer_hashes","output_shape","buffers")
      missing=[key for key in required if key not in row]
      if missing: raise RuntimeError(f"call {i}: missing graph metadata {','.join(missing)}")
      if not all(isinstance(buf,dict) and "base" in buf and "offset" in buf for buf in row["buffers"]):
        raise RuntimeError(f"call {i}: graph buffer/view metadata is incomplete")
      selected.append(row | {"capture_ordinal": len(selected)})
    status = "PASS" if len(selected) == 36 else "BLOCKED"
    decision = ("PASS: exact 36 finalized installed Flash calls captured" if status == "PASS"
                else f"BLOCKED: finalized graph exposed {len(selected)} installed Flash calls; exact 36 required")
    result = {"schema":"tinygrad.nv_prefill_flash_f1_live_capture.v1","packet":"F1.1",
      "status":status,"program":PROGRAM,"exact_population":36,"calls":selected,
      "census":{"predicted":36,"observed":len(selected)},"full_t512_relaunch":False,
      "decision":decision,"source_graph":str(graph_path),
      "capture_sha256":hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()}
  except Exception as exc:
    result = {"schema":"tinygrad.nv_prefill_flash_f1_live_capture.v1","packet":"F1.1",
      "status":"STOP","program":PROGRAM,"exact_population":36,"calls":[],
      "census":{"predicted":36,"observed":0},"full_t512_relaunch":False,
      "decision":"STOP: " + str(exc),"source_graph":str(graph_path)}
  Path(out_path).parent.mkdir(parents=True,exist_ok=True)
  Path(out_path).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,sort_keys=True)); return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("graph"); ap.add_argument("--out",required=True)
  a=ap.parse_args(); raise SystemExit(capture(a.graph,a.out))
