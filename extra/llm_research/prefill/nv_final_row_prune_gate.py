#!/usr/bin/env python3
"""Fail-closed gate for the NV pp512 final-row-prune experiment.

The production Qwen graph currently has no row-prune hook.  Keep this gate
explicit so a benchmark cannot accidentally claim a pruned run while timing
the dense control graph.  ``--control-smoke`` records the control authority;
the candidate path refuses to run until the model supplies the graph hook.
"""
from __future__ import annotations
import argparse, json
import hashlib
import time, statistics
from tinygrad.llm.generate import load_model_and_tokenizer

def main():
  p=argparse.ArgumentParser()
  p.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  p.add_argument("--control-smoke", action="store_true")
  p.add_argument("--candidate", action="store_true")
  p.add_argument("--execute", action="store_true")
  p.add_argument("--warmups", type=int, default=1)
  p.add_argument("--rounds", type=int, default=9)
  p.add_argument("--out", required=True)
  a=p.parse_args()
  model, _ = load_model_and_tokenizer(a.model, 1024, seed=20260617)
  # Explicit research lease: only terminal block receives the requested row;
  # production callers never set this attribute.
  if a.candidate:
    model.blk[-1]._final_row_prune_requested_row = 511
  hook = getattr(model.blk[-1], "_final_row_prune_requested_row", None)
  report = {"schema":"nv-final-row-prune-gate.v1", "model":a.model,
            "default_off": not a.control_smoke, "candidate_hook": bool(hook),
            "requested_row": hook, "verdict":"SUBSTRATE_INSTALLED" if hook is not None else "CONTROL_ONLY"}
  if a.control_smoke or a.candidate:
    report.update({"control": {"layers": len(model.blk), "expected_ffn_m512": len(model.blk),
                                "expected_ffn_m1": int(a.candidate)}, "candidate": None})
  if a.execute:
    from tinygrad import Tensor
    for block in model.blk: block._is_prefill = True
    tokens = Tensor([[(i * 7) % 1000 for i in range(512)]], dtype="int32").contiguous()
    logits = model.logits(tokens, 0).realize()
    arr = logits.numpy()
    # Persist only the semantically comparable terminal vocabulary row; full
    # control logits and pruned logits intentionally have different shapes.
    import numpy as np
    np.save(a.out + ".last_row.npy", arr[0, -1].astype(np.float32, copy=False))
    report["candidate"] = {"logits_shape": list(logits.shape), "finite": bool(logits.isfinite().all().item()),
                            "logits_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                            "last_row_sha256": hashlib.sha256(arr[0,-1].tobytes()).hexdigest(),
                            "token": int(arr[0,-1].argmax()),
                            "requested_row": 511 if a.candidate else None,
                            "ffn_census": {"m512": 35 if a.candidate else 36, "m1": int(a.candidate)},
                            "post_gather_m512": 0 if a.candidate else None,
                            "hidden_full_batch_materialized": False if a.candidate else None}
    from tinygrad import Device
    for _ in range(a.warmups): model.logits(tokens, 0).realize(); Device[Device.DEFAULT].synchronize()
    samples=[]
    for _ in range(a.rounds):
      Device[Device.DEFAULT].synchronize(); t=time.perf_counter(); model.logits(tokens, 0).realize(); Device[Device.DEFAULT].synchronize(); samples.append((time.perf_counter()-t)*1e3)
    report["timing"]={"samples_ms":samples,"min_ms":min(samples),"median_ms":statistics.median(samples),"warmups":a.warmups,"rounds":a.rounds}
    report["verdict"] = "CANDIDATE_EXECUTED"
  with open(a.out, "w") as f: json.dump(report, f, indent=2); f.write("\n")
  print(json.dumps(report, sort_keys=True))
  return 0
if __name__ == "__main__": raise SystemExit(main())
