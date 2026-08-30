#!/usr/bin/env python3
"""D0 executable forced-cut substrate for the exact 18 FFN-down roles.

Each invocation is a fresh model process.  The four cuts are cumulative:
producer, main, publication, and residual.  The runner never invents device
time: HCQ and Buffer observers are installed before importing tinygrad and
their JSON is retained as the evidence source.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys

CUTS = ("producer", "main", "publication", "residual")
ROLES = tuple(f"blk.{i}.ffn_down" for i in range(18))
MARKER_ABI = "tinygrad.nv_compiler_q6k_boundary_marker.v1"

def _read_jsonl(path):
  p = pathlib.Path(path)
  if not p.exists(): return []
  rows=[]
  for line in p.read_text().splitlines():
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: pass
  return rows

def _run(a, arm, cut, profile, buffers, result):
  env = dict(os.environ)
  # These must precede every tinygrad import in the child.
  env.update({"HCQ_SUBMISSION_OBSERVER_JSON": str(profile), "BUFFER_OBSERVER": "1",
    "HCQ_NUM_COMPUTE": "2", "HCQ_NV_READY_PLACEMENT": "0",
    "NV_COMPILER_Q4_IMMA_PP512": "1", "NV_COMPILER_Q4_IMMA_K_PP512": "1",
    "NV_COMPILER_Q6_IMMA_PP512": "1", "NV_COMPILER_Q6_IMMA_PP512_ROLES": "ffn_down",
    "PROFILE": "1", "BUFFER_OBSERVER_JSON": str(buffers),
    "NV_Q6DOWN_FORCED_CUT": cut})
  if arm == "fp16": env.pop("NV_COMPILER_Q6_IMMA_PP512", None); env.pop("NV_COMPILER_Q6_IMMA_PP512_ROLES", None)
  cmd=[sys.executable, "-m", "extra.llm_research.prefill.nv_compiler_q6k_model_arm",
       "--arm", "candidate" if arm == "q6" else "control", "--roles", "ffn_down",
       "--model", a.model, "--max-context", str(a.max_context), "--warmups", "1",
       "--rounds", "9", "--out", str(result)]
  if arm == "fp16": cmd.remove("--roles"); cmd.remove("ffn_down")
  return subprocess.run(cmd, env=env, cwd=a.repo, text=True, capture_output=True, check=False)

def main():
  p=argparse.ArgumentParser(); p.add_argument("--repo", default="/home/ubuntu/tinygrad-arkey")
  p.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  p.add_argument("--max-context", type=int, default=4608); p.add_argument("--out", required=True)
  a=p.parse_args(); root=pathlib.Path(a.out); root.parent.mkdir(parents=True,exist_ok=True)
  rows=[]
  for cut in CUTS:
    for arm in ("fp16", "q6"):
      d=root.parent / f"d0-{cut}-{arm}"; d.mkdir(parents=True,exist_ok=True)
      profile=d/"hcq.jsonl"; buffers=d/"buffer.jsonl"; result=d/"model.json"
      cp=_run(a, arm, cut, profile, buffers, result)
      payload=json.loads(result.read_text()) if result.exists() else {}
      hcq=_read_jsonl(profile); be=_read_jsonl(buffers)
      entries=[e for x in hcq for e in x.get("entries",[])]
      rows.append({"cut":cut,"arm":arm,"roles":list(ROLES),"role_count":18,
        "marker_identity":f"{MARKER_ABI}:{cut}:v1","returncode":cp.returncode,
        "model_status":payload.get("status"),"hcq_records":len(entries),
        "device_timestamps_observed":bool(entries and all(e.get("start") and e.get("end") for e in entries)),
        "dependencies_observed":bool(entries and all("deps" in e for e in entries)),
        "buffer_events":len(be),"stdout_tail":cp.stdout[-1000:],"stderr_tail":cp.stderr[-1000:]})
  out={"schema":"tinygrad.nv_q6down_forced_cut_model.v1","packet":"D0",
    "status":"PASS" if all(r["returncode"]==0 and r["device_timestamps_observed"] and r["dependencies_observed"] for r in rows) else "BLOCKED",
    "cuts":list(CUTS),"cumulative":True,"exact_roles":list(ROLES),"arms":["fp16","q6"],
    "evidence":rows,"decision":"D0 executability and G0/G1 only; no composition or performance verdict."}
  pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"packet":"D0","status":out["status"],"out":a.out}))
if __name__ == "__main__": main()
