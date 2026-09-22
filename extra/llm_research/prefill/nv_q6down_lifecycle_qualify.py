#!/usr/bin/env python3
"""Cheap D-lifecycle falsifier before the full hot/cold forced-cut matrix."""
from __future__ import annotations
import argparse, json, os, pathlib, statistics, subprocess, sys

ROUNDS = 9
SAFE_ENV = {
  "QK_PRIMITIVE":"1", "NV_COMPILER_Q4_IMMA_PP512":"1", "NV_COMPILER_Q4_IMMA_K_PP512":"1",
  "NV_COMPILER_Q4_IMMA_QO_PP512":"1", "NV_COMPILER_Q4_IMMA_UNROLL":"4",
  "NV_COMPILER_Q6_IMMA_PP512":"1", "NV_COMPILER_Q6_IMMA_PP512_ROLES":"ffn_down",
  "NV_Q6DOWN_FORCED_CUT":"residual", "HCQ_NUM_COMPUTE":"2", "HCQ_NV_READY_PLACEMENT":"0", "PROFILE":"1"}

def _jsonl(path:pathlib.Path) -> list[dict]:
  if not path.exists(): return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def _run(repo:pathlib.Path, model:str, root:pathlib.Path, name:str, observed:bool) -> dict:
  arm_dir=root/name; arm_dir.mkdir(parents=True, exist_ok=False)
  model_json, buffer_jsonl, hcq_jsonl = arm_dir/"model.json", arm_dir/"buffer.jsonl", arm_dir/"hcq.jsonl"
  env=dict(os.environ); env.update(SAFE_ENV); env.pop("NV_Q4_IMMA_PP512", None); env.pop("NV_UNROLL", None)
  env["HCQ_NV_MULTI_QUEUE_CUT_POLICY"] = str(repo/"docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-flash-direct-deps-cut-v2.json")
  env["NV_Q6DOWN_OBSERVER_ARM"] = name
  for key in ("BUFFER_OBSERVER", "BUFFER_OBSERVER_JSON", "HCQ_SUBMISSION_OBSERVER_JSON"): env.pop(key, None)
  if observed:
    env.update({"BUFFER_OBSERVER":"1", "BUFFER_OBSERVER_JSON":str(buffer_jsonl),
                "HCQ_SUBMISSION_OBSERVER_JSON":str(hcq_jsonl)})
  cmd=[sys.executable, "-m", "extra.llm_research.prefill.nv_compiler_q6k_model_arm", "--arm", "candidate",
       "--roles", "ffn_down", "--model", model, "--max-context", "4608", "--warmups", "1",
       "--rounds", str(ROUNDS), "--out", str(model_json)]
  cp=subprocess.run(cmd, cwd=repo, env=env, text=True, capture_output=True)
  payload=json.loads(model_json.read_text()) if model_json.exists() else {}
  (arm_dir/"process.json").write_text(json.dumps({"command":cmd, "returncode":cp.returncode,
    "stdout_tail":cp.stdout[-2000:], "stderr_tail":cp.stderr[-2000:]}, indent=2)+"\n")
  return {"name":name, "observed":observed, "returncode":cp.returncode, "model":payload,
          "buffer_path":str(buffer_jsonl), "hcq_path":str(hcq_jsonl)}

def _observed_census(arm:dict) -> dict:
  buffer_rows=_jsonl(pathlib.Path(arm["buffer_path"]))
  timed=[row for row in buffer_rows if row.get("context",{}).get("phase")=="timed"]
  buffer_counts={kind:sum(row.get("bytes",0) for row in timed if row.get("kind")==kind) for kind in ("alloc","copyin","copyout")}
  buffer_events={kind:sum(row.get("kind")==kind for row in timed) for kind in ("alloc","copyin","copyout")}
  submissions=[row for row in _jsonl(pathlib.Path(arm["hcq_path"])) if row.get("schema")=="tinygrad.hcq_submission_observer.v1"]
  timed_submissions=submissions[-ROUNDS:]
  entries=[entry for row in timed_submissions for entry in row.get("entries",[])]
  classified=all((entry.get("metadata") or {}).get("graph_kind") in ("program","graph_copy") for entry in entries)
  graph_copies=[entry for entry in entries if (entry.get("metadata") or {}).get("graph_kind")=="graph_copy"]
  graph_copy_bytes=sum(int((entry.get("metadata") or {}).get("copy_bytes",0)) for entry in graph_copies)
  graph_copy_us=sum(float(entry.get("duration",0)) for entry in graph_copies)
  materialization_events=buffer_events["alloc"]+len(graph_copies)
  samples=sorted({row.get("context",{}).get("sample") for row in timed if row.get("context",{}).get("sample") is not None})
  return {"buffer_callback_events":buffer_events, "buffer_callback_bytes":buffer_counts,
    "timed_buffer_samples":samples, "submission_records":len(submissions), "timed_submission_records":len(timed_submissions),
    "timed_graph_entries":len(entries), "graph_kind_complete":classified, "graph_copy_events":len(graph_copies),
    "graph_copy_bytes":graph_copy_bytes, "graph_copy_device_us":graph_copy_us,
    "materialization_definition":"timed Buffer alloc plus HCQ Ops.COPY graph entries",
    "materialization_events":materialization_events}

def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=pathlib.Path,default=pathlib.Path("/home/ubuntu/tinygrad-arkey"))
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
  arms=[_run(args.repo,args.model,args.out,"control_0_observer_off",False),
        _run(args.repo,args.model,args.out,"candidate_1_observer_on",True),
        _run(args.repo,args.model,args.out,"control_2_observer_off",False)]
  models=[arm["model"] for arm in arms]
  correctness=all(arm["returncode"]==0 and model.get("status")=="PASS" and model.get("token")==198 for arm,model in zip(arms,models))
  medians=[model.get("wall",{}).get("median_ms") for model in models]
  controls=[medians[0],medians[2]] if all(isinstance(x,(int,float)) for x in medians) else []
  control_median=statistics.median(controls) if controls else None
  overhead_pct=None if control_median is None else max(0.0,(medians[1]-control_median)/control_median*100.0)
  control_drift_ms=None if not controls else abs(controls[0]-controls[1])
  census=_observed_census(arms[1])
  substrate_pass=correctness and census["timed_submission_records"]==ROUNDS and census["graph_kind_complete"] and overhead_pct is not None and overhead_pct<2.0
  recurring=sum(census["buffer_callback_events"].values())+census["graph_copy_events"]
  status="PASS" if substrate_pass and recurring>0 else "STOP"
  if not substrate_pass: decision="STOP: observer correctness, coverage, classification, or perturbation gate failed."
  elif recurring==0: decision="STOP: fresh timed R9 contains no allocation, copyin, copyout, graph-copy, or buffer materialization event; lifecycle is not a recurring prefill lever."
  else: decision="PASS: a recurring lifecycle event survives the cheap falsifier; authorize the full hot/cold boundary matrix only."
  result={"schema":"tinygrad.nv_q6down_lifecycle_qualification.v1", "packet":"D-L0", "status":status,
    "authority":{"model":args.model,"gpu":"NVIDIA GeForce RTX 5090, sm_120","environment":SAFE_ENV,
      "order":[arm["name"] for arm in arms]}, "correctness":{"status":"PASS" if correctness else "STOP","token":198},
    "observer":{"substrate_status":"PASS" if substrate_pass else "STOP","overhead_pct":overhead_pct,
      "control_drift_ms":control_drift_ms,"medians_ms":dict(zip((arm["name"] for arm in arms),medians)),"ceiling_pct":2.0},
    "lifecycle":census, "decision":decision, "next_packet":"D-L1-full-matrix" if status=="PASS" else None}
  (args.out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  (args.out/"report.md").write_text("# Q6-down lifecycle lever qualification\n\n"
    f"Status: **{status}**\n\n{decision}\n\n"
    f"Observer overhead: `{overhead_pct}` percent. Timed Buffer events: `{sum(census['buffer_callback_events'].values())}`. "
    f"Timed graph-copy events: `{census['graph_copy_events']}`.\n")
  print(json.dumps({"status":status,"result":str(args.out/"result.json"),"decision":decision}))
  if not substrate_pass: raise SystemExit(1)

if __name__=="__main__": main()
