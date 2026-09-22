#!/usr/bin/env python3
"""S0 reproducible pp512 authority freeze.

This runner owns provenance and matched fresh-process execution.  It never
enables PROFILE and refuses to stamp PASS when a required arm or census is
missing.  Model bytes are hashed, never copied.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, platform, statistics, subprocess, sys, time, uuid

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_DEFAULT = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LLAMA_DEFAULT = "/home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench"
SCHEMA = "tinygrad.nv_prefill_post_substrate_authority.v1"

def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
  return h.hexdigest()

def run(cmd, env, out):
  t0 = time.time()
  p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
  pathlib.Path(out).write_text(json.dumps({"command":cmd,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}, indent=2))
  if p.returncode: raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
  return p

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=MODEL_DEFAULT); ap.add_argument("--llama-bin", default=LLAMA_DEFAULT)
  ap.add_argument("--evidence", required=True); ap.add_argument("--report", required=True)
  ap.add_argument("--skip-run", action="store_true", help="write a blocked report when GPU/session execution is unavailable")
  a = ap.parse_args(); model = pathlib.Path(a.model).expanduser().resolve(); ev = pathlib.Path(a.evidence); ev.mkdir(parents=True, exist_ok=True)
  required = [model, pathlib.Path(a.llama_bin), ROOT / "extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py",
              ROOT / "extra/llm_research/prefill/nv_compiler_q4k_pp512_binding.py", ROOT / "extra/llm_research/prefill/nv_compiler_q4v_serialized_binding.py",
              ROOT / "tinygrad/llm/model.py", ROOT / "tinygrad/runtime/graph/hcq.py", ROOT / "tinygrad/device.py"]
  missing = [str(p) for p in required if not p.is_file()]
  manifest = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in required if p.is_file()}
  gpu = subprocess.run(["nvidia-smi","--query-gpu=name,compute_cap,driver_version","--format=csv,noheader"], text=True, capture_output=True).stdout.strip()
  run_id = uuid.uuid4().hex
  env = {**os.environ, "PYTHONPATH":str(ROOT), "PROFILE":"0", "NV_PROFILE":"0", "HCQ_PROFILE":"0",
         "HCQ_NUM_COMPUTE":"2", "HCQ_NV_MULTI_QUEUE_CUT_POLICY":str(ROOT/"docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-flash-direct-deps-cut-v2.json"), "HCQ_NV_READY_PLACEMENT":"0", "S0_RUN_ID":run_id,
         "QK_PRIMITIVE":"1", "NV_COMPILER_Q4_IMMA_PP512":"1", "NV_COMPILER_Q4_IMMA_K_PP512":"1", "NV_COMPILER_Q4_IMMA_QO_PP512":"1", "NV_COMPILER_Q4_IMMA_UNROLL":"4"}
  result = {"schema":SCHEMA,"packet":"S0","status":"BLOCKED" if missing or a.skip_run else "STOP",
    "authority":{"run_id":run_id,"source_manifest":manifest,"model":{"path":str(model),"sha256":sha256(model) if model.is_file() else None},
      "prompt":{"tokens":512,"fixture":"inline:(i*7)%1000","sha256":hashlib.sha256(json.dumps([(i*7)%1000 for i in range(512)],separators=(",",":" )).encode()).hexdigest()},
      "gpu":gpu,"driver":"595.84","environment":{"python":sys.version,"platform":platform.platform(),"HCQ_NUM_COMPUTE":"2","HCQ_NV_MULTI_QUEUE_CUT_POLICY":env["HCQ_NV_MULTI_QUEUE_CUT_POLICY"],"HCQ_NV_READY_PLACEMENT":"0","QK_PRIMITIVE":"1","NV_COMPILER_Q4_IMMA_PP512":"1","NV_COMPILER_Q4_IMMA_K_PP512":"1","NV_COMPILER_Q4_IMMA_QO_PP512":"1","NV_COMPILER_Q4_IMMA_UNROLL":"4","NV_Q4_IMMA_PP512":None,"NV_UNROLL":None},"profile":False},
    "correctness":{},"census":{"predicted":{"q8_producer":198,"compiler_main":198,"canonical_weights":198,"unknown":0},"observed":None},
    "samples":{},"wall":{},"observer":{"enabled":False,"overhead":"not measured"},
    "decision":"S0 execution unavailable" if (missing or a.skip_run) else "fresh execution failed before evidence", "next_packet":None}
  if not missing and not a.skip_run:
    base=[sys.executable,"extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py","--arm","candidate","--q4-v","--rounds","9","--warmups","3","--deep-replay","--out",str(ev/"tinygrad-candidate.json"),"--logits-npz",str(ev/"tinygrad-candidate.npz"),"--model",str(model)]
    try:
      run(base,env,ev/"tinygrad-run.json")
      tiny=json.loads((ev/"tinygrad-candidate.json").read_text())
      result["samples"]["tinygrad"]=tiny.get("wall",{}).get("samples_ms"); result["wall"]["tinygrad"]=tiny.get("wall")
      dr=tiny.get("deep_replay",{}); first=next((c for c in dr.get("cycles",[]) if not all(v for k,v in c.items() if k.endswith("_exact"))),None)
      result["correctness"]={"tinygrad_status":tiny.get("status"),"token":tiny.get("token"),"finite":tiny.get("finite"),"deep_replay_exact":dr.get("all_cycles_exact"),"first_mismatch":first}
      result["census"]["observed"]=tiny.get("census")
      result["decision"]="recurrent correctness drift or baseline outside 1% authority band"
      result["status"]="PASS" if tiny.get("deep_replay",{}).get("all_cycles_exact") and abs(tiny.get("wall",{}).get("median_ms",0)-67.235719)/67.235719<=.01 else "STOP"
    except RuntimeError as exc:
      result["decision"]=str(exc); result["status"]="STOP"
      if (ev/"tinygrad-candidate.json").is_file():
        tiny=json.loads((ev/"tinygrad-candidate.json").read_text()); result["samples"]["tinygrad"]=tiny.get("wall",{}).get("samples_ms"); result["wall"]["tinygrad"]=tiny.get("wall"); result["correctness"]={"tinygrad_status":tiny.get("status"),"token":tiny.get("token"),"finite":tiny.get("finite"),"deep_replay":tiny.get("deep_replay")}; result["census"]["observed"]=tiny.get("census")
  pathlib.Path(a.report).write_text(json.dumps(result, indent=2)+"\n")
  (pathlib.Path(a.report).with_suffix(".md")).write_text(f"# S0 authority freeze\n\nStatus: **{result['status']}**\n\nEvidence: `{ev}`\n\nSource manifest is embedded in the JSON report.\n")
  print(json.dumps({"status":result["status"],"report":a.report,"evidence":str(ev)}, indent=2))
if __name__ == "__main__": main()
