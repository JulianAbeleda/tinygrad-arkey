#!/usr/bin/env python3
"""Unprofiled reverse wall bracket for the closed S6/wide-KV flash lease."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL, LOCK, PYTHON, _install


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  os.environ.update(DEV="NV", PROFILE="0")
  os.environ.pop("HCQ_GRAPH_PROFILE_JSON", None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

  dev = Device["NV"]
  model = _load(MODEL, max_context)
  _install(model, arm == "candidate", max_context)
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = True
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, dev, count, reps)
  finally: gen.close()
  result = {"schema":"tinygrad.nv_flash_llama_vec_wide_wall.v1", "arm":arm, "depth":depth,
            "count":count, "reps":reps, "max_context":max_context, "settled":settled}
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def _child(arm:str, root:pathlib.Path, depth:int, count:int, max_context:int, reps:int) -> dict:
  out = root/f"{arm}.json"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON), str(pathlib.Path(__file__).resolve()),
         "--arm", arm, "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
         "--reps", str(reps), "--out", str(out)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH":str(ROOT), "PROFILE":"0"})
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-6000:]}")
  return json.loads(out.read_text())


def driver(depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  arms = [_child("control_a", root, depth, count, max_context, reps),
          _child("candidate", root, depth, count, max_context, reps),
          _child("control_c", root, depth, count, max_context, reps)]
  def wall(row): return float(row["settled"]["median_ms_per_token"])*1000.0
  control = statistics.median((wall(arms[0]), wall(arms[2])))
  candidate = wall(arms[1])
  hashes = {x["settled"]["token_stream_hash"] for x in arms}
  result = {"schema":"tinygrad.nv_flash_llama_vec_wide_wall.v1", "mode":"reverse-bracket-unprofiled",
    "depth":depth, "count":count, "reps":reps, "max_context":max_context,
    "all_token_hashes_equal":len(hashes)==1, "token_stream_hashes":sorted(hashes),
    "walls_us_per_token":{"control_a":wall(arms[0]), "candidate":candidate,
      "control_c":wall(arms[2]), "control_midpoint":control, "candidate_delta":candidate-control},
    "tokens_per_second":{"control_midpoint":1e6/control, "candidate":1e6/candidate,
      "candidate_delta":1e6/candidate-1e6/control}, "arms":arms}
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm", choices=("control_a","candidate","control_c"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=24)
  ap.add_argument("--max-context",type=int,default=768); ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.depth + 7 + args.count * args.reps > args.max_context:
    raise ValueError("depth + 7 warmup/prelude tokens + count*reps must fit max-context")
  result = run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.out) if args.arm else \
           driver(args.depth,args.count,args.max_context,args.reps,args.out)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
