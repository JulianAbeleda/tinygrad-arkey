#!/usr/bin/env python3
"""Closed-lease reverse wall bracket for active-horizon wide Flash."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL, LOCK, PYTHON


def _install(model, candidate:bool, splits:int, token_bound:int) -> None:
  geometry = {"split_count":splits, "llama_vec_wide":True, "token_bound":token_bound} if candidate else None
  model._flash_decode_tile_geometry_lease = geometry
  for block in model.blk: block._flash_decode_tile_geometry_lease = geometry


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int, splits:int,
              token_bound:int, out:pathlib.Path, profile_jsonl:pathlib.Path|None=None) -> dict:
  os.environ.update(DEV="NV", PROFILE="1" if profile_jsonl else "0")
  if profile_jsonl:
    profile_jsonl.parent.mkdir(parents=True, exist_ok=True); profile_jsonl.unlink(missing_ok=True)
    os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(profile_jsonl)
    from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import _install_graph_tracker
    _install_graph_tracker()
  else: os.environ.pop("HCQ_GRAPH_PROFILE_JSON", None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

  dev = Device["NV"]; model = _load(MODEL, max_context)
  _install(model, arm == "candidate", splits, token_bound)
  model._decode_direct_greedy_promoted = True; model._decode_feedback_pingpong_promoted = True
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, dev, count, reps)
  finally: gen.close()
  if profile_jsonl:
    from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import _flush_final_timestamps
    dev.synchronize(); _flush_final_timestamps(); dev.synchronize()
  result = {"schema":"tinygrad.nv_flash_bounded_wall.v1", "arm":arm, "depth":depth, "count":count,
    "reps":reps, "max_context":max_context, "candidate_splits":splits, "candidate_token_bound":token_bound,
    "settled":settled, "profile_jsonl":str(profile_jsonl) if profile_jsonl else None}
  out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def _child(arm:str, root:pathlib.Path, args) -> dict:
  out = root/f"{arm}.json"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--arm", arm, "--depth", str(args.depth), "--count", str(args.count), "--max-context", str(args.max_context),
    "--reps", str(args.reps), "--splits", str(args.splits), "--token-bound", str(args.token_bound), "--out", str(out)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH":str(ROOT), "PROFILE":"0"})
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-6000:]}")
  return json.loads(out.read_text())


def driver(args) -> dict:
  root = pathlib.Path(str(args.out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  arms = [_child("control_a", root, args), _child("candidate", root, args), _child("control_c", root, args)]
  def wall(row): return float(row["settled"]["median_ms_per_token"])*1000.0
  control = statistics.median((wall(arms[0]), wall(arms[2]))); candidate = wall(arms[1])
  hashes = {x["settled"]["token_stream_hash"] for x in arms}
  result = {"schema":"tinygrad.nv_flash_bounded_wall.v1", "mode":"reverse-bracket-unprofiled",
    "depth":args.depth, "count":args.count, "reps":args.reps, "max_context":args.max_context,
    "candidate_splits":args.splits, "candidate_token_bound":args.token_bound,
    "max_observed_tc":args.depth+7+args.count*args.reps,
    "all_token_hashes_equal":len(hashes)==1, "token_stream_hashes":sorted(hashes),
    "walls_us_per_token":{"control_a":wall(arms[0]), "candidate":candidate, "control_c":wall(arms[2]),
      "control_midpoint":control, "candidate_delta":candidate-control},
    "tokens_per_second":{"control_midpoint":1e6/control, "candidate":1e6/candidate,
      "candidate_delta":1e6/candidate-1e6/control}, "arms":arms}
  result["verdict"] = "WALL_PASS" if result["all_token_hashes_equal"] and candidate < min(wall(arms[0]), wall(arms[2])) else "NO_GO_WALL"
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); return result


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--arm", choices=("control_a","candidate","control_c"))
  ap.add_argument("--depth", type=int, default=512); ap.add_argument("--count", type=int, default=8)
  ap.add_argument("--max-context", type=int, default=1024); ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--splits", type=int, required=True); ap.add_argument("--token-bound", type=int, required=True)
  ap.add_argument("--profile-jsonl", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path, required=True); args = ap.parse_args()
  if args.token_bound != args.splits*128 or args.token_bound > args.max_context:
    raise ValueError("token-bound must equal splits*128 and fit max-context")
  if args.depth+7+args.count*args.reps > args.token_bound:
    raise ValueError("the complete bracket must remain within candidate token-bound")
  result = run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.splits,args.token_bound,args.out,args.profile_jsonl) \
    if args.arm else driver(args)
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
