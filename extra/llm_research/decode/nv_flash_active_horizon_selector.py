#!/usr/bin/env python3
"""Closed-lease production-selector bracket for S6 through Tc=768, S8 after."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL, LOCK, PYTHON

S6_NAME = "flash_vec_llama_score_pv_32_128_6_widekv16"
S8_NAME = "flash_vec_llama_score_pv_32_128_8_widekv16"


def _profile_names(path:pathlib.Path) -> dict:
  counts:dict[str,int] = {}
  if path.exists():
    for line in path.read_text().splitlines():
      for entry in json.loads(line).get("entries", []):
        name = str(entry.get("name", "")); counts[name] = counts.get(name, 0) + 1
  return {"s6_score_entries":counts.get(S6_NAME, 0), "s8_score_entries":counts.get(S8_NAME, 0),
          "distinct_program_names":len(counts)}


def _prewarm_greedy_pairs(model, candidate:bool, max_context:int) -> None:
  """Move both graph captures to load time; the following prompt overwrites the dummy KV."""
  from tinygrad import Tensor, UOp
  v_sp = UOp.variable("start_pos", 0, max_context-1)
  dummy, temp = Tensor([[0]], dtype="int32").contiguous(), Tensor([0.0]).contiguous()
  variants = ((6, 700), (None, 800)) if candidate else ((None, 800),)
  for split_count, ctx in variants:
    for slot in (0, 1):
      for _ in range(3):
        model(dummy, v_sp.bind(ctx), temp, use_flash=True, greedy=True, feedback_slot=slot,
              flash_split_count=split_count).realize()
  model.reset_generation_state()
  if candidate: model._flash_decode_active_horizon_prewarmed = True


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path,
              profile_jsonl:pathlib.Path|None=None, prewarm:bool=False) -> dict:
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
  candidate = arm.startswith("candidate")
  model._flash_decode_active_horizon_lease = candidate
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = True
  # Submit-ahead is intentionally held equal and closed in this selector
  # qualification. It is a separate launch-overlap policy and its current
  # eligibility census names the installed S8 pair only.
  model._decode_submit_ahead_promoted = False
  if prewarm: _prewarm_greedy_pairs(model, candidate, max_context)
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, dev, count, reps)
  finally: gen.close()
  if profile_jsonl:
    from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import _flush_final_timestamps
    dev.synchronize(); _flush_final_timestamps(); dev.synchronize()
  result = {"schema":"tinygrad.nv_flash_active_horizon_selector.v1", "arm":arm, "depth":depth,
    "count":count, "reps":reps, "max_context":max_context, "selector_enabled":candidate,
    "prewarmed_graph_pairs":prewarm,
    "expected_policy":"S6 for 512<=start_pos<768; installed S8 otherwise", "settled":settled,
    "captured_pairs":{"s6":[x.captured is not None for x in model.rollout_greedy_pingpong_jits_flash_s6],
                      "s8":[x.captured is not None for x in model.rollout_greedy_pingpong_jits_flash]},
    "profile_jsonl":str(profile_jsonl) if profile_jsonl else None,
    "profile_census":_profile_names(profile_jsonl) if profile_jsonl else None}
  out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def _child(arm:str, root:pathlib.Path, args, profile:bool=False) -> dict:
  label = f"{arm}-profile" if profile else arm
  out, profile_jsonl = root/f"{label}.json", root/f"{label}.profile.jsonl"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--arm", arm, "--depth", str(args.depth), "--count", str(args.count), "--max-context", str(args.max_context),
    "--reps", str(args.reps), "--out", str(out)]
  if profile: cmd += ["--profile-jsonl", str(profile_jsonl)]
  if args.prewarm: cmd += ["--prewarm"]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH":str(ROOT), "PROFILE":"0"})
  if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-6000:]}")
  return json.loads(out.read_text())


def driver(args) -> dict:
  root = pathlib.Path(str(args.out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  names = ("candidate_a", "control", "candidate_c") if args.candidate_controls else \
    ("control_a", "candidate", "control_c")
  if args.reuse_existing:
    arms = [json.loads((root/f"{name}.json").read_text()) for name in names]
    profile = None if args.skip_profile else json.loads((root/"candidate-profile.json").read_text())
  else:
    arms = [_child(name, root, args) for name in names]
    profile = None if args.skip_profile else _child("candidate", root, args, profile=True)
  def wall(row): return float(row["settled"]["median_ms_per_token"])*1000.0
  control, candidate = ((wall(arms[1]), statistics.median((wall(arms[0]), wall(arms[2]))))
                        if args.candidate_controls else
                        (statistics.median((wall(arms[0]), wall(arms[2]))), wall(arms[1])))
  hashes = {x["settled"]["token_stream_hash"] for x in arms}
  census = profile["profile_census"] if profile else None
  candidate_arms = [arm for arm in arms if arm["selector_enabled"]]
  cold_spike = any(max(arm["settled"]["samples_ms_per_token"]) > 2*candidate/1000.0 for arm in candidate_arms)
  result = {"schema":"tinygrad.nv_flash_active_horizon_selector.v1",
    "mode":"reverse-bracket-plus-profile" if profile else "reverse-bracket-capture-census",
    "depth":args.depth, "count":args.count, "reps":args.reps, "max_context":args.max_context,
    "first_timed_tc":args.depth+8, "max_observed_tc":args.depth+7+args.count*args.reps,
    "crosses_transition":args.depth+8 <= 768 < args.depth+7+args.count*args.reps,
    "prewarmed_graph_pairs":args.prewarm, "all_token_hashes_equal":len(hashes)==1,
    "token_stream_hashes":sorted(hashes), "profile_census":census, "candidate_cold_transition_spike":cold_spike,
    "graph_identity_pass":(census["s6_score_entries"] > 0 and census["s8_score_entries"] > 0) if census else
      all(arm["captured_pairs"][key] == [True, True] for arm in candidate_arms for key in ("s6", "s8")),
    "arm_order":list(names),
    "walls_us_per_token":{"arm_a":wall(arms[0]), "arm_b":wall(arms[1]), "arm_c":wall(arms[2]),
      "control":control, "candidate":candidate, "candidate_delta":candidate-control},
    "tokens_per_second":{"control_midpoint":1e6/control, "candidate":1e6/candidate,
      "candidate_delta":1e6/candidate-1e6/control}, "arms":arms, "profile_arm":profile}
  strict_pass = control > max(wall(arms[0]), wall(arms[2])) if args.candidate_controls else \
    candidate < min(wall(arms[0]), wall(arms[2]))
  result["verdict"] = "SELECTOR_PASS" if result["crosses_transition"] and result["all_token_hashes_equal"] and \
    result["graph_identity_pass"] and not cold_spike and strict_pass else "NO_GO_SELECTOR"
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); return result


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--arm", choices=("control_a", "candidate", "control_c", "candidate_a", "control", "candidate_c"))
  ap.add_argument("--depth", type=int, default=704); ap.add_argument("--count", type=int, default=8)
  ap.add_argument("--max-context", type=int, default=1024); ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--profile-jsonl", type=pathlib.Path); ap.add_argument("--prewarm", action="store_true")
  ap.add_argument("--candidate-controls", action="store_true")
  ap.add_argument("--skip-profile", action="store_true"); ap.add_argument("--reuse-existing", action="store_true")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  if args.max_context < args.depth+7+args.count*args.reps: raise ValueError("run extent exceeds max-context")
  result = run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.out,args.profile_jsonl,args.prewarm) \
    if args.arm else driver(args)
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
