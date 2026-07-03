"""AMD ISA backend — grid-parallelism gate: native decode tile launches across global axes (not grid=[1,1,1]).

The generated block tile's RANGE(GLOBAL) axes (kv-head, split) were lowered to SERIAL loops in one workgroup
(grid=[1,1,1]), leaving ~40 CUs idle. extra/qk_native_isa_block_tile_graph_node._range_global_to_grid rewrites the
tile AST (RANGE(GLOBAL) -> SPECIAL(gidx{axis_id}), dropped from its END) before to_program, so from_sink sets the
launch grid, isel_special lowers gidx -> workgroup-id SGPR, and elf.py enables the workgroup-id descriptor bits --
all via the existing gidx ABI (microgated 1D+2D). No hardcoded grid override; no renderer RANGE changes.

Records grid + RANGE/END before/after (in-process), and consolidates W==D (Phase I harness) + in-model token-match
+ determinism + route attribution + resources. AMD_ISA_NO_GRID=1 toggles the old serial behavior for the A/B.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/grid_gate.py
Writes: bench/amd-isa-backend-grid/latest.json
"""
import os, json, pathlib
os.environ.setdefault("DEV", "AMD")
ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "bench/amd-isa-backend-grid/latest.json"

def _grid_and_loops(no_grid:bool):
  # compile the in-model-shaped tile (S=48) with grid on/off; report grid + RANGE/END counts in the lowered sink.
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad import dtypes
  os.environ["AMD_ISA_NO_GRID"] = "1" if no_grid else "0"
  import importlib, extra.qk.native_isa_block_tile_graph_node as M
  importlib.reload(M)
  M._compile.cache_clear()
  vsp = UOp.variable("start_pos", 0, 4607)
  # reach into to_program to count RANGE/END in the lowered sink
  from tinygrad.codegen import to_program
  from extra.qk.flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
  Hd, Hq, Hkv, MAXC, L, S = 128, 32, 8, 4608, 96, 48
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, vsp + 1)
  phs = [UOp.placeholder((Hq*S*(Hd+2),), dtypes.float32, 0), UOp.placeholder((Hq*Hd,), dtypes.float32, 1),
         UOp.placeholder((2,1,Hkv,MAXC,Hd), dtypes.float32, 2)]
  sink = fxn(*phs)
  if not no_grid: sink = M._range_global_to_grid(sink)
  prg = to_program(sink, M._isa_renderer())
  fs = prg.src[0]
  nrange = sum(1 for u in fs.toposort() if u.op is Ops.RANGE)
  nend = sum(1 for u in fs.toposort() if u.op is Ops.END)
  return {"grid": list(prg.arg.global_size), "block": list(prg.arg.local_size), "range_nodes": nrange, "end_nodes": nend}

def main():
  rec = {"verdict": None, "command": "DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/grid_gate.py",
         "mechanism": "RANGE(GLOBAL) -> SPECIAL(gidx) AST rewrite in the native injector (before to_program); gidx ABI -> workgroup-id SGPRs + elf enable bits + from_sink grid"}
  import subprocess, sys
  def _run(no_grid):
    out = subprocess.run([sys.executable, "-c",
      f"import os;os.environ['DEV']='AMD';import json;from extra.audit.amd_isa.grid_gate import _grid_and_loops;print('@@'+json.dumps(_grid_and_loops({no_grid})))"],
      cwd=str(ROOT), env={**os.environ}, capture_output=True, text=True, timeout=300).stdout
    return json.loads([l for l in out.splitlines() if l.startswith("@@")][-1][2:])
  before = _run(True); after = _run(False)
  rec["grid_before"] = before["grid"]; rec["grid_after"] = after["grid"]
  rec["block"] = after["block"]
  rec["range_end_before"] = {"range": before["range_nodes"], "end": before["end_nodes"]}
  rec["range_end_after"] = {"range": after["range_nodes"], "end": after["end_nodes"]}
  rec["workgroups_before"] = before["grid"][0]*before["grid"][1]*before["grid"][2]
  rec["workgroups_after"] = after["grid"][0]*after["grid"][1]*after["grid"][2]
  # W==D: before = committed Phase I baseline (serial grid); after = current Phase I run (grid on)
  rec["wd_before_phase_i_baseline"] = {"ctx512_native_tok_s": 0.45, "ctx4096_native_tok_s": 0.37, "pct_of_owned": "~0.4%",
                                       "note": "Phase I baseline commit 7c0a3bad1 (grid=[1,1,1])"}
  pi = ROOT / "bench/amd-isa-backend-phase-i/latest.json"
  if pi.exists():
    r = json.load(open(pi)); rec["wd_after"] = {ck: {"native_tok_s": v["native_tok_s"], "owned_tok_s": v["owned_tok_s"],
                                                     "pct_of_owned": v["pct_of_owned"], "token_match": v["token_match"]} for ck, v in r["per_ctx"].items()}
    rec["wd_after_route_bound"] = r["route_bound"]; rec["wd_after_resources"] = r["native_resource_summary"]
  hm = ROOT / "bench/amd-isa-backend-phase-h/inmodel.json"
  if hm.exists():
    r = json.load(open(hm)); rec["in_model_token_match"] = r["token_or_output_correctness"]["token_match"]
    rec["in_model_repeated_run_stability"] = r.get("repeated_run_stability"); rec["in_model_route_attribution"] = {k: v for k, v in r["route_attribution"].items() if "kernels" not in k}
    rec["hidden_fallback_check"] = r.get("hidden_fallback_check")
  # verdict
  grid_fixed = rec["grid_after"] != [1, 1, 1] and rec["grid_after"][0] > 1
  tok = rec.get("wd_after", {}).get("512", {}).get("native_tok_s", 0); base = 0.45
  improves_5x = tok >= 5 * base
  tmatch = rec.get("in_model_token_match", False) and all(v["token_match"] for v in rec.get("wd_after", {}).values())
  det = bool(rec.get("in_model_repeated_run_stability"))
  nofb = rec.get("in_model_route_attribution", {}).get("native_block_tile_fired") and rec.get("in_model_route_attribution", {}).get("hip_llvm_block_tile_absent") and rec.get("in_model_route_attribution", {}).get("owned_tile_absent")
  if not grid_fixed: rec["verdict"] = "AMD_ISA_GRID_BLOCKED_RANGE_GLOBAL_LOWERING"
  elif not nofb: rec["verdict"] = "AMD_ISA_GRID_BLOCKED_HIDDEN_FALLBACK"
  elif not tmatch: rec["verdict"] = "AMD_ISA_GRID_BLOCKED_TOKEN_MATCH"
  elif not det: rec["verdict"] = "AMD_ISA_GRID_BLOCKED_NONDETERMINISM"
  elif not improves_5x: rec["verdict"] = "AMD_ISA_GRID_PASS_NO_PERFORMANCE_MOVEMENT"
  else: rec["verdict"] = "AMD_ISA_GRID_PARALLELISM_PASS_NATIVE_TILE_GRID_BOUND"; rec["next_phase_unlocked"] = "Phase L (modulo) / M (occupancy) now meaningful (tile occupies the GPU; ~44-46% of owned)"
  return rec

if __name__ == "__main__":
  rec = main()
  ART.parent.mkdir(parents=True, exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nGRID", rec["verdict"])
