"""Native-codegen microprimitive search (Step 4) — does tinygrad-NATIVE codegen emit the owned tile's proven
machine-code primitives? Candidate = a bounded tinygrad expression that SHOULD map to a target primitive. For each:
compile (capture the code object) -> local numerical correctness vs numpy -> ISA audit -> record. Authority = ISA
evidence + local correctness, NEVER W==D (this lane cannot promote a decode default).

  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_native_codegen_microsearch.py
See docs/native-codegen-microprimitive-search-{scope,execution-scope}-20260623.md."""
from __future__ import annotations
import os, json, pathlib, numpy as np
from tinygrad import Tensor, Device, dtypes
from extra.qk_amdgpu_isa_primitive_audit import audit

OUT = pathlib.Path("bench/qk-native-codegen-microsearch"); OUT.mkdir(parents=True, exist_ok=True)
DEV = Device[Device.DEFAULT]
_CAP = []
def _install_capture():
  comp = DEV.compiler; orig = comp.compile
  def wrap(src):
    lib = orig(src); _CAP.append(lib); return lib
  comp.compile = wrap

# Candidate grid: each targets one primitive. fn builds a tinygrad Tensor; ref is the numpy expectation.
def _rng(*s): return (np.random.default_rng(0).standard_normal(s) * 0.3).astype(np.float16)
def cand_cross_lane(n):   # reduce over a warp-sized axis -> cross-lane IF the renderer supported it
  a, b = _rng(256, n), _rng(256, n)
  return (Tensor(a) * Tensor(b)).sum(axis=1), (a.astype(np.float32) * b.astype(np.float32)).sum(1)
def cand_fp16_dot():      # fp16 multiply-accumulate -> v_dot2 IF lowered as a fused dot
  a, b = _rng(512, 128), _rng(512, 128)
  return (Tensor(a) * Tensor(b)).sum(axis=1), (a.astype(np.float32) * b.astype(np.float32)).sum(1)
def cand_lds_reduce():    # large workgroup reduction -> LDS staging (expected native)
  a = _rng(64, 4096)
  return Tensor(a).sum(axis=1), a.astype(np.float32).sum(1)
def cand_vector_load():   # contiguous fp16 elementwise -> vector global loads
  a = _rng(8192)
  return (Tensor(a) * 2.0).contiguous(), a.astype(np.float32) * 2.0

GRID = [
  {"id": "cross_lane_n32", "target": "has_cross_lane", "fn": lambda: cand_cross_lane(32)},
  {"id": "cross_lane_n64", "target": "has_cross_lane", "fn": lambda: cand_cross_lane(64)},
  {"id": "fp16_dot", "target": "has_v_dot2", "fn": cand_fp16_dot},
  {"id": "lds_reduce", "target": "has_lds", "fn": cand_lds_reduce},
  {"id": "vector_load", "target": "has_vector_global_load", "fn": cand_vector_load},
]

def run_candidate(c):
  _CAP.clear()
  t, ref = c["fn"]()
  out = t.numpy()                                  # compile + run (captures code objects)
  rel = float(np.sqrt(np.mean((out.astype(np.float32) - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-9))
  # audit every captured code object; OR the target flag across them (any kernel that emits it counts)
  flags = {"has_v_dot2": False, "has_lds": False, "has_cross_lane": False, "has_vector_global_load": False, "has_spill": False}
  counts = {}; n_obj = 0; max_vgpr = 0; max_scratch = 0
  for i, lib in enumerate(_CAP):
    p = OUT / f"_{c['id']}_{i}.co"; p.write_bytes(lib); n_obj += 1
    a = audit(str(p))
    for f, val in (a.get("flags") or {}).items():
      if f in flags: flags[f] = flags[f] or bool(val)
    for k, v in (a.get("instr_counts") or {}).items():
      if v: counts[k] = counts.get(k, 0) + v
    for kn in a.get("kernels", []):
      max_vgpr = max(max_vgpr, kn.get("vgpr_count") or 0)
      max_scratch = max(max_scratch, kn.get("private_segment_scratch_bytes") or 0)
    p.unlink(missing_ok=True)
  return {"id": c["id"], "target": c["target"], "rel_rmse": round(rel, 6), "correct": rel <= 1e-2,
          "n_code_objects": n_obj, "isa_flags": flags, "target_present": flags.get(c["target"], False),
          "max_vgpr": max_vgpr, "scratch_bytes": max_scratch, "no_spill": (max_scratch == 0 and not flags["has_spill"]),
          "ins_counts": {k: counts[k] for k in counts if counts[k]}}

def main():
  _install_capture()
  results = [run_candidate(c) for c in GRID]
  for r in results:
    print(f"  {r['id']:16} target={r['target']:24} present={str(r['target_present']):5} correct={r['correct']} "
          f"flags={[k for k,v in r['isa_flags'].items() if v]}")
  found = [r["id"] for r in results if r["target_present"] and r["correct"]]
  gaps = [r["target"] for r in results if not r["target_present"]]
  verdict = "NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND" if found else "NATIVE_CODEGEN_MICROSEARCH_NO_TARGET_FOUND"
  summary = {"date": "2026-06-23", "phase": "NATIVE_CODEGEN_MICROSEARCH", "results": results,
             "targets_found_native": found, "targets_still_gap": sorted(set(gaps)),
             "authority": "ISA evidence + local correctness (rel_rmse<=1e-2); NO W==D / decode promotion",
             "verdict": verdict}
  json.dump(summary, open(OUT / "result.json", "w"), indent=2)
  print("MICROSEARCH " + json.dumps({"verdict": verdict, "found": found, "gaps": sorted(set(gaps))}))

if __name__ == "__main__":
  main()
