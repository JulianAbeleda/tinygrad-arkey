#!/usr/bin/env python3
"""Validation-only integrated Q4_K/int8 WMMA loop-emitter gate."""
from __future__ import annotations
import argparse, contextlib, io, json, os, platform, re, subprocess, sys, time
from pathlib import Path
from extra.qk import model_profiles
ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_QUANT = "Q4_K"

def workload_authority():
  profile = model_profiles.qwen3_14b_q4k_m_gfx1100_profile()
  if (profile.family, profile.size_label, profile.quant, profile.device_profile) != ("qwen3", "14B", "Q4_K_M", "gfx1100"):
    raise ValueError(f"refusing non-Qwen3-14B Q4_K_M/gfx1100 profile: {profile.id!r}")
  try: role = profile.role_shape("ffn_down", phase="prefill")
  except KeyError as exc: raise ValueError(f"missing prefill ffn_down authority in profile {profile.id!r}") from exc
  if role.role != "ffn_down" or role.phase != "prefill" or role.quant != profile.quant:
    raise ValueError(f"invalid ffn_down authority in profile {profile.id!r}: {role!r}")
  if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in role.mnk):
    raise ValueError(f"invalid ffn_down shape in profile {profile.id!r}: {role.mnk!r}")
  evidence = {"profile_id": profile.id, "family": profile.family, "size_label": profile.size_label,
              "device_profile": profile.device_profile, "family_quant": profile.quant,
              "candidate_quant": CANDIDATE_QUANT, "role": role.role, "phase": role.phase,
              "shape": {"M": role.M, "N": role.N, "K": role.K}, "shape_source": "model_profile",
              "profile_shape_is_loaded_tensor_quant_authority": False,
              "loaded_tensor_quant_authority": {"status": "NOT_LOADED", "quant": None,
                "required_source": "loaded_GGUF_tensor_type", "runtime_binding": "fail_closed"}, "passed": True}
  return (("smoke_32x32x512", 32, 32, 512), (role.role, *role.mnk)), evidence

CASES, _ = workload_authority()
OUT = ROOT / "bench/qwen3-14b-integrated-loop-compile-gate/latest.json"

def worker(role, m, n, k, seed):
  import numpy as np
  from tinygrad import Context, GlobalCounters, Tensor, dtypes
  from extra.qk.layout import q8_1_quantize
  from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
  from extra.qk.prefill_int8_wmma_spec import Q4KInt8WMMATiledPrefillSpec, emit_q4k_int8_wmma_tiled_scheduler_tensor
  words, ref = _make_q4k_words(n, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).realize()
  xq, scales = q8_1_quantize(x.cast(dtypes.float32))
  xdq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) * scales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  oracle = (xdq @ ref.T).numpy()
  spec = Q4KInt8WMMATiledPrefillSpec(n=n, k=k, m=m, role=role, m_tile=16, n_tile=16, group_tile=8,
                                     wmma_m=16, wmma_n=16, wmma_k=16, implementation="integrated_loop")
  dbg = io.StringIO(); before = GlobalCounters.kernel_count; start = time.perf_counter()
  try:
    with contextlib.redirect_stdout(dbg), Context(DEBUG=4):
      graph = emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, scales, spec)
      graph_ms = (time.perf_counter() - start) * 1000; cstart = time.perf_counter()
      got = graph.realize().numpy(); compile_ms = (time.perf_counter() - cstart) * 1000
    trace = dbg.getvalue(); err = float(_rel_rmse(got, oracle))
    instruction_evidence = {"sudot4": bool(re.search(r"sudot4|dot4", trace, re.I)), "wmma": "wmma" in trace.lower()}
    dynamic_owner = {"compile": {"passed": True}, "correctness": {"passed": err < RTOL},
                     "instruction": {"passed": any(instruction_evidence.values()), **instruction_evidence}}
    return {"status": "PASS" if err < RTOL else "FAIL", "graph_build_ms": graph_ms, "compile_ms": compile_ms,
      "kernel_count": GlobalCounters.kernel_count-before, "correctness": {"status": "PASS" if err < RTOL else "FAIL", "rel_rmse": err, "rtol": RTOL},
      "instruction_evidence": instruction_evidence, "dynamic_owner_compile": dynamic_owner["compile"],
      "dynamic_owner_correctness": dynamic_owner["correctness"], "dynamic_owner_instruction": dynamic_owner["instruction"],
      "fallback": {"used": False, "policy": "fail_closed"}, "error": None}
  except Exception as exc:
    return {"status": "COMPILE_FAILURE", "graph_build_ms": (time.perf_counter()-start)*1000, "compile_ms": None,
      "kernel_count": GlobalCounters.kernel_count-before, "correctness": {"status": "NOT_CAPTURED"},
      "instruction_evidence": {"sudot4": False, "wmma": False}, "fallback": {"used": False, "policy": "fail_closed"},
      "error": f"{type(exc).__name__}: {exc}"}

def run(timeout):
  cases, authority = workload_authority()
  rows=[]
  for i, (role,m,n,k) in enumerate(cases):
    cmd=[sys.executable, __file__, "--worker", role, str(m), str(n), str(k), str(20260715+i)]
    t=time.perf_counter()
    try:
      p=subprocess.run(cmd, cwd=ROOT, env={**os.environ,"PYTHONPATH":str(ROOT)}, text=True, capture_output=True, timeout=timeout)
      lines=p.stdout.strip().splitlines(); row=json.loads(lines[-1]) if lines else {"status":"COMPILE_FAILURE","error":(p.stderr or "no worker output")[-2000:]}
    except subprocess.TimeoutExpired:
      row={"status":"COMPILE_FAILURE","error":f"timeout after {timeout}s","graph_build_ms":None,"compile_ms":None,"kernel_count":None,"correctness":{"status":"NOT_CAPTURED"},"instruction_evidence":{"sudot4":False,"wmma":False},"fallback":{"used":False,"policy":"fail_closed"}}
    row.update(role=role, candidate_quant=CANDIDATE_QUANT, shape={"M":m,"N":n,"K":k}, wall_ms=(time.perf_counter()-t)*1000); rows.append(row)
    if row["status"] == "COMPILE_FAILURE": break
  shape = authority["shape"]
  return {"schema":"qwen3_14b_integrated_loop_compile_gate.v2","model":"Qwen3-14B-Q4_K_M","hardware":platform.platform(),"emitter":"emit_q4k_int8_wmma_tiled_scheduler_tensor",
    "workload_authority": authority,
    "profile":f"32x32x512 smoke, then ffn_down {shape['M']}x{shape['N']}x{shape['K']}",
    "stop_rule":"smoke_32x32x512 then profile-authoritative ffn_down; stop at first concrete compile failure",
    "fallback_policy":"fail_closed; no route substitution","rows":rows}

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--worker",nargs=5); ap.add_argument("--timeout",type=int,default=300); ap.add_argument("--output",type=Path,default=OUT); a=ap.parse_args()
  if a.worker: print(json.dumps(worker(a.worker[0],*map(int,a.worker[1:])))); raise SystemExit
  report=run(a.timeout); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
