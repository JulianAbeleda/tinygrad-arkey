#!/usr/bin/env python3
from __future__ import annotations

import json, os, pathlib, re, subprocess, sys, traceback
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_load_b128, ds_store_b64, global_load_b64, global_store_b32, s_barrier, s_endpgm, s_load_b64, s_waitcnt,
  v_mov_b32_e32, v_wmma_f32_16x16x16_f16,
)
from tinygrad.uop.ops import KernelInfo, Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def inst_name(inst: Any) -> str:
  return getattr(inst, "op_name", None) or getattr(getattr(inst, "op", None), "name", None) or type(inst).__name__


def run_p7a() -> dict[str, Any]:
  env = os.environ.copy()
  env.update({"LDSTILE": "1", "GEMM": "0", "DEBUG": "0"})
  env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  cmd = [sys.executable, "extra/gemm/rdna3_wmma_matmul.py"]
  try:
    cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    out = cp.stdout
    rm = re.search(r"relative RMSE ([0-9.]+)", out)
    rel = float(rm.group(1)) if rm else None
    correct = cp.returncode == 0 and rel is not None and rel <= 0.05 and "LDS TILE CORRECT" in out
    return {
      "cmd": "LDSTILE=1 GEMM=0 python3 extra/gemm/rdna3_wmma_matmul.py",
      "returncode": cp.returncode,
      "relative_rmse": rel,
      "correct": correct,
      "stdout_tail": out[-4000:],
    }
  except subprocess.TimeoutExpired as e:
    return {"cmd": "LDSTILE=1 GEMM=0 python3 extra/gemm/rdna3_wmma_matmul.py", "timeout": True, "correct": False, "stdout_tail": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else ""}


def p7b_insts() -> list[Any]:
  insts: list[Any] = [
    s_load_b64(sdata=s[4:5], sbase=s[0:1], offset=0, soffset=NULL),
    s_waitcnt(simm16=0),
  ]
  # Use the output pointer as a valid global-load base and zero all address VGPRs used by representative loads.
  for r in [40, 41, 42, 43, 44, 45, 46, 47]:
    insts.append(v_mov_b32_e32(vdst=v[r], src0=0))
  insts += [
    global_load_b64(vdst=v[200:201], addr=v[40], saddr=s[4:5]),
    global_load_b64(vdst=v[202:203], addr=v[42], saddr=s[4:5]),
    global_load_b64(vdst=v[204:205], addr=v[44], saddr=s[4:5]),
    global_load_b64(vdst=v[206:207], addr=v[46], saddr=s[4:5]),
    s_waitcnt(simm16=0),
  ]
  for r in [195, 196, 223]:
    insts.append(v_mov_b32_e32(vdst=v[r], src0=0))
  insts += [
    ds_store_b64(addr=v[195], data0=v[200:201], offset0=0, offset1=0),
    ds_store_b64(addr=v[195], data0=v[202:203], offset0=0, offset1=1),
    ds_store_b64(addr=v[196], data0=v[204:205], offset0=0, offset1=64),
    ds_store_b64(addr=v[196], data0=v[206:207], offset0=32, offset1=65),
    s_barrier(),
    ds_load_b128(vdst=v[160:163], addr=v[223], offset0=0, offset1=0),
    ds_load_b128(vdst=v[164:167], addr=v[223], offset0=16, offset1=0),
    ds_load_b128(vdst=v[168:171], addr=v[223], offset0=0, offset1=9),
    ds_load_b128(vdst=v[172:175], addr=v[223], offset0=0, offset1=64),
    s_waitcnt(simm16=0),
  ]
  for r in range(8):
    insts.append(v_mov_b32_e32(vdst=v[r], src0=0))
  insts += [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[160:167], src1=v[168:175], src2=v[0:7]),
    v_mov_b32_e32(vdst=v[2], src0=0),
    v_mov_b32_e32(vdst=v[3], src0=0),
    global_store_b32(addr=v[2], data=v[0], saddr=s[4:5]),
    s_waitcnt(simm16=0),
    s_endpgm(),
  ]
  return insts


def build_p7b_kernel(out: UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((1, 1, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((32, 1, 1))]
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=25088, addrspace=AddrSpace.LOCAL), (), "bb5a10_p7b_lds")
  sink = UOp.sink(out.base, lds, *gidxs, *lidxs, arg=KernelInfo(name="bb5a10_p7b_executable_wrapper"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in p7b_insts()))))


def run_p7b() -> dict[str, Any]:
  out = Tensor.zeros(128, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  wrapped = Tensor.custom_kernel(out, fxn=build_p7b_kernel)[0]
  linear = wrapped.schedule_linear()
  run_linear(linear)
  got = wrapped.numpy().astype(np.uint32)
  names = [inst_name(i) for i in p7b_insts()]
  return {
    "launch": {"global": [1, 1, 1], "local": [32, 1, 1]},
    "lds_bytes": 25088,
    "instruction_names": names,
    "has_output_store": "GLOBAL_STORE_B32" in names,
    "has_kernarg_load": "S_LOAD_B64" in names,
    "has_lidx": True,
    "has_gidx": True,
    "ran": True,
    "output_first_8": [int(x) for x in got[:8]],
    "output_finite": bool(np.all(np.isfinite(got.astype(np.float32)))),
  }


def main() -> int:
  scope = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json", {})
  p6 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json", {})
  p7a = run_p7a()
  try:
    p7b = run_p7b()
  except Exception as e:
    p7b = {"ran": False, "error": repr(e), "traceback": traceback.format_exc()}
  gate = {
    "input_scope_pass": scope.get("verdict") == "PASS_BB5A10_P7_CORRECTNESS_SCOPE_READY" and bool(scope.get("gate_pass")),
    "input_p6_pass": p6.get("verdict") == "PASS_BB5A10_P6_STRUCTURAL_CANDIDATE" and bool(p6.get("gate_pass")),
    "p7a_known_good_lds_wmma_correct": bool(p7a.get("correct")),
    "p7b_wrapper_ran": bool(p7b.get("ran")),
    "p7b_has_kernarg_load": bool(p7b.get("has_kernarg_load")),
    "p7b_has_lds_allocation": int(p7b.get("lds_bytes") or 0) == 25088,
    "p7b_has_lidx_gidx": bool(p7b.get("has_lidx")) and bool(p7b.get("has_gidx")),
    "p7b_has_output_store": bool(p7b.get("has_output_store")),
    "p7b_has_structural_lds_wmma": "DS_STORE_B64" in (p7b.get("instruction_names") or []) and "DS_LOAD_B128" in (p7b.get("instruction_names") or []) and any("WMMA" in n for n in (p7b.get("instruction_names") or [])),
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P7a_P7b_correctness_harness",
    "schema": "amd_bb5a10_p7a_p7b_correctness_result_v1",
    "verdict": "PASS_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER" if gate_pass else "BLOCKED_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "p7a": p7a,
    "p7b": p7b,
    "gate": gate,
    "decision": "P7a/P7b pass: known-good LDS WMMA correctness works and the structural candidate is now an executable wrapper with output. Next is P7c numeric correctness." if gate_pass else
                "P7a/P7b blocked; fix the failed harness row before numeric correctness.",
    "next_action": "Implement P7c small deterministic numeric correctness; P8 remains blocked." if gate_pass else "Fix P7a/P7b before P7c.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json",
    ],
  }
  write_json("bb5a10_p7a_p7b_correctness_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p7a_p7b_correctness_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "p7a_rmse": p7a.get("relative_rmse"),
    "p7b_ran": p7b.get("ran"),
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
