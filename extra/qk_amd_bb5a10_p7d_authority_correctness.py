#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad import Context, Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.helpers import colored, getenv
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_load_b128, ds_store_b64, global_load_b128, global_store_b16, s_add_i32, s_barrier, s_cbranch_scc1,
  s_cmp_lt_i32, s_endpgm, s_load_b128, s_load_b64, s_mov_b32, s_sendmsg, s_waitcnt, v_add_nc_u32_e32,
  v_and_b32_e32, v_cvt_f16_f32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_mov_b32_e32,
  v_mul_lo_u32, v_wmma_f32_16x16x16_f16,
)
from tinygrad.uop.ops import KernelInfo, Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"
FA, FB, ACC = 20, 32, 44


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def waitcnt_lgkm(n: int):
  return s_waitcnt(simm16=(0x7) | ((n & 0x3F) << 4) | (0x3F << 10))


def waitcnt_vm(n: int):
  return s_waitcnt(simm16=(0x7) | ((0x3F) << 4) | ((n & 0x3F) << 10))


def build_authority_k_tile(k: int) -> list[Any]:
  assert k % 16 == 0
  lds_a, lds_b = 0, 16 * 16 * 2
  insts: list[Any] = []
  branches: list[tuple[int, str]] = []
  labels: dict[str, int] = {}

  def pc() -> int: return sum(i.size() for i in insts)
  def e(i): insts.append(i); return i
  def label(name: str) -> None: labels[name] = pc()
  def br(name: str) -> None: branches.append((len(insts) - 1, name))

  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  e(v_and_b32_e32(v[1], 15, v[0]))
  e(v_mul_lo_u32(v[2], v[1], k * 2))
  e(v_mov_b32_e32(v[3], 0))
  e(v_mov_b32_e32(v[18], v[2]))
  e(v_mov_b32_e32(v[19], 0))
  e(v_lshlrev_b32_e32(v[16], 5, v[1]))
  e(v_add_nc_u32_e32(v[17], lds_b, v[16]))
  for i in range(8): e(v_mov_b32_e32(v[ACC+i], 0))

  e(s_mov_b32(s[16], 0))
  label("KLOOP")
  e(global_load_b128(vdst=v[FA:FA+3],   addr=v[2:2],   saddr=s[4:5], offset=0))
  e(global_load_b128(vdst=v[FA+4:FA+7], addr=v[2:2],   saddr=s[4:5], offset=16))
  e(global_load_b128(vdst=v[FB:FB+3],   addr=v[18:18], saddr=s[6:7], offset=0))
  e(global_load_b128(vdst=v[FB+4:FB+7], addr=v[18:18], saddr=s[6:7], offset=16))
  e(waitcnt_vm(0))
  for base, addr in [(FA, 16), (FB, 17)]:
    e(ds_store_b64(addr=v[addr], data0=v[base:base+1], offset0=0, offset1=0))
    e(ds_store_b64(addr=v[addr], data0=v[base+2:base+3], offset0=8, offset1=0))
    e(ds_store_b64(addr=v[addr], data0=v[base+4:base+5], offset0=16, offset1=0))
    e(ds_store_b64(addr=v[addr], data0=v[base+6:base+7], offset0=24, offset1=0))
  e(waitcnt_lgkm(0))
  e(s_barrier())
  e(ds_load_b128(vdst=v[FA:FA+3],   addr=v[16], offset0=0,  offset1=0))
  e(ds_load_b128(vdst=v[FA+4:FA+7], addr=v[16], offset0=16, offset1=0))
  e(ds_load_b128(vdst=v[FB:FB+3],   addr=v[17], offset0=0,  offset1=0))
  e(ds_load_b128(vdst=v[FB+4:FB+7], addr=v[17], offset0=16, offset1=0))
  e(waitcnt_lgkm(0))
  e(v_wmma_f32_16x16x16_f16(vdst=v[ACC:ACC+7], src0=v[FA:FA+7], src1=v[FB:FB+7], src2=v[ACC:ACC+7]))
  e(s_barrier())
  e(v_add_nc_u32_e32(v[2], 32, v[2]))
  e(v_add_nc_u32_e32(v[18], 32, v[18]))
  e(s_add_i32(s[16], s[16], 1))
  e(s_cmp_lt_i32(s[16], k // 16))
  e(s_cbranch_scc1(simm16=0)); br("KLOOP")

  e(v_and_b32_e32(v[10], 15, v[0]))
  e(v_lshrrev_b32_e32(v[11], 4, v[0])); e(v_and_b32_e32(v[11], 1, v[11]))
  e(v_lshlrev_b32_e32(v[12], 4, v[11]))
  e(v_add_nc_u32_e32(v[12], v[12], v[10]))
  e(v_lshlrev_b32_e32(v[12], 1, v[12]))
  for i in range(8):
    e(v_cvt_f16_f32_e32(v[14], v[ACC+i]))
    e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
    if i < 7: e(v_add_nc_u32_e32(v[12], 64, v[12]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())

  for idx, target in branches:
    off = (labels[target] - sum(i.size() for i in insts[:idx+1])) // 4
    assert -32768 <= off <= 32767
    insts[idx].simm16 = off
  return insts


def run_authority_subset(k: int) -> dict[str, Any]:
  rng = np.random.default_rng(7)
  a_np = (rng.standard_normal((16, k)) * 0.1).astype(np.float16)
  b_np = (rng.standard_normal((k, 16)) * 0.1).astype(np.float16)
  bt_np = np.ascontiguousarray(b_np.T)
  a = Tensor(a_np, device="AMD")
  bt = Tensor(bt_np, device="AMD")
  c = Tensor.empty(16, 16, dtype=dtypes.half, device="AMD")
  Tensor.realize(a, bt, c)
  insts = build_authority_k_tile(k)

  def asm_kernel(A: UOp, B: UOp, C: UOp) -> UOp:
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=2048, addrspace=AddrSpace.LOCAL), (), "bb5a10_p7d_lds")
    sink = UOp.sink(A.base, B.base, C.base, lds, UOp.special(1, "gidx0"), UOp.special(32, "lidx0"),
                    arg=KernelInfo(name=colored("bb5a10_p7d_authority_k", "cyan"), estimates=Estimates(ops=16*16*k*2, mem=(16*k+16*k+16*16)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))))

  c = Tensor.custom_kernel(a, bt, c, fxn=asm_kernel)[2]
  with Context(DEBUG=0): run_linear(c.schedule_linear())
  got = c.float().numpy()
  ref = a_np.astype(np.float32) @ b_np.astype(np.float32)
  rel = float(np.sqrt(np.mean((got - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-9))
  names = [getattr(i, "op_name", type(i).__name__) for i in insts]
  return {
    "shape": [16, 16, k],
    "authority_contract": {"M": 512, "N": 12288, "K": 4096, "subset": "single_16x16_output_tile_full_K"},
    "relative_rmse": rel,
    "correct": rel <= 0.001,
    "tolerance": 0.001,
    "instruction_counts": {
      "ds_store_b64": names.count("DS_STORE_B64"),
      "ds_load_b128": names.count("DS_LOAD_B128"),
      "v_wmma": sum("WMMA" in n for n in names),
      "global_store_b16": names.count("GLOBAL_STORE_B16"),
      "s_barrier": names.count("S_BARRIER"),
      "s_cbranch_scc1": names.count("S_CBRANCH_SCC1"),
    },
    "samples": {"got_row0": [float(x) for x in got[0, :6]], "ref_row0": [float(x) for x in ref[0, :6]]},
  }


def main() -> int:
  p7c = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json", {})
  k = getenv("K", 4096)
  try:
    subset = run_authority_subset(k)
  except Exception as e:
    subset = {"correct": False, "error": repr(e), "shape": [16, 16, k]}
  counts = subset.get("instruction_counts") or {}
  gate = {
    "input_p7c_pass": p7c.get("verdict") == "PASS_BB5A10_P7C_SMALL_NUMERIC_CORRECTNESS" and bool(p7c.get("gate_pass")),
    "authority_k_depth_exercised": (subset.get("shape") or [0, 0, 0])[2] == 4096,
    "multi_k_loop_present": (counts.get("s_cbranch_scc1") or 0) > 0,
    "uses_selected_compatible_ds_store_b64": (counts.get("ds_store_b64") or 0) > 0,
    "uses_ds_load_b128": (counts.get("ds_load_b128") or 0) > 0,
    "uses_wmma": (counts.get("v_wmma") or 0) > 0,
    "has_output_store": (counts.get("global_store_b16") or 0) > 0,
    "relative_rmse_le_1e_3": subset.get("relative_rmse") is not None and float(subset.get("relative_rmse")) <= 0.001,
    "subset_correct": bool(subset.get("correct")),
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P7d_authority_shape_correctness",
    "schema": "amd_bb5a10_p7d_authority_correctness_result_v1",
    "verdict": "PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS" if gate_pass else "BLOCKED_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "authority_subset": subset,
    "gate": gate,
    "decision": "P7d passes: selected-compatible LDS staging is correct over full authority K=4096 for one 16x16 output tile. Next is P7e handoff packaging, then P8 timing." if gate_pass else
                "P7d blocked; debug K-loop accumulation, LDS reuse/barriers, or authority-subset numeric tolerance before P8.",
    "next_action": "Build P7e P8 handoff package." if gate_pass else "Fix P7d before P7e/P8.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json"],
  }
  write_json("bb5a10_p7d_authority_correctness_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "relative_rmse": subset.get("relative_rmse"),
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
