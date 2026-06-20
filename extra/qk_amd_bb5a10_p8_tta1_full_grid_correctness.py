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
  s_cmp_lt_i32, s_endpgm, s_load_b128, s_load_b64, s_lshl_b32, s_mov_b32, s_sendmsg, s_waitcnt,
  v_add_nc_u32_e32, v_and_b32_e32, v_cvt_f16_f32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32,
  v_mov_b32_e32, v_mul_lo_u32, v_wmma_f32_16x16x16_f16,
)
from tinygrad.uop.ops import KernelInfo, Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"
M, N, K = 512, 12288, 4096
TM, TN = 16, 16
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


def build_tta1_kernel(k: int = K, n: int = N) -> list[Any]:
  assert k % 16 == 0
  lds_a, lds_b = 0, TM * TN * 2
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
  e(s_lshl_b32(s[10], s[3], 4))                 # row_base = gidx1 * 16
  e(s_lshl_b32(s[11], s[2], 4))                 # col_base = gidx0 * 16
  e(v_and_b32_e32(v[1], 15, v[0]))              # lane within 16x16 tile
  e(v_add_nc_u32_e32(v[2], s[10], v[1])); e(v_mul_lo_u32(v[2], v[2], k * 2))
  e(v_mov_b32_e32(v[3], 0))
  e(v_add_nc_u32_e32(v[18], s[11], v[1])); e(v_mul_lo_u32(v[18], v[18], k * 2))
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

  e(v_and_b32_e32(v[10], 15, v[0]))             # output col within tile
  e(v_lshrrev_b32_e32(v[11], 4, v[0])); e(v_and_b32_e32(v[11], 1, v[11]))  # row parity
  e(v_add_nc_u32_e32(v[12], s[10], v[11]))      # row_base + parity
  e(v_add_nc_u32_e32(v[13], s[11], v[10]))      # col_base + lane
  e(v_mul_lo_u32(v[12], v[12], n))
  e(v_add_nc_u32_e32(v[12], v[12], v[13]))
  e(v_lshlrev_b32_e32(v[12], 1, v[12]))
  for i in range(8):
    e(v_cvt_f16_f32_e32(v[14], v[ACC+i]))
    e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
    if i < 7: e(v_add_nc_u32_e32(v[12], n * 4, v[12]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())

  for idx, target in branches:
    off = (labels[target] - sum(i.size() for i in insts[:idx+1])) // 4
    assert -32768 <= off <= 32767
    insts[idx].simm16 = off
  return insts


def sample_tiles() -> list[tuple[int, int]]:
  return [(0, 0), (0, N // TN - 1), (M // TM - 1, 0), (M // TM - 1, N // TN - 1), (M // (2 * TM), N // (2 * TN))]


def run_tta1() -> dict[str, Any]:
  rng = np.random.default_rng(11)
  a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
  bt_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)
  a = Tensor(a_np, device="AMD")
  bt = Tensor(bt_np, device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
  Tensor.realize(a, bt, c)
  insts = build_tta1_kernel()
  grid = (N // TN, M // TM, 1)

  def asm_kernel(A: UOp, B: UOp, C: UOp) -> UOp:
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=2048, addrspace=AddrSpace.LOCAL), (), "bb5a10_tta1_lds")
    gidxs = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(A.base, B.base, C.base, lds, *gidxs, UOp.special(32, "lidx0"),
                    arg=KernelInfo(name=colored("bb5a10_tta1_full_grid", "cyan"), estimates=Estimates(ops=M*N*K*2, mem=(M*K+N*K+M*N)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))))

  c = Tensor.custom_kernel(a, bt, c, fxn=asm_kernel)[2]
  with Context(DEBUG=0): run_linear(c.schedule_linear())
  got = c.float().numpy()
  samples: list[dict[str, Any]] = []
  errs: list[float] = []
  for gy, gx in sample_tiles():
    r0, c0 = gy * TM, gx * TN
    ref = a_np[r0:r0+TM, :].astype(np.float32) @ bt_np[c0:c0+TN, :].astype(np.float32).T
    tile = got[r0:r0+TM, c0:c0+TN]
    rel = float(np.sqrt(np.mean((tile - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-9))
    errs.append(rel)
    samples.append({
      "tile_gidx": [gy, gx],
      "row_col": [r0, c0],
      "relative_rmse": rel,
      "correct": rel <= 0.001,
      "got_row0": [float(x) for x in tile[0, :6]],
      "ref_row0": [float(x) for x in ref[0, :6]],
    })
  names = [getattr(i, "op_name", type(i).__name__) for i in insts]
  return {
    "authority_shape": [M, N, K],
    "tile": [TM, TN, K],
    "grid": list(grid),
    "local_size": [32, 1, 1],
    "sample_count": len(samples),
    "max_relative_rmse": max(errs),
    "correct": all(e <= 0.001 for e in errs),
    "samples": samples,
    "instruction_counts": {
      "s_lshl_b32": names.count("S_LSHL_B32"),
      "ds_store_b64": names.count("DS_STORE_B64"),
      "ds_load_b128": names.count("DS_LOAD_B128"),
      "v_wmma": sum("WMMA" in n for n in names),
      "global_store_b16": names.count("GLOBAL_STORE_B16"),
      "s_cbranch_scc1": names.count("S_CBRANCH_SCC1"),
    },
  }


def main() -> int:
  scope = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta_completion_scope_result.json", {})
  try:
    tta1 = run_tta1()
  except Exception as e:
    tta1 = {"correct": False, "error": repr(e)}
  counts = tta1.get("instruction_counts") or {}
  gate = {
    "input_completion_scope_pass": scope.get("verdict") == "PASS_BB5A10_P8_TTA_COMPLETION_SCOPE_READY" and bool(scope.get("gate_pass")),
    "full_grid_declared": tta1.get("grid") == [768, 32, 1],
    "authority_shape_declared": tta1.get("authority_shape") == [M, N, K],
    "uses_gidx_base_setup": (counts.get("s_lshl_b32") or 0) >= 2,
    "uses_selected_compatible_ds_store_b64": (counts.get("ds_store_b64") or 0) > 0,
    "uses_ds_load_b128": (counts.get("ds_load_b128") or 0) > 0,
    "uses_wmma": (counts.get("v_wmma") or 0) > 0,
    "has_output_store": (counts.get("global_store_b16") or 0) > 0,
    "has_k_loop": (counts.get("s_cbranch_scc1") or 0) > 0,
    "sampled_tiles_correct": bool(tta1.get("correct")),
    "max_rmse_le_1e_3": tta1.get("max_relative_rmse") is not None and float(tta1.get("max_relative_rmse")) <= 0.001,
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_TTA1_full_grid_correctness_bridge",
    "schema": "amd_bb5a10_p8_tta1_full_grid_correctness_v1",
    "verdict": "PASS_BB5A10_P8_TTA1_FULL_GRID_CORRECTNESS" if gate_pass else "BLOCKED_BB5A10_P8_TTA1_FULL_GRID_CORRECTNESS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "tta1": tta1,
    "gate": gate,
    "decision": "TTA1 passes: gidx0/gidx1 full-grid mapping is correct for sampled authority tiles. Next is TTA2 full-launch sampled correctness." if gate_pass else
                "TTA1 blocked; debug gidx base formulas, C output mapping, or K-loop address increments before timing.",
    "next_action": "Implement TTA2 authority-shape sampled correctness." if gate_pass else "Fix TTA1 before TTA2/TTA3/P8.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p8_tta_completion_scope_result.json"],
  }
  write_json("bb5a10_p8_tta1_full_grid_correctness_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "max_relative_rmse": tta1.get("max_relative_rmse"),
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
