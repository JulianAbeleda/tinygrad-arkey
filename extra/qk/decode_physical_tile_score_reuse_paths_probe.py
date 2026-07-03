#!/usr/bin/env python3
"""Probe the two plausible fixes for PALL lifecycle q.k-per-column recompute.

Path A: score-once split state. Compute online m/l once per (kvh, split, head) with the PALL physical score path.
        This proves score-once is expressible, but it does not produce PV.

Path B: score-broadcast fused PV. Compute q.k once per token, then update several PV output columns from that score.
        This is the actual primitive we need for a fast fused lifecycle.
"""
from __future__ import annotations

import ctypes, json, os, pathlib, re, time, traceback
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes, Device
from extra.qk.decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-primitive-space"

def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  return system(f"{objdump} -d -", input=lib)

def _desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _ = elf_loader(lib)
  ro = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[ro:ro+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
  rsrc1 = desc.compute_pgm_rsrc1
  return {"vgpr": ((rsrc1 & 0x3f) + 1) * 8, "sgpr": (((rsrc1 >> 6) & 0xf) + 1) * 8,
          "lds": desc.group_segment_fixed_size, "scratch": desc.private_segment_fixed_size}

def _flags(asm: str) -> dict[str, bool]:
  return {"has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
          "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
          "has_cross_lane": bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)", asm)),
          "has_spill": bool(re.search(r"\bscratch_(load|store)", asm))}

def _run_probe() -> dict[str, Any]:
  dev = Device[Device.DEFAULT]; captured: dict[str, bytes] = {}; orig = dev.runtime
  def hook(name, lib, **kw):
    if name.startswith("flash_pall_score_") and name not in captured: captured[name] = lib
    return orig(name, lib, **kw)
  dev.runtime = hook
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  G, S = Hq // Hkv, (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qt, ct = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  state = Tensor.empty(Hq * S * 2, dtype=dtypes.float32).custom_kernel(qt, ct,
    fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0].realize().numpy().reshape(Hq, S, 2)
  ref_state = np.zeros((Hq, S, 2), np.float32)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * (1.0 / np.sqrt(Hd))
        m = np.max(scores).astype(np.float32)
        p = np.exp2((scores - m) * _LOG2E).astype(np.float32)
        ref_state[h, s, 0], ref_state[h, s, 1] = p.sum(), m
  state_diff = state - ref_state
  rows = []
  for Wp in (1, 8, 32, 128):
    fxn = score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc, Wp)
    got = Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(Tensor(state.reshape(-1)), qt, ct, fxn=fxn)[0].realize().numpy().reshape(Hq, S, Wp)
    ref = np.zeros((Hq, S, Wp), np.float32)
    for kvh in range(Hkv):
      for s in range(S):
        t0, t1 = s * L, min((s + 1) * L, Tc)
        for g in range(G):
          h = kvh * G + g
          scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * (1.0 / np.sqrt(Hd))
          m = np.max(scores).astype(np.float32)
          p = np.exp2((scores - m) * _LOG2E).astype(np.float32)
          for c in range(Wp): ref[h, s, c] = p @ cache[1, kvh, t0:t1, c].astype(np.float32)
    times = []
    for _ in range(3):
      st = time.perf_counter()
      Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(Tensor(state.reshape(-1)), qt, ct, fxn=fxn)[0].realize().numpy()
      times.append(time.perf_counter() - st)
    diff = got - ref
    rows.append({"Wp": Wp, "median_s": float(np.median(times)), "numeric": {"max_abs": float(np.max(np.abs(diff))),
      "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}})
  kernels = {}
  for name, lib in captured.items():
    asm = _disasm(lib); (OUT / f"disasm_{name}.txt").write_text(asm)
    d = _desc(lib); d["primitive_flags"] = _flags(asm); kernels[name] = d
  return {"score_once_state": {"numeric": {"max_abs": float(np.max(np.abs(state_diff))),
            "rel_rmse": float(np.sqrt(np.mean(state_diff * state_diff)) / (np.sqrt(np.mean(ref_state * ref_state)) + 1e-12))}},
          "score_broadcast_pv": {"rows": rows}, "kernels": kernels}

def build() -> dict[str, Any]:
  try:
    attempt = _run_probe()
  except Exception as e:
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "SCORE_REUSE_PATHS_BLOCKED__BUILDER_EXCEPTION", "exception_type": type(e).__name__,
            "exception": str(e), "traceback_tail": traceback.format_exc()[-5000:]}
  rows = attempt["score_broadcast_pv"]["rows"]
  mult = rows[-1]["median_s"] / rows[0]["median_s"] if rows and rows[0]["median_s"] else None
  score_once_pass = attempt["score_once_state"]["numeric"]["max_abs"] <= 1e-3 and attempt["score_once_state"]["numeric"]["rel_rmse"] <= 1e-5
  broadcast_numeric_pass = all(r["numeric"]["max_abs"] <= 1e-3 and r["numeric"]["rel_rmse"] <= 1e-5 for r in rows)
  verdict = "SCORE_REUSE_PATHS_PASS__BROADCAST_PROBE_READY" if score_once_pass and broadcast_numeric_pass and mult is not None and mult < 16.0 else \
            "SCORE_REUSE_PATHS_PARTIAL__SCORE_ONCE_ONLY" if score_once_pass else "SCORE_REUSE_PATHS_FAIL"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
          "attempt": attempt, "runtime_multiple_32col_vs_1col": mult,
          "decision": "If broadcast passes with sublinear scaling, route it next. If only score-once passes, the missing primitive remains fused score reuse across PV columns."}

def main() -> int:
  os.chdir(ROOT); OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_reuse_paths_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-reuse-paths-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__": raise SystemExit(main())
