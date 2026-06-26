#!/usr/bin/env python3
"""PALL physical-tile lifecycle gate.

Attempts the next step after the clean PALL score route: compose q.k score, online softmax state, and PV accumulation
inside one generated physical-tile lifecycle while retaining the score primitives (LDS + cross-lane + fdot2).

This is intentionally standalone. Route integration and W==D are only valid if this gate proves numeric correctness and
the required ISA in the same lifecycle kernel.
"""
from __future__ import annotations

import ctypes, json, os, pathlib, re, time, traceback
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
_F32 = dtypes.float32
_LOG2E = 1.4426950408889634

def _fc(v: float) -> UOp: return UOp.const(_F32, v)
def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * _LOG2E).exp2()

def flash_pall_score_state_pv_lifecycle_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S: int, Tc: int):
  """Composed physical lifecycle probe.

  Grid owns (kv-head, split, output column). Lanes shard the q.k head dimension for each token, reduce the score across
  the wave, then lane 0 updates online m/l and PV for the current output column.

  This proves lifecycle composition, not final optimal LaneMap reuse: q.k is still recomputed for each output column.
  That remaining gap is explicitly reported by the gate.
  """
  if Hd % 64 != 0: raise ValueError(f"PALL lifecycle requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; W = Hd + 2; R = Hd // 32; RP = Hd // 64
  def kernel(pout: UOp, q: UOp, cache: UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(W, 2, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    is_v = d < Hd
    is_l = d.eq(Hd)
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))

    r = UOp.range(R, 4, axis_type=AxisType.REDUCE)
    e = lane * R + r
    klds = UOp.placeholder((Hd,), dtypes.half, 193, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(dtypes.half), in_r).end(r)
    bar = UOp.barrier(UOp.group(kstage))

    rp = UOp.range(RP, 5, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    g_dot = UOp.range(G, 6)
    h_dot = kvh * G + g_dot
    dot = UOp.placeholder((G,), _F32, 194, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 7)
    dot_init = dot.after(kvh, s, d, j)[zi].store(0.0).end(zi)
    dot = dot.after(dot_init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h_dot * Hd + e2].cast(dtypes.half), q[h_dot * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dot.after(rp)[g_dot], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dot_upd = dot[g_dot].store(dot2).end(g_dot).end(rp)
    dot_f = dot.after(dot_upd)

    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((G,), _F32, 195, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 196, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 197, addrspace=AddrSpace.REG)
    za = UOp.range(G, 8)
    init = acc.after(kvh, s, d)[za].store(0.0).end(za)
    zl = UOp.range(G, 9)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 10)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)

    g_state = UOp.range(G, 11)
    old_m = mx.after(j)[g_state]
    sc_lane = in_r.where(dot_f[g_state] * (1.0 / (Hd ** 0.5)), _fc(-float("inf")))
    sc = _warp_reduce_sum_staged(sc_lane, lane, 32)
    new_m = old_m.maximum(sc)
    corr = in_r.where(_fexp(old_m - new_m), _fc(1.0))
    p = in_r.where(_fexp(sc - new_m), _fc(0.0))
    upd = acc[g_state].store(acc.after(j)[g_state] * corr + p * vd, lane.eq(0))
    upd = den.after(upd)[g_state].store(den.after(j)[g_state] * corr + p, lane.eq(0))
    upd = mx.after(upd)[g_state].store(new_m, lane.eq(0)).end(g_state).end(j)

    g2 = UOp.range(G, 12)
    af, lf, mf = acc.after(upd), den.after(upd), mx.after(upd)
    val = is_v.where(af[g2], is_l.where(lf[g2], mf[g2]))
    return pout[((kvh * G + g2) * S + s) * W + d].store(val, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_pall_score_state_pv_lifecycle_{Hq}_{Hd}"))
  return kernel

def _parse_desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _ = elf_loader(lib)
  rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
  rsrc1 = desc.compute_pgm_rsrc1; gran_vgpr = rsrc1 & 0x3f; gran_sgpr = (rsrc1 >> 6) & 0xf
  return {"vgpr": (gran_vgpr + 1) * 8, "sgpr": (gran_sgpr + 1) * 8, "lds": desc.group_segment_fixed_size,
          "scratch": desc.private_segment_fixed_size, "kernarg": desc.kernarg_size, "rsrc1": hex(rsrc1)}

def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  return system(f"{objdump} -d -", input=lib)

def _hist(asm: str) -> dict[str, int]:
  h = {"total": 0, "valu": 0, "s_inst": 0, "vmem_load": 0, "vmem_store": 0, "ds": 0, "cross_lane": 0, "fma_dot": 0, "scratch": 0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+)\b", line)
    if not m: continue
    op = m.group(1); h["total"] += 1
    if op.startswith("v_"): h["valu"] += 1
    if op.startswith("s_"): h["s_inst"] += 1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"] += 1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"] += 1
    if op.startswith("ds_"): h["ds"] += 1
    if op.startswith(("ds_bpermute", "ds_permute", "ds_swizzle")) or op.startswith("v_permlane"): h["cross_lane"] += 1
    if "fma" in op or "dot" in op or "mac" in op: h["fma_dot"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
  return h

def _flags(asm: str) -> dict[str, bool]:
  return {"has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
          "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
          "has_cross_lane": bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)", asm)),
          "has_spill": bool(re.search(r"\bscratch_(load|store)", asm))}

def _attempt() -> dict[str, Any]:
  dev = Device[Device.DEFAULT]; captured: dict[str, bytes] = {}; orig_runtime = dev.runtime
  def hook(name, lib, **kw):
    if name.startswith("flash_pall_score_state_pv_lifecycle") and name not in captured: captured[name] = lib
    return orig_runtime(name, lib, **kw)
  dev.runtime = hook
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  G, S, W = Hq // Hkv, (Tc + L - 1) // L, Hd + 2
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)),
    fxn=flash_pall_score_state_pv_lifecycle_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0].realize().numpy().reshape(Hq, S, W)
  ref = np.zeros((Hq, S, W), np.float32); scale = 1.0 / np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
        m = np.max(scores).astype(np.float32)
        p = np.exp2((scores - m) * 1.4426950408889634).astype(np.float32)
        ref[h, s, :Hd] = p @ cache[1, kvh, t0:t1, :].astype(np.float32)
        ref[h, s, Hd] = p.sum()
        ref[h, s, Hd + 1] = m
  diff = got - ref
  kernels = {}
  for name, lib in captured.items():
    asm = _disasm(lib); (OUT / f"disasm_{name}.txt").write_text(asm)
    d = _parse_desc(lib); d["hist"] = _hist(asm); d["primitive_flags"] = _flags(asm); kernels[name] = d
  return {"checked": True, "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L, "Tc": Tc, "S": S, "W": W},
          "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
                      "rmse": float(np.sqrt(np.mean(diff * diff))), "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))},
          "kernels": kernels}

def build() -> dict[str, Any]:
  try:
    attempt = _attempt()
  except Exception as e:
    tb = traceback.format_exc()
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "candidate_id": "decode_attention_physical_tile_pall_lifecycle", "verdict": "PALL_LIFECYCLE_BLOCKED__BUILDER_EXCEPTION",
            "exception_type": type(e).__name__, "exception": str(e), "traceback_tail": tb[-5000:],
            "decision": "Fix generated lifecycle builder before route integration or W==D."}
  k = next(iter(attempt.get("kernels", {}).values()), {}); flags = k.get("primitive_flags", {})
  numeric = attempt.get("numeric", {})
  numeric_pass = bool(numeric.get("finite") and numeric.get("max_abs", 1.0) <= 1e-3 and numeric.get("rel_rmse", 1.0) <= 1e-5)
  missing = [name for name, ok in (("DotLowering.v_dot2", flags.get("has_v_dot2")),
    ("TileMemory.lds_tile", flags.get("has_lds")), ("CrossLane.reduce_broadcast", flags.get("has_cross_lane"))) if not ok]
  if not numeric_pass: verdict = "PALL_LIFECYCLE_BLOCKED__NUMERIC"
  elif missing: verdict = "PALL_LIFECYCLE_BLOCKED__MISSING_PRIMITIVE_ISA"
  elif flags.get("has_spill"): verdict = "PALL_LIFECYCLE_BLOCKED__SPILL"
  else: verdict = "PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_pall_lifecycle", "verdict": verdict,
          "attempt": attempt, "missing_primitives": missing,
          "known_limit": "This lifecycle probe recomputes q.k per output column because current generated axis ownership cannot reuse one lane-sharded score across the PV output-column axis.",
          "decision": "Route only if verdict is PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT; otherwise use the blocker as the next codegen/lifecycle target."}

def main() -> int:
  os.chdir(ROOT); OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "pall_lifecycle_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"pall-lifecycle-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] in ("PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT", "PALL_LIFECYCLE_BLOCKED__NUMERIC", "PALL_LIFECYCLE_BLOCKED__MISSING_PRIMITIVE_ISA") else 1

if __name__ == "__main__": raise SystemExit(main())
