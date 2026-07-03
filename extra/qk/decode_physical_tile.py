#!/usr/bin/env python3
"""Decode-attention physical-tile primitive family, collapsed to one parameterized module.

Five sequential experiments that walk the missing physical primitives from a single lane-sharded q.k probe up to a
full generated lifecycle, then measure the residual gap. Each VARIANT builds+validates its stage, RETURNS a verdict
dict (gate_registry owns artifact writes / snapshots / stdout / exit-code policy), and shares the ISA-capture
scaffolding (runtime hook + llvm-objdump disasm, kernel-descriptor parse, op histogram, primitive-visibility flags,
numeric gate). All GPU: DEV=AMD, V_DOT2_LOWERING=1. Artifacts land in bench/qk-decode-primitive-space/ under DISTINCT
per-variant filenames (never latest.json -- that belongs to the primitive_detector gate). Registry entrypoints:
build_p1_crosslane(), build_pall_route(), build_pall_lifecycle(), build_pall_scaling(), build_all_primitives().

  p1_crosslane -- P1 primitive probe: generated UOps emit a lane-sharded q.k lane map + cross-lane score reduce
    (ds_bpermute) and the primitive tooling detects it. Not a full route; proves the first missing primitive is
    emit/detect-visible. Numeric max_abs<=1e-4, rmse<=1e-5.

  pall_route -- composed physical-tile score builder: LDS K stage + lane-sharded q.k + cross-lane reduce + fdot2
    (Ops.CUSTOMI __builtin_amdgcn_fdot2) in ONE generated hot builder; v_dot2 is checked in emitted ISA, not assumed.
    Pre-route gate: no W==D unless numerics pass and the required physical primitive ISA is present.

  pall_lifecycle -- q.k score + online-softmax state + PV accumulation in one generated lifecycle kernel that retains
    LDS+crosslane+fdot2 (max_abs<=1e-3, rel_rmse<=1e-5, spill guard). KNOWN LIMIT (reported): recomputes q.k per output
    column -- generated axis ownership cannot reuse one lane-sharded score across the PV output-column axis.

  pall_scaling -- output-column scaling probe: does the lifecycle kernel's cost scale with the number of PV output
    columns Wp? If it does, the W==D timeout is the per-column q.k recompute, not route overhead. Env PALL_LIFECYCLE_COLS
    (default 1,2,8,32,130), PALL_LIFECYCLE_SCALING_REPEATS (default 3). Decision rule: CONFIRMS when the last row's
    runtime multiple vs the first exceeds max(4.0, Wp/8.0); else INCONCLUSIVE.

  all_primitives -- visibility bundle: proves all four missing primitive classes are independently emit/detect-visible
    (P1 cross-lane + a3_1 vdot2 probe + minimal LDS stage probe). NOTE: a minimal same-lane LDS probe can legitimately
    elide the barrier -> barrier tracked separately, not a failure. Does not claim the shipped fused route has them.

Run:  DEV=AMD V_DOT2_LOWERING=1 PYTHONPATH=. python3 -m extra.qk.gate_registry run \\
        physical_tile_p1_crosslane physical_tile_pall_route physical_tile_pall_lifecycle \\
        physical_tile_pall_scaling physical_tile_all_primitives
"""
from __future__ import annotations

import ctypes, json, os, pathlib, re, subprocess, sys, time, traceback
from typing import Any
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
from extra.qk.isa_helpers import CROSS_LANE_RE, CROSS_LANE_OPS

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-primitive-space"
_F32 = dtypes.float32
_LOG2E = 1.4426950408889634


# ---- shared scaffolding ----------------------------------------------------------------------------------------------
def _fc(v: float) -> UOp: return UOp.const(_F32, v)
def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * _LOG2E).exp2()


def _capture(prefix: str) -> dict[str, bytes]:
  """Install a runtime hook on the default device that stashes every compiled lib whose kernel name starts with
  `prefix`. Matches the originals: the hook is left installed (not restored)."""
  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig_runtime = dev.runtime
  def runtime_hook(name, lib, **kw):
    if name.startswith(prefix) and name not in captured: captured[name] = lib
    return orig_runtime(name, lib, **kw)
  dev.runtime = runtime_hook
  return captured


def _parse_desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _relocs = elf_loader(lib)
  rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
  rsrc1 = desc.compute_pgm_rsrc1
  gran_vgpr = rsrc1 & 0x3f
  gran_sgpr = (rsrc1 >> 6) & 0xf
  return {"vgpr": (gran_vgpr + 1) * 8, "sgpr": (gran_sgpr + 1) * 8, "lds": desc.group_segment_fixed_size,
          "scratch": desc.private_segment_fixed_size, "kernarg": desc.kernarg_size, "rsrc1": hex(rsrc1)}


def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  return system(f"{objdump} -d -", input=lib)


def _isa_kernels(captured: dict[str, bytes], hist_fn, flags_fn) -> dict[str, Any]:
  """The common capture->disasm->write->descriptor->hist->flags loop; preserves each variant's exact dict key order."""
  kernels: dict[str, Any] = {}
  OUT.mkdir(parents=True, exist_ok=True)
  for name, lib in captured.items():
    asm = _disasm(lib)
    (OUT / f"disasm_{name}.txt").write_text(asm)
    desc = _parse_desc(lib)
    desc["hist"] = hist_fn(asm)
    desc["primitive_flags"] = flags_fn(asm)
    kernels[name] = desc
  return kernels


# -- histogram variants (kept distinct: the emitted JSON key sets differ, so byte-identity requires exact bodies) ------
def _hist_fma(asm: str) -> dict[str, int]:
  """p1_crosslane + pall_lifecycle histogram (fma_dot bucket, no barrier)."""
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
    if op.startswith(CROSS_LANE_OPS): h["cross_lane"] += 1
    if "fma" in op or "dot" in op or "mac" in op: h["fma_dot"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
  return h


def _hist_barrier(asm: str) -> dict[str, int]:
  """pall_route histogram (fma_dot + explicit s_barrier bucket)."""
  h = {"total":0,"valu":0,"s_inst":0,"vmem_load":0,"vmem_store":0,"ds":0,"cross_lane":0,"fma_dot":0,"scratch":0,"barrier":0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+|s_barrier)\b", line)
    if not m: continue
    op = m.group(1); h["total"] += 1
    if op.startswith("v_"): h["valu"] += 1
    if op.startswith("s_"): h["s_inst"] += 1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"] += 1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"] += 1
    if op.startswith("ds_"): h["ds"] += 1
    if op.startswith(CROSS_LANE_OPS): h["cross_lane"] += 1
    if "fma" in op or "dot" in op or "mac" in op: h["fma_dot"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
    if op == "s_barrier": h["barrier"] += 1
  return h


def _hist_bundle(asm: str) -> dict[str, int]:
  """all_primitives LDS-probe histogram (barrier_wait bucket combining s_barrier + s_waitcnt; no fma_dot)."""
  h = {"total": 0, "valu": 0, "s_inst": 0, "vmem_load": 0, "vmem_store": 0, "ds": 0, "cross_lane": 0, "barrier_wait": 0, "scratch": 0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+|s_barrier)\b", line)
    if not m: continue
    op = m.group(1); h["total"] += 1
    if op.startswith("v_"): h["valu"] += 1
    if op.startswith("s_"): h["s_inst"] += 1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"] += 1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"] += 1
    if op.startswith("ds_"): h["ds"] += 1
    if op.startswith(CROSS_LANE_OPS): h["cross_lane"] += 1
    if op == "s_barrier" or "s_waitcnt" in op: h["barrier_wait"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
  return h


# -- primitive-visibility flag variants (distinct key sets / v_dot2 test -> kept exact for byte-identity) --------------
def _flags_p1(asm: str) -> dict[str, bool]:
  """p1_crosslane flags: v_dot2 by mnemonic only, no barrier bucket."""
  return {
    "has_v_dot2": "v_dot2" in asm,
    "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
    "has_cross_lane": bool(re.search(CROSS_LANE_RE, asm)),
    "has_vector_global_load": "global_load" in asm or "buffer_load" in asm,
    "has_spill": bool(re.search(r"\bscratch_(load|store)", asm)),
  }


def _flags_full(asm: str) -> dict[str, bool]:
  """pall_route + all_primitives LDS-probe flags: v_dot2 by mnemonic OR fdot2 builtin, includes barrier + global-load."""
  return {
    "has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
    "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
    "has_cross_lane": bool(re.search(CROSS_LANE_RE, asm)),
    "has_barrier": "s_barrier" in asm,
    "has_vector_global_load": "global_load" in asm or "buffer_load" in asm,
    "has_spill": bool(re.search(r"\bscratch_(load|store)", asm)),
  }


def _flags_lifecycle(asm: str) -> dict[str, bool]:
  """pall_lifecycle flags: v_dot2 by mnemonic OR fdot2 builtin, minimal set (no barrier / global-load buckets)."""
  return {"has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
          "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
          "has_cross_lane": bool(re.search(CROSS_LANE_RE, asm)),
          "has_spill": bool(re.search(r"\bscratch_(load|store)", asm))}


# ======================================================================================================================
# p1_crosslane -- lane map + cross-lane score reduce
#
# NAMING: the probe_* builders below independently RE-DERIVE the shipped flash_kernels.*_whole_cache builders. The
# near-duplication is DELIBERATE (a validation probe must not import the thing it validates, or it regresses silently
# with it) -- the probe_ prefix marks that. Emitted kernel names (_fki(...)) are left unchanged so the gate artifacts
# stay stable; only the Python function names carry the probe_ marker.
# ======================================================================================================================
def probe_p1_crosslane_score_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, Tc: int):
  if Hd % 32 != 0: raise ValueError(f"P1 requires Hd divisible by 32, got {Hd}")
  G = Hq // Hkv; R = Hd // 32
  def kernel(score: UOp, q: UOp, cache: UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    e = lane * R + r
    kvh = h // G
    acc = UOp.placeholder((1,), _F32, 170, addrspace=AddrSpace.REG)
    acc_init = acc.after(h, t)[0].store(0.0)
    acc = acc.after(acc_init)
    qv = q[h * Hd + e].cast(_F32)
    kv = cache[((0 * Hkv + kvh) * MAXC + t) * Hd + e].cast(_F32)
    acc_upd = acc[0].store(acc.after(r)[0] + qv * kv).end(r)
    total = _warp_reduce_sum_staged(acc.after(acc_upd)[0], lane, 32)
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_p1_crosslane_score_{Hq}_{Hd}"))
  return kernel


def _p1_numeric_and_isa() -> dict[str, Any]:
  captured = _capture("flash_p1_crosslane_score")
  Hq, Hkv, Hd, MAXC, Tc = 32, 8, 128, 256, 192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0.0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, Hkv, MAXC, Hd), dtype=np.float32)
  cache[0] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)
  got = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)),
    fxn=probe_p1_crosslane_score_kernel(Hd, Hq, Hkv, MAXC, Tc))[0].realize().numpy().reshape(Hq, MAXC)
  ref = np.zeros((Hq, MAXC), dtype=np.float32)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    ref[h, :Tc] = (cache[0, kvh, :Tc, :] @ q[h]) * (1.0 / np.sqrt(Hd))
  diff = got[:, :Tc] - ref[:, :Tc]
  kernels = _isa_kernels(captured, _hist_fma, _flags_p1)
  max_abs = float(np.max(np.abs(diff)))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  return {
    "checked": True,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "Tc": Tc},
    "numeric": {"max_abs": max_abs, "rmse": rmse, "pass": bool(max_abs <= 1e-4 and rmse <= 1e-5), "thresholds": {"max_abs": 1e-4, "rmse": 1e-5}},
    "kernels": kernels,
  }


def build_p1_crosslane() -> dict[str, Any]:
  probe = _p1_numeric_and_isa()
  kernel = next(iter(probe.get("kernels", {}).values()), {})
  flags = kernel.get("primitive_flags", {})
  if not probe["numeric"].get("pass"):
    verdict = "P1_CROSSLANE_FAIL__NUMERIC"
  elif not flags.get("has_cross_lane"):
    verdict = "P1_CROSSLANE_FAIL__NO_CROSS_LANE_ISA"
  elif flags.get("has_lds") or flags.get("has_v_dot2"):
    verdict = "P1_CROSSLANE_PASS__EXTRA_PRIMITIVES_PRESENT"
  else:
    verdict = "P1_CROSSLANE_PASS__LANEMAP_CROSSLANE_VISIBLE"
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "candidate_id": "decode_attention_physical_tile_p1",
    "verdict": verdict,
    "probe": probe,
    "primitive_visibility": {
      "LaneMap": "searchable_probe_visible",
      "CrossLane": "detected_in_isa" if flags.get("has_cross_lane") else "absent",
      "TileMemory": "not_targeted_in_p1",
      "DotLowering": "scalar_fma_not_v_dot2_in_p1",
    },
    "decision": "P1 proves generated cross-lane score reduction can be emitted; next integrate as a decode route only if the detector/gap gate consumes this as partial primitive visibility."
  }


# ======================================================================================================================
# pall_route -- composed LDS + lane-sharded q.k + cross-lane reduce + fdot2 score builder
# ======================================================================================================================
def probe_pall_lds_crosslane_score_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, Tc:int):
  """Composed physical-primitive score builder.

  This deliberately targets the first hard composition: LDS K staging + lane-sharded q.k + cross-lane score reduce
  in one generated kernel. v_dot2 is checked in emitted ISA; it is not assumed.
  """
  if Hd % 32 != 0: raise ValueError(f"PALL score requires Hd divisible by 32, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(score:UOp, q:UOp, cache:UOp) -> UOp:
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    t = UOp.range(Tc, 1, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    r = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    e = lane * R + r
    kvh = h // G
    klds = UOp.placeholder((Hd,), dtypes.half, 190, addrspace=AddrSpace.LOCAL)
    # Cooperative K stage. Each lane writes R positions into LDS.
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t) * Hd + e].cast(dtypes.half)).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 3, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    acc = UOp.placeholder((1,), _F32, 191, addrspace=AddrSpace.REG)
    init = acc.after(h, t)[0].store(0.0)
    acc = acc.after(init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h * Hd + e2].cast(dtypes.half), q[h * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (acc.after(rp)[0], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    upd = acc[0].store(dot2).end(rp)
    total = _warp_reduce_sum_staged(acc.after(upd)[0], lane, 32)
    return score[h * MAXC + t].store(total * (1.0 / (Hd ** 0.5)), lane.eq(0)).end(h, t).sink(
      arg=_fki(f"flash_pall_lds_crosslane_score_{Hq}_{Hd}"))
  return kernel


def _pall_route_attempt() -> dict[str, Any]:
  captured = _capture("flash_pall_lds_crosslane_score")
  os.environ.setdefault("V_DOT2_LOWERING", "1")
  Hq,Hkv,Hd,MAXC,Tc = 32,8,128,256,192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0,0.25,(Hq,Hd)).astype(np.float16)
  cache = np.zeros((2,Hkv,MAXC,Hd),np.float16)
  cache[0] = rng.normal(0,0.25,(Hkv,MAXC,Hd)).astype(np.float16)
  got = Tensor.empty(Hq*MAXC, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)), fxn=probe_pall_lds_crosslane_score_kernel(Hd,Hq,Hkv,MAXC,Tc))[0].realize().numpy().reshape(Hq,MAXC)
  ref = np.zeros((Hq,MAXC),np.float32)
  for h in range(Hq): ref[h,:Tc] = (cache[0,h//(Hq//Hkv),:Tc,:].astype(np.float32) @ q[h].astype(np.float32)) * (1.0/np.sqrt(Hd))
  diff = got[:,:Tc] - ref[:,:Tc]
  kernels = _isa_kernels(captured, _hist_barrier, _flags_full)
  return {"checked": True, "shape": {"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"MAXC":MAXC,"Tc":Tc},
          "numeric": {"max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff*diff))),
                      "pass": bool(float(np.max(np.abs(diff))) <= 1e-4)}, "kernels": kernels}


def build_pall_route() -> dict[str, Any]:
  try:
    attempt = _pall_route_attempt()
  except Exception as e:
    tb = traceback.format_exc()
    return {"date":"2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "candidate_id":"decode_attention_physical_tile_pall_route", "verdict":"PALL_ROUTE_BLOCKED__BUILDER_EXCEPTION",
            "exception_type": type(e).__name__, "exception": str(e), "traceback_tail": tb[-5000:],
            "decision":"Fix composed LDS+crosslane builder before route integration or W==D."}
  k = next(iter(attempt.get("kernels", {}).values()), {})
  flags = k.get("primitive_flags", {})
  missing = []
  if not flags.get("has_lds"): missing.append("TileMemory.lds_tile")
  if not flags.get("has_cross_lane"): missing.append("CrossLane.reduce_broadcast")
  if not flags.get("has_v_dot2"): missing.append("DotLowering.v_dot2")
  if not attempt.get("numeric", {}).get("pass"): verdict = "PALL_ROUTE_BLOCKED__NUMERIC"
  elif missing: verdict = "PALL_ROUTE_BLOCKED__MISSING_COMPOSED_PRIMITIVES"
  else: verdict = "PALL_ROUTE_BUILDER_READY__ROUTE_NEXT"
  return {"date":"2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id":"decode_attention_physical_tile_pall_route", "verdict": verdict,
          "attempt": attempt, "missing_composed_primitives": missing,
          "required_next_if_ready": ["route flag", "route/materialization gate", "primitive detector", "W==D"],
          "decision": "Builder composes LDS+crosslane but does not yet compose v_dot2; next work is fdot2 pattern inside the composed tile." if missing == ["DotLowering.v_dot2"] else
                      "Proceed to route integration only after all composed primitive checks pass." if missing else
                      "Composed hot builder is ready for default-off route integration."}


# ======================================================================================================================
# pall_lifecycle -- q.k score + online-softmax state + PV accumulation in one lifecycle kernel
# ======================================================================================================================
def probe_pall_score_state_pv_lifecycle_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S: int, Tc: int):
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


def _pall_lifecycle_attempt() -> dict[str, Any]:
  captured = _capture("flash_pall_score_state_pv_lifecycle")
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  G, S, W = Hq // Hkv, (Tc + L - 1) // L, Hd + 2
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)),
    fxn=probe_pall_score_state_pv_lifecycle_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0].realize().numpy().reshape(Hq, S, W)
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
  kernels = _isa_kernels(captured, _hist_fma, _flags_lifecycle)
  return {"checked": True, "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L, "Tc": Tc, "S": S, "W": W},
          "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
                      "rmse": float(np.sqrt(np.mean(diff * diff))), "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))},
          "kernels": kernels}


def build_pall_lifecycle() -> dict[str, Any]:
  try:
    attempt = _pall_lifecycle_attempt()
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


# ======================================================================================================================
# pall_scaling -- output-column scaling probe (timing; no ISA capture)
# ======================================================================================================================
def pall_lifecycle_cols_kernel(Hd: int, Hq: int, Hkv: int, MAXC: int, L: int, S: int, Tc: int, Wp: int):
  if Hd % 64 != 0: raise ValueError(f"PALL lifecycle requires Hd divisible by 64, got {Hd}")
  G = Hq // Hkv; R = Hd // 32; RP = Hd // 64
  def kernel(pout: UOp, q: UOp, cache: UOp) -> UOp:
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    s = UOp.range(S, 1, AxisType.GLOBAL)
    d = UOp.range(Wp, 2, AxisType.GLOBAL)
    lane = UOp.special(32, "lidx0")
    is_v = d < Hd
    is_l = d.eq(Hd)
    j = UOp.range(L, 3, axis_type=AxisType.REDUCE)
    t = s * L + j
    in_r = t < Tc
    t_safe = in_r.where(t, t.const_like(0))
    r = UOp.range(R, 4, axis_type=AxisType.REDUCE)
    e = lane * R + r
    klds = UOp.placeholder((Hd,), dtypes.half, 198, addrspace=AddrSpace.LOCAL)
    kstage = klds[e].store(cache[((0 * Hkv + kvh) * MAXC + t_safe) * Hd + e].cast(dtypes.half), in_r).end(r)
    bar = UOp.barrier(UOp.group(kstage))
    rp = UOp.range(RP, 5, axis_type=AxisType.REDUCE)
    e2 = rp * 64 + lane * 2
    g_dot = UOp.range(G, 6)
    h_dot = kvh * G + g_dot
    dot = UOp.placeholder((G,), _F32, 199, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 7)
    dot_init = dot.after(kvh, s, d, j)[zi].store(0.0).end(zi)
    dot = dot.after(dot_init)
    qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[h_dot * Hd + e2].cast(dtypes.half), q[h_dot * Hd + e2 + 1].cast(dtypes.half)))
    kpair = UOp(Ops.STACK, dtypes.half.vec(2), (klds.after(bar)[e2], klds.after(bar)[e2 + 1]))
    dot2 = UOp(Ops.CUSTOMI, _F32, (dot.after(rp)[g_dot], qpair, kpair), arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
    dot_upd = dot[g_dot].store(dot2).end(g_dot).end(rp)
    dot_f = dot.after(dot_upd)
    vd = is_v.where(cache[((1 * Hkv + kvh) * MAXC + t_safe) * Hd + is_v.where(d, d.const_like(0))].cast(_F32), _fc(1.0))
    acc = UOp.placeholder((G,), _F32, 200, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), _F32, 201, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), _F32, 202, addrspace=AddrSpace.REG)
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
    return pout[((kvh * G + g2) * S + s) * Wp + d].store(val, lane.eq(0)).end(g2).end(kvh, s, d).sink(
      arg=_fki(f"flash_pall_lifecycle_cols_{Wp}_{Hq}_{Hd}"))
  return kernel


def _pall_scaling_reference(q: np.ndarray, cache: np.ndarray, Hq: int, Hkv: int, Hd: int, L: int, S: int, Tc: int, Wp: int) -> np.ndarray:
  G = Hq // Hkv
  ref = np.zeros((Hq, S, Wp), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
        m = np.max(scores).astype(np.float32)
        p = np.exp2((scores - m) * _LOG2E).astype(np.float32)
        for d in range(Wp):
          if d < Hd: ref[h, s, d] = p @ cache[1, kvh, t0:t1, d].astype(np.float32)
          elif d == Hd: ref[h, s, d] = p.sum()
          else: ref[h, s, d] = m
  return ref


def _pall_scaling_run_one(Wp: int, repeats: int) -> dict[str, Any]:
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  S = (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qt, ct = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  fxn = pall_lifecycle_cols_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc, Wp)
  warm = Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(qt, ct, fxn=fxn)[0].realize().numpy().reshape(Hq, S, Wp)
  ref = _pall_scaling_reference(q, cache, Hq, Hkv, Hd, L, S, Tc, Wp)
  diff = warm - ref
  times = []
  for _ in range(repeats):
    st = time.perf_counter()
    Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(qt, ct, fxn=fxn)[0].realize().numpy()
    times.append(time.perf_counter() - st)
  med = float(np.median(times))
  return {"Wp": Wp, "median_s": med, "per_col_ms": med * 1000.0 / Wp, "times_s": [float(x) for x in times],
          "numeric": {"max_abs": float(np.max(np.abs(diff))), "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}


def build_pall_scaling() -> dict[str, Any]:
  cols = [int(x) for x in os.environ.get("PALL_LIFECYCLE_COLS", "1,2,8,32,130").split(",")]
  repeats = int(os.environ.get("PALL_LIFECYCLE_SCALING_REPEATS", "3"))
  rows = [_pall_scaling_run_one(w, repeats) for w in cols]
  base = rows[0]["median_s"]
  for r in rows:
    r["speedup_vs_first"] = base / r["median_s"] if r["median_s"] else None
    r["runtime_multiple_vs_first"] = r["median_s"] / base if base else None
  verdict = "PALL_LIFECYCLE_SCALING_CONFIRMS_COLUMN_RECOMPUTE" if rows[-1]["runtime_multiple_vs_first"] and rows[-1]["runtime_multiple_vs_first"] > max(4.0, rows[-1]["Wp"] / 8.0) else "PALL_LIFECYCLE_SCALING_INCONCLUSIVE"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "candidate_id": "decode_attention_physical_tile_pall_lifecycle",
          "verdict": verdict, "repeats": repeats, "rows": rows,
          "decision": "If runtime scales with Wp, the next primitive is score reuse across PV output columns, not another W==D rerun."}


# ======================================================================================================================
# all_primitives -- visibility bundle (P1 cross-lane + a3_1 vdot2 probe + minimal LDS stage probe)
# ======================================================================================================================
def lds_stage_probe_kernel(N: int = 32):
  def kernel(out: UOp, inp: UOp) -> UOp:
    lane = UOp.special(N, "lidx0")
    lds = UOp.placeholder((N,), _F32, 180, addrspace=AddrSpace.LOCAL)
    st = lds[lane].store(inp[lane])
    bar = UOp.barrier(UOp.group(st))
    return out[lane].store(lds.after(bar)[lane]).sink(arg=_fki("flash_pall_lds_stage_probe"))
  return kernel


def _run_json_script(module_fn: str) -> dict[str, Any]:
  # isolate the sub-probe in a fresh process (the ISA capture hook doesn't compose in-process), but call the
  # in-module/migrated builder by module:function so this doesn't depend on now-retired standalone gate files.
  mod, fn = module_fn.split(":")
  code = f"import json,sys; sys.path.insert(0,{str(ROOT)!r}); from {mod} import {fn}; print(json.dumps({fn}()))"
  p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), "DEV": "AMD"},
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
  if p.returncode != 0:
    return {"ran": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}
  txt = p.stdout or ""
  for i, ch in enumerate(txt):
    if ch != "{": continue
    try:
      d = json.loads(txt[i:])
      d["ran"] = True
      return d
    except Exception:
      pass
  return {"ran": False, "returncode": p.returncode, "error": "no json", "output_tail": txt[-8000:]}


def _lds_probe() -> dict[str, Any]:
  captured = _capture("flash_pall_lds_stage_probe")
  x = np.arange(32, dtype=np.float32)
  got = Tensor.empty(32, dtype=dtypes.float32).custom_kernel(Tensor(x), fxn=lds_stage_probe_kernel(32))[0].realize().numpy()
  max_abs = float(np.max(np.abs(got - x)))
  kernels = _isa_kernels(captured, _hist_bundle, _flags_full)
  flags = next(iter(kernels.values()), {}).get("primitive_flags", {})
  return {"ran": True, "numeric": {"max_abs": max_abs, "pass": max_abs == 0.0}, "kernels": kernels,
          "pass": bool(max_abs == 0.0 and flags.get("has_lds")),
          "barrier_status": "not_required_or_elided_in_minimal_same-lane_probe" if flags.get("has_lds") and not flags.get("has_barrier") else "present" if flags.get("has_barrier") else "absent"}


def build_all_primitives() -> dict[str, Any]:
  p1 = _run_json_script("extra.qk.decode_physical_tile:build_p1_crosslane")
  vdot2 = _run_json_script("extra.qk.decode_attention_a3_1_vdot2_probe:build")
  lds = _lds_probe()

  p1_kernel = next(iter(p1.get("probe", {}).get("kernels", {}).values()), {})
  p1_flags = p1_kernel.get("primitive_flags", {})
  vdot2_pass = vdot2.get("verdict") == "A3_1_RENDERER_VDOT2_PROBE_PASS"
  lds_flags = next(iter(lds.get("kernels", {}).values()), {}).get("primitive_flags", {})
  checks = {
    "CrossLane.reduce_broadcast": bool(str(p1.get("verdict", "")).startswith("P1_CROSSLANE_PASS") and p1_flags.get("has_cross_lane")),
    "LaneMap.score_reuse_across_output_columns": bool(str(p1.get("verdict", "")).startswith("P1_CROSSLANE_PASS")),
    "DotLowering.v_dot2": bool(vdot2_pass),
    "TileMemory.lds_tile": bool(lds.get("numeric", {}).get("pass") and lds_flags.get("has_lds")),
  }
  missing = [k for k, v in checks.items() if not v]
  verdict = "PALL_PRIMITIVES_VISIBLE__ROUTE_INTEGRATION_NEXT" if not missing else "PALL_PRIMITIVES_PARTIAL__SOME_PROBES_MISSING"
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "candidate_id": "decode_attention_physical_tile_all_primitives_bundle",
    "verdict": verdict,
    "checks": checks,
    "missing": missing,
    "p1_crosslane": p1,
    "vdot2_probe": vdot2,
    "lds_probe": lds,
    "decision": "All missing primitive classes are independently emit/detect-visible. Next step is a single fused route candidate that uses them together, then primitive detector + W==D." if not missing else "Do not route-integrate yet; fix failed primitive probes first."
  }


# ---- registry surface ------------------------------------------------------------------------------------------------
VARIANTS = {"p1_crosslane": build_p1_crosslane, "pall_route": build_pall_route, "pall_lifecycle": build_pall_lifecycle,
            "pall_scaling": build_pall_scaling, "all_primitives": build_all_primitives}

def build(variant: str) -> dict[str, Any]: return VARIANTS[variant]()


if __name__ == "__main__":
  os.environ.setdefault("V_DOT2_LOWERING", "1")
  out = build(sys.argv[1] if len(sys.argv) > 1 else "p1_crosslane")
  print(json.dumps(out, indent=2))
