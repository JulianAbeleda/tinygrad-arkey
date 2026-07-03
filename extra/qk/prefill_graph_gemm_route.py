from __future__ import annotations

from functools import lru_cache

from tinygrad import Tensor, dtypes, getenv
from tinygrad.device import Device
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import colored
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from extra.qk.prefill import wmma as ref


@lru_cache(maxsize=None)
def _resolve_schedule(out_f: int, in_f: int):
  # TG-P4 refactor (NFC): resolve the prefill GEMM schedule parameters (tile/waves/pipeline/role-selective) into a
  # data dict. Byte-for-byte the same computation that used to live inline in _kernel; both _kernel (legacy/rollback
  # path) and the spec-driven generated path (extra/qk/prefill_schedule_spec.py) resolve through here, so the emitted
  # schedule is identical no matter which selector fired.
  import os
  m, n, k = 512, out_f, in_f
  # DEFAULT: eightwave layout over cross-iteration double-buffer. DBUF beat the old PLRA default in whole-prefill;
  # eightwave then confirmed another +3.1/+2.8/+2.7/+2.3/+1.9% over baseline at 512..8192. The combined
  # eightwave+old_plra path regressed hard, so explicit CFG/DBUF/PLRA/PLRAB overrides suppress default eightwave unless
  # PREFILL_GEMM_8WAVE=1 is also explicitly set. Reversible: PREFILL_GEMM_8WAVE=0. Old PLRA route:
  # PREFILL_GEMM_8WAVE=0 PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1. See
  # docs/prefill-eightwave-oldplra-interaction-scope-20260624.md.
  waves_m, waves_n, wm, wn, bk, pad, dbuf, plra = 2, 2, 4, 4, 32, 16, 1, 0
  if out_f <= 1024:  # small-N roles (kv_proj) are WG-starved at BN=128 -> halve BN to 2x the workgroups
    waves_n, wn = 1, 4
  # Phase-B per-shape config OVERRIDE (additive, default unchanged): PREFILL_GEMM_CFG_{out_f}_{in_f}="wm,wn,wavesn,bk,pad,dbuf,plra"
  ov = os.environ.get(f"PREFILL_GEMM_CFG_{out_f}_{in_f}")
  plrab = 0
  if ov:
    wm, wn, waves_n, bk, pad, dbuf, plra = (int(x) for x in ov.split(","))
  # Adversarial-audit Tensile-like 8-wave layout: W4x2 T2x4 -> 128x128 tile, acc=64 (half), DBUF (block prefetch) +
  # PLRAB (substep A+B prefetch) fit at ~188 VGPR -- the deep pipeline build_gemm_lds2 can express.
  explicit_emit_style = bool(ov) or any(x in os.environ for x in ("PREFILL_GEMM_DBUF", "PREFILL_GEMM_PLRA", "PREFILL_GEMM_PLRAB"))
  eightwave_env = os.environ.get("PREFILL_GEMM_8WAVE")
  eightwave_on = (eightwave_env not in ("0", "false", "False", "FALSE", "off", "OFF", "no", "NO")) if eightwave_env is not None else not explicit_emit_style
  if eightwave_on and out_f % 128 == 0:
    waves_m, waves_n, wm, wn, dbuf, plra, plrab = 4, 2, 2, 4, 1, 0, 1
  # Structural-emit stress-study overrides (additive, default UNSET -> baseline unchanged): global knobs to sweep the
  # GEMM emit across ALL graph-gemm roles. DepthU=BK, cross-iter prefetch=DBUF, substep pipeline=PLRA/PLRAB,
  # SALU-addr=LEANADDR. Invalid combos (tile/VGPR/LDS overflow) raise in build and are caught per-candidate by the
  # sweep driver (marked FAILED). See docs/prefill-structural-emit-search-*.
  def _envint(nm, dv):
    v = os.environ.get(nm)
    if v is None: return dv
    try: return int(v)
    except ValueError: return dv
  bk    = _envint("PREFILL_GEMM_BK", bk)
  dbuf  = _envint("PREFILL_GEMM_DBUF", dbuf)
  plra  = _envint("PREFILL_GEMM_PLRA", plra)
  plrab = _envint("PREFILL_GEMM_PLRAB", plrab)
  leanaddr = _envint("PREFILL_GEMM_LEANADDR", 0)
  # PROMOTED DEFAULT = the ROLE-SELECTIVE pipe (both flags below default-on). The software-pipelined route is on for the
  # latency-bound roles (attn q/o, attn k/v, ffn-down) and OFF for the already-saturated ffn gate/up (the pipe regressed
  # that one ~17%), so gate/up keeps its faster lds path. Net: beats the all-roles "global pipe" by ~3% and the old lds
  # default by ~12-23% through ctx8192, output-equivalent. Rollback chain: PREFILL_PIPE_ROLE_SELECTIVE=0 -> global pipe
  # (all roles); then PREFILL_GEMM_PIPELINE=0 -> old lds default.
  pipe_mode = bool(_envint("PREFILL_GEMM_PIPELINE", 1))
  pipe_tm = _envint("PREFILL_GEMM_PIPELINE_TM", 2)
  pipe_tn = _envint("PREFILL_GEMM_PIPELINE_TN", 2)
  # default-ON: exclude ffn gate/up (uniquely out_f==12288) from the pipe -> it takes the faster lds path; the rest stay piped.
  if _envint("PREFILL_PIPE_ROLE_SELECTIVE", 1) and out_f == 12288:
    pipe_mode = False
  pad = _envint("PREFILL_GEMM_PAD", pad)
  bm, bn, threads = waves_m * wm * 16, waves_n * wn * 16, waves_m * waves_n * 32
  reloc = os.environ.get("PREFILL_GEMM_RELOC", "1") not in ("0", "false", "False", "FALSE", "off", "OFF", "no", "NO")
  try: reloc_max_wgs = max(1, int(os.environ.get("PREFILL_GEMM_RELOC_MAX_WGS", "1")))
  except ValueError: reloc_max_wgs = 1
  return {"m": m, "n": n, "k": k, "waves_m": waves_m, "waves_n": waves_n, "wm": wm, "wn": wn, "bk": bk, "pad": pad,
          "dbuf": dbuf, "plra": plra, "plrab": plrab, "leanaddr": leanaddr, "pipe_mode": bool(pipe_mode),
          "pipe_tm": pipe_tm, "pipe_tn": pipe_tn, "bm": bm, "bn": bn, "threads": threads,
          "reloc": reloc, "reloc_max_wgs": reloc_max_wgs,
          "role_selective_excluded": bool(_envint("PREFILL_PIPE_ROLE_SELECTIVE", 1) and out_f == 12288)}


def _emit_schedule(p: dict, name: str | None = None):
  # Emit the resolved schedule to (insts, lds_bytes, bm, bn, threads, name). Shared by the legacy _kernel and the
  # spec-driven generated route; `name` lets the generated route carry a distinct program name for route identity.
  m, n, k, bm, bn, bk = p["m"], p["n"], p["k"], p["bm"], p["bn"], p["bk"]
  if m % bm or n % bn or k % bk: return None
  if p["pipe_mode"]:
    # Pipeline route (PROMOTED, default-on @ tm2_tn2): software-pipeline-style compute kernel, not LDS-staged.
    # TIER_A win from higher ILP on the latency-bound sub-BLAS roles (attn_kv/qo, ffn_down). Rollback: PREFILL_GEMM_PIPELINE=0.
    insts = ref.build_gemm_pipe(m, n, k, p["pipe_tm"], p["pipe_tn"])
    lds_bytes = 1
  else:
    insts = ref.build_gemm_lds2(m, n, k, p["waves_m"], p["waves_n"], p["wm"], p["wn"], p["bk"], p["pad"], p["dbuf"],
                                PLRA=p["plra"], PLRAB=p["plrab"], LEANADDR=p["leanaddr"])
    lds_bytes = max((bk * 2 + p["pad"]) * (bm + bn) * (2 if p["dbuf"] else 1), 65536 // 8)
    # Inc-3 waitcnt relocation (additive, default ON): apply ONLY at LOW OCCUPANCY, where LDS-load latency is EXPOSED.
    # The win is overlapping WMMA compute with exposed LDS latency (benefit is proportional to 1/occupancy); at high
    # occupancy it is pure extra-waitcnt overhead and REGRESSES. Causal occupancy sweep (same kv kernel, vary only the
    # LDS allocation): relocation delta = +0.08% @4 WG/CU -> -3.03% @2 WG/CU -> +4.26% @1 WG/CU. So gate on LDS-limited
    # workgroups/CU (estimated from `lds_bytes`), NOT on waves_n. Threshold PREFILL_GEMM_RELOC_MAX_WGS (default 1 = only
    # the lowest-occupancy configs). See docs/prefill-asm-instruction-scheduler-inc3-result-20260623.md.
    if p["reloc"] and max(1, 65536 // lds_bytes) <= p["reloc_max_wgs"]:
      from extra.qk.asm_scheduler import relocate_lgkm_waits
      insts = relocate_lgkm_waits(insts)
  return insts, lds_bytes, bm, bn, p["threads"], name or f"prefill_graph_gemm_{m}_{n}_{k}"


@lru_cache(maxsize=None)
def _kernel(out_f: int, in_f: int):
  # legacy/rollback emit path (PREFILL_GENERATED_SCHEDULE=0). Identical schedule to the generated route.
  return _emit_schedule(_resolve_schedule(out_f, in_f))


def route_pf16_graph_gemm(lin, x: Tensor) -> Tensor | None:
  # NOTE: the gfx1100 arch restriction for default-on lives in model.PREFILL_GRAPH_GEMM (computed once at import);
  # it is NOT checked here because Device[...] access is disallowed during JIT capture (ALLOW_DEVICE_USAGE). The
  # T==512 / tile-divisible / bias / role guards below restrict to the validated dense prefill shapes; everything
  # else silently falls back to the normal PREFILL_V2 matmul.
  roles = str(getenv("PREFILL_GRAPH_GEMM_ROLES", ""))
  if roles:
    role = getattr(lin, "_prefill_graph_role", None)
    if role is None or role not in {r.strip() for r in roles.split(",") if r.strip()}: return None
  w = getattr(lin, "_pf16_w", None)
  b = getattr(lin, "bias", None)
  if w is None or b is not None or x.ndim < 2: return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int): return None
  if x.shape[-2] != 512: return None
  out_f, in_f = w.shape
  if in_f != x.shape[-1]: return None
  # TG-P4: the prefill GEMM schedule is emitted from a data PrefillGEMMScheduleSpec (machine_authored_generated route
  # prefill_pipe_role_selective_generated). Default-on; BoltBeam QK_ROUTE_POLICY can also select it. The generated
  # route resolves the SAME schedule params as the legacy path (byte-identical instruction stream) but carries a
  # distinct program name. Rollback to the legacy emit: PREFILL_GENERATED_SCHEDULE=0.
  role = getattr(lin, "_prefill_graph_role", None)
  from tinygrad.llm.model import _qk_route_policy_selects_prefill_generated, _QK_ROUTE_POLICY_STRICT
  gen_selected = _qk_route_policy_selects_prefill_generated(out_f, in_f)
  gen_on = bool(getenv("PREFILL_GENERATED_SCHEDULE", 1))
  if gen_on:
    from extra.qk.prefill_schedule_spec import describe_prefill_schedule, emit_prefill_gemm_from_spec
    spec = describe_prefill_schedule(out_f, in_f, role=role)
    built = emit_prefill_gemm_from_spec(spec)
  else:
    # strict hidden-fallback guard: a policy-selected prefill tensor must bind to the generated schedule; reaching
    # the legacy emit with the generated route rolled back (PREFILL_GENERATED_SCHEDULE=0) under strict fails loud.
    if gen_selected and _QK_ROUTE_POLICY_STRICT:
      raise ValueError(f"TG_P4_BLOCKED_HIDDEN_FALLBACK: QK_ROUTE_POLICY selects prefill_pipe_role_selective_generated "
                       f"for prefill tensor (out={out_f}, in={in_f}) but PREFILL_GENERATED_SCHEDULE is off -> it fell "
                       f"back to the legacy schedule emit")
    built = _kernel(out_f, in_f)
  if built is None: return None
  insts, lds_bytes, bm, bn, threads, name = built
  a = x.reshape(512, in_f).cast(dtypes.float16).contiguous()
  bt = w.cast(dtypes.float16).contiguous()
  c = Tensor.empty(512, out_f, dtype=dtypes.half, device=x.device).contiguous()
  grid = (out_f // bn, 512 // bm, 1)
  def asm_kernel(A, Bt, C):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(threads, "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=512*out_f*in_f*2, mem=(512*in_f+out_f*in_f+512*out_f)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=i) for i in insts]))))
  out = Tensor.custom_kernel(a, bt, c, fxn=asm_kernel)[2]
  return out.reshape(*x.shape[:-1], out_f)
