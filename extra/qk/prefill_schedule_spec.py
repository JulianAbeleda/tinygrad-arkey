#!/usr/bin/env python3
"""TG-P4: prefill GEMM SCHEDULE spec + spec-driven emission.

The shipped prefill route (extra/qk/prefill_graph_gemm_route.py) selects a role-selective software-pipelined /
LDS-staged WMMA schedule and emits it as an RDNA3 assembly instruction stream (extra/qk/prefill/wmma.py
build_gemm_pipe / build_gemm_lds2). build_gemm_pipe is a PARAMETERIZED schedule generator: it derives the loop
structure, double-buffer VGPR layout and waitcnt placement from the (M,N,K,TM,TN) schedule. TG-P4 makes that
schedule DATA: a PrefillGEMMScheduleSpec captures the resolved tile / wave / pipeline / role-policy parameters, and
emit_prefill_gemm_from_spec lowers the spec through the same generator. The RDNA3 WMMA instruction set is the target
grammar (as wave32 is for the Q4_K G3 route); the SCHEDULE is machine-authored from the spec.

This is a provenance conversion: the generated route resolves the schedule through _resolve_schedule and emits through
the spec. The old fixed call site and PREFILL_GENERATED_SCHEDULE rollback have been removed from runtime.

role_policy: the current default pipes the latency-bound roles (attn q/o, attn k/v, ffn_down) and EXCLUDES the
saturated ffn_gate_up, which keeps its faster LDS path. That fact is carried on the spec as `route_family` ('pipe' vs
'lds') + `protected_roles`, resolved by role first and shape second for legacy callers (no model-name hardcode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

# ffn_gate_up is the protected (pipe-excluded) role. Shape fallback is retained for legacy callers that cannot annotate
# the role yet; runtime schedule selection should call the helper instead of matching this shape literal directly.
PROTECTED_PIPE_ROLES = ("ffn_gate_up",)
_GATE_UP_SHAPES = frozenset(((12288, 4096), (12288, 5120)))
PIPELINE_TARGET_SUBSTRATE = ("tinygrad.schedule.wmma.shaped_wmma", "tinygrad.schedule.rangeify")
HAND_ASM_LDS2_PROFILE = "hand_asm_lds2"
PREFILL_GEMM_PROFILES = ("auto", HAND_ASM_LDS2_PROFILE)


def prefill_pipe_excluded_by_role_shape_policy(out_f: int, in_f: int, *, role: str | None = None) -> bool:
  """Return whether the role/shape should keep the LDS route under role-selective prefill.

  Role is authoritative when present. Shape fallback preserves existing 8B behavior for unannotated legacy call sites and
  keeps the known gate/up profiles protected without exposing model-size literals in runtime schedule selection.
  """
  if role is not None: return role in PROTECTED_PIPE_ROLES and (out_f, in_f) in _GATE_UP_SHAPES
  return (out_f, in_f) in _GATE_UP_SHAPES


def prefill_pipe_role_selective_generated_pure_search_proof() -> dict[str, Any]:
  """Small, non-authoritative proof scaffold for strict pure-machine-search tooling.

  The route is still route-selective from a spec, but the active lowering path still emits a raw
  `UOp(Ops.INS, ...)` program in `route_pf16_graph_gemm`, so it remains externally handwritten despite the
  schedule-data conversion.
  """
  return {
    "route_id": "prefill_pipe_role_selective_generated",
    "status": "blocked_for_strict_pure_search",
    "is_pure": False,
    "blocker": "Ops.INS",
    "executing_surface": {
      "writer": "extra/qk/prefill_graph_gemm_route.py::route_pf16_graph_gemm",
      "lowering_chain": "describe_prefill_schedule -> emit_prefill_gemm_from_spec -> _emit_schedule -> "
                        "build_gemm_pipe / build_gemm_lds2 -> UOp(Ops.INS, ...)",
    },
    "target_lowering_substrate": {
      "goal": "backend-owned matrix instructions via Tinygrad IR",
      "path": ("tinygrad.schedule.wmma", "tinygrad.schedule.shaped_wmma", "tinygrad.schedule.rangeify"),
      "target": PIPELINE_TARGET_SUBSTRATE,
    },
    "notes": (
      "The manifest already calls this route `external_handwritten_kernel`; this helper is intended to keep the blocker explicit "
      "for audits/tests and avoid any false 'pure' interpretation."
    )
  }


@dataclass(frozen=True)
class PrefillGEMMScheduleSpec:
  """Data description of a prefill GEMM schedule. `route_family` selects the lowering (software-pipeline vs LDS-
  staged); the tile/wave/pipeline fields are the resolved schedule parameters that emit_prefill_gemm_from_spec
  hands to the RDNA3 WMMA generator. Serializable via to_json()."""
  m: int
  n: int
  k: int
  route_family: str                 # "pipe" | "lds" | "global_register_resident"
  tile_m: int                       # bm = waves_m*wm*16
  tile_n: int                       # bn = waves_n*wn*16
  tile_k: int                       # bk (DepthU)
  waves_m: int
  waves_n: int
  wm: int
  wn: int
  pipe_tm: int
  pipe_tn: int
  pipeline_depth: int               # double-buffer depth (2 for the pipe route)
  threads: int
  dbuf: int = 1
  plra: int = 0
  plrab: int = 0
  pad: int = 16
  leanaddr: int = 0
  reloc: bool = True
  reloc_max_wgs: int = 1
  role: str = ""
  protected_roles: tuple[str, ...] = PROTECTED_PIPE_ROLES
  waitcnt_policy: str = "targeted_vmcnt"
  target: str = "amd_gfx1100"

  @property
  def kernel_name(self) -> str:
    return f"prefill_gen_sched_gemm_{self.m}_{self.n}_{self.k}"

  @property
  def pipeline_policy(self):
    """Return the common policy using the route's authoritative storage contract.

    LDS footprint is owned by :class:`WMMALDSSpec`: its padded A/B strides and
    DBUF count are the layout consumed by the LDS lowering. Keeping that
    derivation in one place avoids schedule-level byte estimates drifting from
    the emitted layout.
    """
    from tinygrad.codegen.opt.compiler_policies import pipeline_policy_for_route
    if self.route_family == "lds":
      from extra.qk.wmma_lds_spec import extract_wmma_lds_spec
      lds_spec = extract_wmma_lds_spec(self)
      if lds_spec is None:
        raise ValueError("cannot build pipeline policy for an illegal LDS schedule")
      return pipeline_policy_for_route("lds", buffer_count=lds_spec.lds_buffers,
                                       slot_bytes=lds_spec.lds_buffer_bytes, stages=self.pipeline_depth)
    if self.route_family == "global_register_resident":
      from tinygrad.codegen.opt.compiler_policies import RegisterPipePlan
      return RegisterPipePlan(stages=self.pipeline_depth).policy
    return pipeline_policy_for_route(self.route_family, stages=self.pipeline_depth)

  def to_json(self) -> dict[str, Any]:
    return {"m": self.m, "n": self.n, "k": self.k, "route_family": self.route_family, "tile_m": self.tile_m,
            "tile_n": self.tile_n, "tile_k": self.tile_k, "waves_m": self.waves_m, "waves_n": self.waves_n,
            "wm": self.wm, "wn": self.wn, "pipe_tm": self.pipe_tm, "pipe_tn": self.pipe_tn,
            "pipeline_depth": self.pipeline_depth, "threads": self.threads, "dbuf": self.dbuf, "plra": self.plra,
            "plrab": self.plrab, "pad": self.pad, "leanaddr": self.leanaddr, "role": self.role,
            "protected_roles": list(self.protected_roles), "waitcnt_policy": self.waitcnt_policy,
            "target": self.target, "kernel_name": self.kernel_name}


def _params_to_spec(p: dict, role: str | None) -> PrefillGEMMScheduleSpec:
  return PrefillGEMMScheduleSpec(
    m=p["m"], n=p["n"], k=p["k"], route_family=("pipe" if p["pipe_mode"] else "lds"),
    tile_m=p["bm"], tile_n=p["bn"], tile_k=p["bk"], waves_m=p["waves_m"], waves_n=p["waves_n"], wm=p["wm"],
    wn=p["wn"], pipe_tm=p["pipe_tm"], pipe_tn=p["pipe_tn"], pipeline_depth=2 if p["pipe_mode"] else (2 if p["dbuf"] else 1),
    threads=p["threads"], dbuf=p["dbuf"], plra=p["plra"], plrab=p["plrab"], pad=p["pad"], leanaddr=p["leanaddr"],
    reloc=p["reloc"], reloc_max_wgs=p["reloc_max_wgs"], role=role or "")


def describe_prefill_schedule(out_f: int, in_f: int, *, role: str | None = None) -> PrefillGEMMScheduleSpec:
  """Resolve the current default schedule for a prefill (out_f, in_f) GEMM into a PrefillGEMMScheduleSpec. Uses the
  graph-GEMM resolver, so the spec is a faithful snapshot of the resolved schedule."""
  profile = os.environ.get("PREFILL_GEMM_PROFILE", "auto").strip().lower()
  if profile not in PREFILL_GEMM_PROFILES:
    raise ValueError(f"PREFILL_GEMM_PROFILE must be one of {', '.join(PREFILL_GEMM_PROFILES)}, got {profile!r}")
  if profile == HAND_ASM_LDS2_PROFILE:
    if any(os.environ.get(key, "0").strip().lower() not in ("", "0", "false", "off", "no")
           for key in ("PREFILL_WMMA_PIPE_PRIMITIVE", "PREFILL_WMMA_LDS_PRIMITIVE")):
      raise ValueError(f"{HAND_ASM_LDS2_PROFILE} is the raw hand-ASM oracle and cannot be combined with generated primitive flags")
    # Frozen 2026-06-20 all-LDS2 schedule. This exact profile emits the validated 751-instruction attn_qo stream
    # (2x2 waves, 4x4 WMMA tiles, BK32, PAD16, single-buffer PLRA) and intentionally bypasses later pipe/DBUF defaults.
    return PrefillGEMMScheduleSpec(
      m=512, n=out_f, k=in_f, route_family="lds", tile_m=128, tile_n=128, tile_k=32,
      waves_m=2, waves_n=2, wm=4, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=1, threads=128,
      dbuf=0, plra=1, plrab=0, pad=16, leanaddr=0, reloc=False, reloc_max_wgs=1, role=role or "")
  from extra.qk.prefill_graph_gemm_route import _resolve_schedule
  return _params_to_spec(_resolve_schedule(out_f, in_f, role), role)


def _spec_to_params(spec: PrefillGEMMScheduleSpec) -> dict:
  return {"m": spec.m, "n": spec.n, "k": spec.k, "waves_m": spec.waves_m, "waves_n": spec.waves_n, "wm": spec.wm,
          "wn": spec.wn, "bk": spec.tile_k, "pad": spec.pad, "dbuf": spec.dbuf, "plra": spec.plra, "plrab": spec.plrab,
          "leanaddr": spec.leanaddr, "pipe_mode": spec.route_family == "pipe", "pipe_tm": spec.pipe_tm,
          "pipe_tn": spec.pipe_tn, "bm": spec.tile_m, "bn": spec.tile_n, "threads": spec.threads,
          "reloc": spec.reloc, "reloc_max_wgs": spec.reloc_max_wgs}


def emit_prefill_gemm_from_spec(spec: PrefillGEMMScheduleSpec):
  """Lower a PrefillGEMMScheduleSpec to (insts, lds_bytes, bm, bn, threads, name)."""
  if os.environ.get("PREFILL_WMMA_PIPE_PRIMITIVE") == "1" and spec.route_family == "pipe":
    from extra.qk import wmma_pipe_spec
    pipe_spec = wmma_pipe_spec.extract_wmma_pipe_spec(spec)
    if pipe_spec is not None: return wmma_pipe_spec.lower_wmma_pipe_spec(pipe_spec)
  if os.environ.get("PREFILL_WMMA_LDS_PRIMITIVE") == "1" and spec.route_family == "lds":
    from extra.qk import wmma_lds_spec
    lds_spec = wmma_lds_spec.extract_wmma_lds_spec(spec)
    if lds_spec is not None: return wmma_lds_spec.lower_wmma_lds_spec(lds_spec)
  if spec.route_family == "global_register_resident":
    raise ValueError("register-resident schedules lower only through the compiler matmul route")
  from extra.qk.prefill_graph_gemm_route import _emit_schedule
  return _emit_schedule(_spec_to_params(spec), name=spec.kernel_name)


def register_resident_postrange_opts(spec: PrefillGEMMScheduleSpec):
  """Select TC geometry without importing an LDS layout or LDS proof object."""
  if spec.route_family != "global_register_resident": raise ValueError("register opts require a register-resident schedule")
  from tinygrad.codegen.opt import Opt, OptOps
  return (Opt(OptOps.TC, 0, (-1, 2, 1)),)
