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
  route_family: str                 # "pipe" (build_gemm_pipe) | "lds" (build_gemm_lds2)
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
  from extra.qk.prefill_graph_gemm_route import _emit_schedule
  return _emit_schedule(_spec_to_params(spec), name=spec.kernel_name)
