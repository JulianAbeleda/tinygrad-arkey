"""C5 progressive candidate correctness for the attn_qo direct-L2 vs LDS arc.

This is the CODE for the "small -> exact" progressive correctness ladder.  It
owns NO dangerous behaviour of its own: every GPU dispatch flows through the one
process boundary in
:func:`extra.qk.prefill.isolated_guarded_executor.run_isolated_guarded_execution`
(SPAWN, guarded, health-checked, hard-timeout).  The parent constructs no live
runtime; the per-stage runtime is RECONSTRUCTED inside the spawned child by the
module-level :func:`build_attn_qo_stage_bundle` from picklable descriptors.

What it proves (once per candidate, BEFORE any timing -- there is NO speed claim
in C5):

  * a DETERMINISTIC CPU reference for the exact attn_qo output on seeded,
    NONCONSTANT fp16 inputs (:func:`attn_qo_reference_inputs`).  The compiled
    kernels compute ``C = A @ B^T`` with fp16 multiplicands and an fp32
    accumulator (see the pair generator's ``dtypes``/``layout`` and the
    ``a @ b.transpose()`` / custom-kernel lowering in
    ``attn_qo_executable_preparation.py``), so the reference is the fp32
    accumulation of the same fp16 inputs, compared with the guarded lifecycle's
    rtol/atol against the fp16 device output;
  * a progressive stage ladder small -> exact (:func:`default_stage_ladder`).
    Each stage compiles THAT candidate at THAT stage shape via a module-level
    build fn (picklable args), runs it through the isolated guarded executor in a
    FRESH child, reads the full output, compares it to the CPU reference, and
    STOPS on the first fault -- larger stages are not attempted for that
    candidate;
  * identity-join (P0-3): every stage result is bound to the DISTINCT canonical
    candidate identity for that transport/shape, and the final record binds the
    exact candidate's canonical identity.  ``direct_l2`` and ``lds`` never share
    a candidate identity.

Admission constraint (honest limitation, see
``extra/qk/runtime_specs.py`` ``admit_full_kernel_candidate``): the
register-resident / ``direct_l2`` template is admitted ONLY at exactly
``(512, 4096, 4096)``.  The existing compile path therefore cannot produce a
smaller ``direct_l2`` artifact, so the ``direct_l2`` ladder is the single exact
stage.  The ``lds`` ladder shrinks only M (keeping N=K=4096) so every stage
reuses the identical resolved schedule/tile as the exact shape (grid divides by
construction).  The ladder is data-driven, so a future proved-smaller template
extends it without code change.

``execute=False`` (default) prints the bounded stage plan (shapes, buffer bytes,
per-stage kernel, argument order, candidate identity) and dispatches NOTHING.
``execute=True`` (the LEAD invokes this) runs the ladder through the isolated
executor.  Any timeout / device loss / guard corruption / numerical fault STOPS
the ladder -- there is no reset, retry, or larger-stage continuation.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill.isolated_guarded_executor import (ExecutableBundle, ExecutionRequest, IsolatedExecutionResult,
                                                        build_tinygrad_bundle, make_tinygrad_bundle_builder,
                                                        run_isolated_guarded_execution)
from tinygrad.runtime.execution_bridge_contracts import dispatch_state

SCHEMA = "attn-qo-progressive-correctness.v1"
EXACT_SHAPE = (512, 4096, 4096)
RUNTIME_DEVICE = "AMD"
# Seeded nonconstant inputs live in a small symmetric range so the fp16 dot over
# K terms stays well inside fp16 precision (cf. the anchor's 1/64-scaled cases).
INPUT_SCALE = 0.0625  # 1/16
DEFAULT_SEED = 0x5150

# Per-transport dispatch ABI (kernel argument order).  BOTH transports now
# compile the compiler-rendered ``a @ b^T`` matmul (direct_l2 register-resident,
# lds the proven WMMA-LDS kernel), so both emit the destination global first,
# then a=(M,K), then b=(N,K).  The lds order is derived from the compiled
# PROGRAM's globals (see ``_derive_lds_argument_order``).  ``output`` is the
# auto-allocated destination in the guarded lifecycle.
DIRECT_ARGUMENT_ORDER = ("output", "a", "b")
LDS_ARGUMENT_ORDER = ("output", "a", "b")
_ARGUMENT_ORDERS = {"direct_l2": DIRECT_ARGUMENT_ORDER, "lds": LDS_ARGUMENT_ORDER}


def argument_order_for(transport: str) -> tuple[str, ...]:
  order = _ARGUMENT_ORDERS.get(transport)
  if order is None: raise ValueError(f"unsupported attn_qo transport: {transport!r}")
  return order


# --- Deterministic CPU reference ---------------------------------------------

def attn_qo_reference_inputs(shape: tuple[int, int, int], *, seed: int = DEFAULT_SEED,
                             scale: float = INPUT_SCALE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Deterministic seeded fp16 inputs + the fp32 full-output reference.

  ``a`` is ``(M, K)`` row-major, ``b`` is ``(N, K)`` (transposed-row-major, i.e.
  ``B^T`` operand), and the reference ``c = A_f32 @ B_f32^T`` is ``(M, N)``.  This
  is exactly what both compiled candidates compute (``a @ b.transpose()`` / the
  LDS2 ``a @ b^T`` GEMM) with an fp32 accumulator over fp16 multiplicands; the
  fp16 device output is compared to this fp32 reference within rtol/atol.  The
  seed is mixed with the shape so distinct stages get distinct, reproducible,
  NONCONSTANT inputs.
  """
  m, n, k = shape
  rng = np.random.default_rng((seed & 0xFFFFFFFF) ^ (m * 73856093) ^ (n * 19349663) ^ (k * 83492791))
  a = rng.uniform(-scale, scale, size=(m, k)).astype(np.float16)
  b = rng.uniform(-scale, scale, size=(n, k)).astype(np.float16)
  reference = (a.astype(np.float32) @ b.astype(np.float32).T).astype(np.float32)
  return a, b, reference


# --- Stage ladder ------------------------------------------------------------

def default_stage_ladder(candidate: str) -> tuple[tuple[int, int, int], ...]:
  """The per-candidate small -> exact ladder, always ending at the exact shape.

  ``direct_l2`` is admission-locked to the exact shape (only that register
  template is proved), so its ladder is the single exact stage.  ``lds`` shrinks
  only M (N=K=4096 fixed) so every stage reuses the identical resolved schedule
  and tile; the grid divides by construction.
  """
  if candidate == "direct_l2":
    return (EXACT_SHAPE,)
  if candidate == "lds":
    return ((128, 4096, 4096), (256, 4096, 4096), EXACT_SHAPE)
  raise ValueError(f"unsupported attn_qo transport: {candidate!r}")


# --- Per-stage canonical identity (CPU-only, no compile) ---------------------

def stage_candidate_identity(candidate: str, shape: tuple[int, int, int]) -> str:
  """The DISTINCT canonical candidate identity for (transport, shape).

  Computed purely CPU-side from the pair generator's payload schema, so the
  identity-join is available without compiling or dispatching.
  """
  from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import _identity, _payload
  payload = copy.deepcopy(_payload(candidate))
  payload["workload"]["shape"] = {"m": shape[0], "n": shape[1], "k": shape[2]}
  return _identity(payload)


# --- Shape-parametric compile (REUSES the existing compile adapters) ----------

def compile_attn_qo_stage(*, transport: str, shape: tuple[int, int, int], target: str | None = None) -> dict[str, Any]:
  """Compile ONE candidate at a stage shape through the existing adapter table.

  For the exact shape this is identical to
  ``attn_qo_executable_preparation.compile_attn_qo_program`` (same payload,
  identity, and adapter).  For a smaller shape it builds a stage payload and runs
  the SAME transport adapter, so no kernel is reinvented.  A ``direct_l2`` stage
  other than the exact shape is rejected by ``admit_full_kernel_candidate`` (the
  register template is only proved at 512x4096x4096) -- that surfaces as a
  controlled compile failure and, under ``execute=True``, a fault-stop.

  RUNS compile-only (no dispatch).  Invoked from inside the spawned child by
  :func:`build_attn_qo_stage_bundle`; also usable by the LEAD for a compile-only
  audit of a stage.
  """
  from extra.qk.prefill.attn_qo_executable_preparation import (_COMPILE_ADAPTERS, _dev_target,
                                                               _isolated_compile_environment, _workload_axes)
  from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import _identity, _payload, generate_pair
  from extra.qk.runtime_specs import admit_full_kernel_candidate_set, full_kernel_candidate_set_from_legacy

  adapter = _COMPILE_ADAPTERS.get(transport)
  if adapter is None:
    raise ValueError(f"unsupported attn_qo transport: {transport!r}; registered: {tuple(sorted(_COMPILE_ADAPTERS))}")
  payload = copy.deepcopy(_payload(transport))
  payload["workload"]["shape"] = {"m": shape[0], "n": shape[1], "k": shape[2]}
  identity = _identity(payload)
  candidate_row = {"payload": payload, "canonical_identity": identity, "storage": transport}
  workload = payload["workload"]
  profile, role, shp, target_dict = _workload_axes(workload)
  dev_target = _dev_target(target, target_dict)
  # Two-buffer LDS admits GFX1100_TWO_BUFFER_STAGE1_CAPABILITY only via the set path.
  admission = admit_full_kernel_candidate_set(full_kernel_candidate_set_from_legacy(payload, identity)).admissions[0]
  with _isolated_compile_environment():
    program, evidence = adapter(candidate_row, admission, workload, dev_target)
  pair = generate_pair()
  return {"schema": "attn_qo.stage_preparation.v1", "transport": transport, "shape": tuple(shp),
          "pair_key": pair["pair_key"], "schedule_digest": pair["schedule_digest"],
          "canonical_identity": identity, "program": program, "compile_evidence": evidence,
          "dispatch_performed": False}


# --- Module-level, picklable, child-only build fn -----------------------------

def _always_alive() -> bool:
  """Module-level (picklable) in-child health hook for the compiled bundle."""
  return True


def build_attn_qo_stage_bundle(*, transport: str, m: int, n: int, k: int, compile_target: str | None,
                               runtime_device: str, argument_order: Sequence[str]) -> ExecutableBundle:
  """Recompile the stage PROGRAM and construct its runtime FRESH, IN THE CHILD.

  RUNS IN THE SPAWNED CHILD only.  A UOp/runtime is not picklable, so the parent
  hands only picklable descriptors (transport/shape/device/argument order); this
  re-runs :func:`compile_attn_qo_stage` and
  :func:`isolated_guarded_executor.build_tinygrad_bundle` in-process so the child
  owns a freshly-compiled program and a freshly-initialized device.
  """
  prepared = compile_attn_qo_stage(transport=transport, shape=(m, n, k), target=compile_target)
  return build_tinygrad_bundle(program=prepared["program"], compile_evidence=prepared["compile_evidence"],
                               device=runtime_device, argument_order=tuple(argument_order), health=_always_alive)


def _stage_builder(candidate: str, shape: tuple[int, int, int], *, compile_target: str | None,
                   runtime_device: str) -> Any:
  """A PICKLABLE builder spec for one stage (module-level build fn + scalars)."""
  return make_tinygrad_bundle_builder(build=build_attn_qo_stage_bundle, transport=candidate, m=shape[0],
                                      n=shape[1], k=shape[2], compile_target=compile_target,
                                      runtime_device=runtime_device, argument_order=argument_order_for(candidate))


# --- Typed states ------------------------------------------------------------

@dataclass(frozen=True)
class StagePlan:
  """The bounded, dispatch-free description of one ladder stage."""
  candidate: str
  shape: tuple[int, int, int]
  canonical_identity: str
  argument_order: tuple[str, ...]
  buffers_bytes: Mapping[str, int]
  flops: int

  def to_dict(self) -> dict[str, Any]:
    return {"candidate": self.candidate, "shape": list(self.shape), "canonical_identity": self.canonical_identity,
            "argument_order": list(self.argument_order), "buffers_bytes": dict(self.buffers_bytes),
            "flops": self.flops}


@dataclass(frozen=True)
class StageResult:
  """One truthful terminal outcome for a single stage (never auto-retried)."""
  plan: StagePlan
  dispatch_state: str
  passed: bool
  executed: bool
  result: Mapping[str, Any] | None = None
  errors: tuple[str, ...] = ()

  def to_dict(self) -> dict[str, Any]:
    return {"plan": self.plan.to_dict(), "dispatch_state": self.dispatch_state, "passed": self.passed,
            "executed": self.executed, "result": dict(self.result) if self.result is not None else None,
            "errors": list(self.errors)}


@dataclass(frozen=True)
class ProgressiveCorrectnessRecord:
  """The identity-joined, per-candidate progressive-correctness record.

  ``passed`` requires that EVERY planned stage ran and passed AND the ladder
  reached the exact shape.  No speed/timing claim is made anywhere.
  """
  candidate: str
  mode: str  # "audit" | "execute"
  canonical_identity: str  # the DISTINCT exact-shape candidate identity (P0-3 join)
  stages: tuple[StageResult, ...]
  passed: bool
  reached_exact: bool
  stopped_at: tuple[int, int, int] | None = None
  errors: tuple[str, ...] = ()
  schema: str = SCHEMA

  def to_dict(self) -> dict[str, Any]:
    return {"schema": self.schema, "candidate": self.candidate, "mode": self.mode,
            "canonical_identity": self.canonical_identity, "reached_exact": self.reached_exact,
            "stopped_at": list(self.stopped_at) if self.stopped_at is not None else None,
            "passed": self.passed, "stages": [s.to_dict() for s in self.stages], "errors": list(self.errors)}


def _stage_plan(candidate: str, shape: tuple[int, int, int]) -> StagePlan:
  m, n, k = shape
  return StagePlan(candidate=candidate, shape=shape, canonical_identity=stage_candidate_identity(candidate, shape),
                   argument_order=argument_order_for(candidate),
                   buffers_bytes={"a": m * k * 2, "b": n * k * 2, "output": m * n * 2},  # fp16 = 2 bytes
                   flops=m * n * k * 2)


def _format_plan(candidate: str, plans: Sequence[StagePlan], policy: GuardPolicy) -> str:
  lines = [f"attn_qo progressive correctness -- bounded stage plan for {candidate!r} "
           "(AUDIT ONLY, NO DISPATCH):",
           f"  ladder: {len(plans)} stage(s), small -> exact, STOP on first fault",
           f"  guard bytes: +{policy.prefix_bytes}/{policy.suffix_bytes}; tolerances rtol={policy.rtol} atol={policy.atol}"]
  for i, plan in enumerate(plans):
    lines.append(f"  [{i}] shape={plan.shape} flops={plan.flops} argument_order={list(plan.argument_order)}")
    lines.append(f"      buffers_bytes={dict(plan.buffers_bytes)} identity={plan.canonical_identity[:16]}...")
  lines.append("  dispatch: none (compile/audit-only)")
  return "\n".join(lines)


# --- The progressive correctness harness -------------------------------------

def run_progressive_correctness(*, candidate: str, execute: bool = False,
                                stages: Sequence[tuple[int, int, int]] | None = None,
                                seed: int = DEFAULT_SEED, compile_target: str | None = None,
                                runtime_device: str = RUNTIME_DEVICE, policy: GuardPolicy | None = None,
                                builder_factory: Callable[[str, tuple[int, int, int]], Any] | None = None,
                                runner: Callable[..., IsolatedExecutionResult] | None = None,
                                health_probe: Callable[[], bool] | None = None,
                                timeout_seconds: float | None = None, terminate_grace_seconds: float = 0.25,
                                health_timeout_seconds: float = 30.0,
                                persist: Callable[[dict[str, Any]], None] | None = None,
                                printer: Callable[[str], None] | None = print) -> ProgressiveCorrectnessRecord:
  """Run the progressive attn_qo correctness ladder for ONE candidate.

  ``execute=False`` (default): print the bounded stage plan (shapes, buffer
  bytes, per-stage argument order + candidate identity) and return a
  ``not_attempted`` record.  NOTHING dispatches and no runtime is constructed.

  ``execute=True`` (the LEAD invokes this): for each stage in order, compile the
  candidate at that shape via a picklable module-level build fn, run it through
  :func:`run_isolated_guarded_execution` (SPAWN, guarded, health-checked) in a
  FRESH child, read the full output, compare it to the deterministic CPU
  reference, and STOP on the first fault (no larger stage is attempted).  Exact
  correctness is thus established once, before any timing; there is NO speed
  claim here.

  ``builder_factory``/``runner``/``health_probe`` are injectable seams so the
  fake-runtime tests drive every path without a real GPU.
  """
  if candidate not in _ARGUMENT_ORDERS: raise ValueError(f"unsupported attn_qo transport: {candidate!r}")
  policy = policy or GuardPolicy()
  ladder = tuple(tuple(s) for s in (stages if stages is not None else default_stage_ladder(candidate)))
  if not ladder: raise ValueError("stage ladder must be non-empty")
  if ladder[-1] != EXACT_SHAPE: raise ValueError(f"stage ladder must end at the exact shape {EXACT_SHAPE}")
  plans = [_stage_plan(candidate, shape) for shape in ladder]
  exact_identity = stage_candidate_identity(candidate, EXACT_SHAPE)

  if not execute:
    if printer is not None: printer(_format_plan(candidate, plans, policy))
    audit_stages = tuple(StageResult(plan=p, dispatch_state=dispatch_state("not_attempted"), passed=False,
                                     executed=False) for p in plans)
    return ProgressiveCorrectnessRecord(candidate=candidate, mode="audit", canonical_identity=exact_identity,
                                        stages=audit_stages, passed=False, reached_exact=False)

  runner = runner or run_isolated_guarded_execution
  factory = builder_factory or (lambda c, shp: _stage_builder(c, shp, compile_target=compile_target,
                                                              runtime_device=runtime_device))
  results: list[StageResult] = []
  stopped_at: tuple[int, int, int] | None = None
  reached_exact = False
  for plan in plans:
    a, b, reference = attn_qo_reference_inputs(plan.shape, seed=seed)
    identity = {"arc": SCHEMA, "candidate": candidate, "shape": list(plan.shape),
                "canonical_identity": plan.canonical_identity, "argument_order": list(plan.argument_order)}
    request = ExecutionRequest(inputs={"a": a, "b": b}, reference=reference, policy=policy,
                               identity=identity, output_dtype=np.float16)
    out = runner(builder=factory(candidate, plan.shape), request=request, health_probe=health_probe,
                 timeout_seconds=timeout_seconds if timeout_seconds is not None else policy.timeout_seconds,
                 terminate_grace_seconds=terminate_grace_seconds, health_timeout_seconds=health_timeout_seconds,
                 persist=persist)
    # Identity-join (P0-3): the returned identity must carry THIS stage's DISTINCT
    # candidate identity; a mismatch fails the stage closed rather than silently.
    join_ok = dict(out.identity).get("canonical_identity") == plan.canonical_identity
    errors = tuple(out.errors) + (() if join_ok else ("stage identity join failed: returned identity "
                                                      "does not match the candidate canonical identity",))
    passed = bool(out.passed and join_ok)
    results.append(StageResult(plan=plan, dispatch_state=out.dispatch_state, passed=passed, executed=True,
                               result=out.to_dict(), errors=errors))
    if not passed:
      stopped_at = plan.shape
      break
    if plan.shape == EXACT_SHAPE: reached_exact = True

  passed = bool(reached_exact and all(r.passed for r in results) and len(results) == len(plans))
  record = ProgressiveCorrectnessRecord(candidate=candidate, mode="execute", canonical_identity=exact_identity,
                                        stages=tuple(results), passed=passed, reached_exact=reached_exact,
                                        stopped_at=stopped_at)
  if not passed and persist is not None: persist(record.to_dict())
  return record


__all__ = ["SCHEMA", "EXACT_SHAPE", "RUNTIME_DEVICE", "DIRECT_ARGUMENT_ORDER", "LDS_ARGUMENT_ORDER",
           "argument_order_for", "attn_qo_reference_inputs", "default_stage_ladder", "stage_candidate_identity",
           "compile_attn_qo_stage", "build_attn_qo_stage_bundle", "StagePlan", "StageResult",
           "ProgressiveCorrectnessRecord", "run_progressive_correctness"]


def main() -> int:
  parser = argparse.ArgumentParser(description="Guarded attn_qo candidate correctness")
  parser.add_argument("candidate", choices=tuple(_ARGUMENT_ORDERS))
  parser.add_argument("--execute", action="store_true", help="dispatch through the isolated guarded executor")
  parser.add_argument("--timeout-seconds", type=float, default=None)
  args = parser.parse_args()
  record = run_progressive_correctness(candidate=args.candidate, execute=args.execute,
                                       timeout_seconds=args.timeout_seconds, printer=None)
  print(json.dumps(record.to_dict(), indent=2))
  return 0 if (record.passed if args.execute else True) else 1


if __name__ == "__main__": raise SystemExit(main())
