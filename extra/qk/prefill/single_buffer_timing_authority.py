#!/usr/bin/env python3
"""Kernel-only timing authority for the proven exact single-buffer candidate."""
from __future__ import annotations

import math, statistics
from contextlib import nullcontext
from typing import Any, Callable

from extra.qk.runtime_specs import GFX1100_SINGLE_BUFFER_CAPABILITY
from extra.qk.timing_harness import pinned_peak_from_env
from extra.qk.mmq_amd_telemetry import collect_telemetry

SCHEMA = "prefill-single-buffer-kernel-timing.v1"
M, N, K = 512, 12288, 4096


def _validate_execution(execution:dict[str, Any]) -> tuple[str, str, str]:
  if not execution.get("passed") or not execution.get("structural_binding", {}).get("pre_gpu_eligible"):
    raise ValueError("timing requires a passing structurally proven execution authority")
  identity = execution.get("canonical_identity")
  if not isinstance(identity,str) or len(identity)!=64 or any(c not in "0123456789abcdef" for c in identity):
    raise ValueError("execution candidate identity is not canonical SHA-256")
  if execution.get("capability_id") != GFX1100_SINGLE_BUFFER_CAPABILITY.capability_id:
    raise ValueError("execution capability join is missing or unsupported")
  program_hash = execution.get("program", {}).get("binary_sha256")
  runtime_hash = execution.get("runtime", {}).get("executed_binary_sha256")
  if not isinstance(program_hash, str) or runtime_hash != program_hash: raise ValueError("execution binary join is missing or unequal")
  git = execution.get("environment", {}).get("git", {})
  if not isinstance(git.get("revision"), str) or git.get("dirty") is not False:
    raise ValueError("timing requires a clean execution-authority commit join")
  return identity, program_hash, git["revision"]


def run_kernel_timing(execution:dict[str, Any], kernel_call:Callable[..., float], *, warmups:int=5, rounds:int=21,
                      telemetry:Callable[..., dict[str, Any]]=collect_telemetry,
                      clock_context:Callable[[], Any]=pinned_peak_from_env) -> dict[str, Any]:
  """Time only a prepared runtime kernel call; preparation/compile/I/O must occur before entry."""
  identity, binary_hash, revision = _validate_execution(execution)
  if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 1: raise ValueError("warmups must be positive")
  if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 3: raise ValueError("rounds must be >= 3")
  before = telemetry("single_buffer_kernel_before", samples=1)
  with clock_context() if clock_context is not None else nullcontext(None) as clock_pin:
    for _ in range(warmups):
      elapsed = kernel_call(wait=True)
      if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("warmup kernel call did not return positive finite device time")
    samples_s = []
    for _ in range(rounds):
      elapsed = kernel_call(wait=True)
      if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("timed kernel call did not return positive finite device time")
      samples_s.append(float(elapsed))
  after = telemetry("single_buffer_kernel_after", samples=1)
  median_s, min_s = statistics.median(samples_s), min(samples_s)
  ops = 2*M*N*K
  return {"schema": SCHEMA, "passed": True, "canonical_identity": identity, "binary_sha256": binary_hash,
          "git_revision": revision, "joins": {"candidate": True, "binary": True, "commit": True},
          "protocol": {"scope": "kernel_only", "wait": True, "compile_excluded": True, "input_setup_excluded": True,
                       "output_transfer_excluded": True, "warmups": warmups, "rounds": rounds},
          "samples_ms": [x*1e3 for x in samples_s], "median_ms": median_s*1e3, "min_ms": min_s*1e3,
          "median_tflops": ops/median_s/1e12, "max_tflops": ops/min_s/1e12,
          "clock_pin": clock_pin, "telemetry": {"before": before, "after": after}}


def run_prepared_candidate_timing(payload:dict[str, Any], candidate_hash:str, *, case:str="constant", warmups:int=5,
                                  rounds:int=21, telemetry:Callable[...,dict[str,Any]]=collect_telemetry,
                                  clock_context:Callable[[],Any]=pinned_peak_from_env) -> dict[str, Any]:
  """Run correctness once, then time the exact same prepared candidate CALL."""
  from extra.qk.prefill.single_buffer_execution_authority import run
  prepared = []
  execution = run(payload, candidate_hash, case=case, prepared_out=prepared)
  if len(prepared) != 1: raise RuntimeError("execution authority did not return one prepared candidate context")
  report = run_kernel_timing(execution, prepared[0].kernel_call, warmups=warmups, rounds=rounds,
                             telemetry=telemetry, clock_context=clock_context)
  report["execution_authority"] = {"schema": execution["schema"], "passed": execution["passed"],
    "canonical_identity": execution["canonical_identity"], "program": execution["program"],
    "runtime": execution["runtime"], "environment": execution["environment"]}
  return report
