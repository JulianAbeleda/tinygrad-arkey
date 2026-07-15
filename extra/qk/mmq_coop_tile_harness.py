"""Isolated evidence harness for the bounded AMD Q4_K x Q8_1 coop tile.

This is deliberately a validation tool, not a route selector.  The candidate
and optional comparator are constructed and called in the spawned child; the
parent only receives typed evidence from the guarded executor.
"""
from __future__ import annotations

from typing import Any
import hashlib
import numpy as np

from extra.qk.mmq_q4k_q8_atom import (
  run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile, run_q4k_q8_1_mmq_tile_amd,
  q8_1_mmq_ds4_from_row_major,
)
from extra.qk.mmq_q4k_q8_reference import (q4k_q8_1_mmq_ds4_tile_reference, describe_q4k_q8_1_mmq_tile,
                                           Q8_1_MMQ_DS4_LAYOUT)
from extra.qk.mmq_owner_coverage import structural_static_store_only_owner_map
from extra.qk.prefill.guarded_execution import GuardPolicy, make_tinygrad_executable_hooks
from extra.qk.prefill.isolated_guarded_executor import (
  ExecutableBundle, ExecutionRequest, build_tinygrad_bundle,
  make_tinygrad_bundle_builder,
  run_isolated_guarded_execution,
)
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec
from tinygrad import Tensor


def _health() -> bool:
  try: return bool(np.isfinite(Tensor([1.0], device="AMD").realize().numpy()).all())
  except BaseException: return False


def _build_emitted_amd_bundle(*, mode: str = "candidate", device: str = "AMD") -> ExecutableBundle:
  """Compile and bind the candidate inside the spawned child.

  This is intentionally separate from ``_reference_output``: a CPU result can
  never satisfy the emitted-kernel provenance required by this harness.
  """
  from extra.qk.mmq_compile_evidence import compile_mmq_program
  from extra.qk.mmq_experiment import canonical_candidate
  program = compile_mmq_program(canonical_candidate("gated_matrix_v0"), device)
  binary = next(u.arg for u in program.src if getattr(u.op, "name", u.op) == "BINARY")
  evidence = {"passed": True, "binary_sha256": hashlib.sha256(binary).hexdigest()}
  return build_tinygrad_bundle(program=program, compile_evidence=evidence, device=device,
                               argument_order=("output", "q4k", "values", "scales", "sums"), health=_health)


def validate_bounded_coop_tile(raw: np.ndarray, ds4: Any, *, timeout_seconds: float = 30.0,
                               compare_direct: bool = True) -> dict[str, Any]:
  """Return fail-closed evidence for one 16x16x256 tile."""
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=16, n=16, k=256, m_tile=16, n_tile=16,
                                     k_groups=8, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  if (spec.m, spec.k, raw.shape) != (16, 256, (16, 1, 144)):
    raise ValueError("harness is bounded to M=N=16, K=256, one Q4_K block")
  # Canonical CPU authority is recorded explicitly and is never called a GPU result.
  reference = q4k_q8_1_mmq_ds4_tile_reference(raw, ds4, spec).astype(np.float32)
  owners = structural_static_store_only_owner_map(spec)
  points = {(x.m, x.n) for x in owners}
  coverage = {"events": len(owners), "unique": len(points), "expected": 256,
              "complete": len(owners) == len(points) == 256 and points == {(m, n) for m in range(16) for n in range(16)}}
  request = ExecutionRequest(inputs={"q4k": np.asarray(raw), "values": np.asarray(ds4.values),
    "scales": np.asarray(ds4.scales), "sums": np.asarray(ds4.sums)}, reference=reference,
    policy=GuardPolicy(rtol=1e-6, atol=1e-3), identity={"candidate": "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0",
    "geometry": {"tile": [16, 16, 256], "workgroup": [32, 16, 1], "lds_bytes": 256}, "owner_coverage": coverage})
  out = run_isolated_guarded_execution(builder=make_tinygrad_bundle_builder(build=_build_emitted_amd_bundle, mode="candidate"),
                                        request=request, health_probe=_health, timeout_seconds=timeout_seconds)
  evidence = out.to_dict()
  guarded = evidence.get("guarded") or {}
  evidence["reference_authority"] = {"kind": "canonical_cpu", "compared": bool(guarded.get("full_output_compared"))}
  evidence["gpu_kernel_authority"] = {"kind": "emitted_amd_program", "compiled_and_bound": out.dispatch_state not in ("not_attempted", "failed") or bool(guarded.get("dispatch_performed")),
                                       "dispatch_performed": bool(guarded.get("dispatch_performed")),
                                       "claimable": bool(out.passed and guarded.get("dispatch_performed") and guarded.get("full_output_compared"))}
  if out.passed and not guarded.get("dispatch_performed"):
    evidence["passed"] = False
    evidence.setdefault("errors", []).append("GPU correctness requires a dispatched compiled program")
  evidence["owner_coverage"] = coverage
  evidence["resource_metadata"] = {"lds_bytes": 256, "bounded": True, "program_identity": request.identity["candidate"]}
  if out.passed and compare_direct:
    direct = run_q4k_q8_1_mmq_tile_amd(raw, ds4.values, ds4.scales, spec).output
    evidence["direct_packed"] = {"compared": True, "passed": bool(np.allclose(reference, direct, rtol=1e-6, atol=1e-3))}
  return evidence
