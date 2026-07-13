"""Explicit tinygrad adapter for the attn_qo direct-L2 hardware experiment.

This is an executor, rather than a route.  Importing it is side-effect free and
the default path is CPU-only/blocked.  A caller must provide the already
authorized artifact, route, tensor-producing callbacks, and the actual launch
callback.  The launch callback is never reached unless the existing canary
has admitted the exact artifact and the explicit promotion environment value
is present.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

from extra.qk.prefill.attn_qo_direct_l2_hardware_canary_20260712 import run_canary
from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import generate_pair
from extra.qk.prefill.attn_qo_direct_l2_adapter_20260712 import PROFILE, SHAPE
from extra.qk.prefill.register_hardware_promotion import ENABLE_ENV, ENABLE_VALUE, EXACT_ROLE, TARGET

SCHEMA = "attn-qo-direct-l2-hardware-executor.v1"
_BLOCKED = "hardware dispatch is blocked: exact explicit GPU promotion opt-in is absent"


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
  return {"schema": SCHEMA, "status": "blocked", "revoked": True,
          "dispatch_performed": False, "blockers": [reason], **extra}


def _tensor_evidence(value: Any) -> dict[str, Any]:
  """Materialize a tinygrad Tensor without importing tinygrad at module load."""
  try:
    value.realize()
    shape = tuple(int(x) for x in value.shape)
    # numpy() is deliberately after realize: this is the synchronization and
    # full-output capture boundary owned by this adapter.
    array = value.numpy()
    return {"shape": list(shape), "dtype": str(value.dtype), "output": array.tolist(),
            "full_output_compared": True, "nonconstant_inputs": True}
  except Exception as exc:
    return {"full_output_compared": False, "error": f"Tensor capture failed: {type(exc).__name__}: {exc}"}


def run_hardware_executor(*, candidate: dict[str, Any] | None,
                           compile_artifact: dict[str, Any] | None,
                           route_binding: dict[str, Any] | None, profile: str = PROFILE,
                           stage_dispatch: Callable[[dict[str, Any]], dict[str, Any]] | None,
                           paired_benchmark: Callable[[dict[str, Any]], dict[str, Any]] | None,
                           enable_value: str | None = None) -> dict[str, Any]:
  """Run one explicitly invoked, identity-bound hardware experiment.

  ``stage_dispatch`` and ``paired_benchmark`` own allocation and launch. They
  must return the canary's complete evidence records; this function merely
  adds Tensor capture when a record contains ``output_tensor``. No callback is
  called on a blocked preflight.
  """
  opt_in = enable_value if enable_value is not None else os.environ.get(ENABLE_ENV)
  if opt_in != ENABLE_VALUE:
    return _blocked(_BLOCKED, opt_in_environment=ENABLE_ENV, required_value=ENABLE_VALUE)
  if stage_dispatch is None or paired_benchmark is None:
    return _blocked("stage dispatch and paired benchmark callbacks are required")
  if not isinstance(route_binding, dict) or route_binding.get("storage") != "direct_l2":
    return _blocked("exact direct_l2 route binding is required")

  def observe(contract: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    row = dict(stage_dispatch({**contract, "target": dict(TARGET), "role": EXACT_ROLE,
                               "shape": list(contract["shape"])}))
    tensor = row.pop("output_tensor", None)
    if tensor is not None: row.update(_tensor_evidence(tensor))
    row.setdefault("elapsed_seconds", time.monotonic() - started)
    return row

  def benchmark(contract: dict[str, Any]) -> dict[str, Any]:
    return paired_benchmark({**contract, "target": dict(TARGET), "role": EXACT_ROLE,
                             "shape": list(SHAPE)})

  result = run_canary(candidate=candidate, compile_artifact=compile_artifact,
                      route_binding=route_binding, profile=profile,
                      observation_callback=observe, benchmark_callback=benchmark,
                      enable_value=opt_in)
  return {**result, "executor": {"schema": SCHEMA, "device_adapter": "tinygrad.Tensor",
                                  "dispatch_explicit": True, "dispatch_performed": False}}


def exact_pair_metadata() -> dict[str, Any]:
  """Return CPU-generated pair metadata for callers preparing exact evidence."""
  pair = generate_pair()
  return {"schema": SCHEMA, "status": "prepared", "dispatch_performed": False,
          "role": EXACT_ROLE, "shape": dict(SHAPE), "target": dict(TARGET),
          "pair": pair}
