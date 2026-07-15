"""Search-session adapter for descriptor-generated Q4_K x Q8_1 candidates.

This is research glue only.  It deliberately has no route-registration or
dispatch-default side effects: the generated harness owns compilation and the
isolated guarded executor owns execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from extra.qk.q4k_q8_mmq_generated_harness import validate_generated_candidate
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec
from extra.qk.q4k_q8_mmq_search import MMQDescriptor, SearchPolicy, run_search
from extra.qk.mmq_capability import MMQHardwareCapability, MMQRequest, GFX11_MMQ_CAPABILITY

BLOCKED_FAIL_CLOSED = "BLOCKED_FAIL_CLOSED"


def _spec(base: Q4KQ8MMQPrefillSpec, descriptor: MMQDescriptor) -> Q4KQ8MMQPrefillSpec:
  """Apply only generated axes; descriptor identity is retained separately."""
  axes = dict(descriptor.axes)
  axes.pop("resources", None)  # resource budgets are search constraints, not spec fields
  fields = Q4KQ8MMQPrefillSpec.__dataclass_fields__
  return Q4KQ8MMQPrefillSpec(**{**base.__dict__, **{k: v for k, v in axes.items() if k in fields}})


@dataclass
class CandidateSession:
  base_spec: Q4KQ8MMQPrefillSpec
  words: np.ndarray
  xq: np.ndarray
  scales: np.ndarray
  reference: np.ndarray
  validator: Callable[..., Mapping[str, Any]] = validate_generated_candidate
  candidate_timer: Callable[..., Mapping[str, Any]] | None = None
  direct_timer: Callable[..., Mapping[str, Any]] | None = None
  timeout_seconds: float = 30.0
  request: MMQRequest | None = None
  capability: MMQHardwareCapability = GFX11_MMQ_CAPABILITY

  def prepare(self, descriptor: MMQDescriptor) -> dict[str, Any]:
    spec = _spec(self.base_spec, descriptor)
    self.capability.validate()
    if self.request is not None: self.request.validate()
    if spec.lds_bytes > self.capability.max_lds_bytes:
      raise ValueError("descriptor LDS exceeds hardware capability")
    spec = Q4KQ8MMQPrefillSpec(**{**spec.__dict__, "wave_width": self.capability.wave_width,
                                  "lds_bytes": spec.lds_bytes})
    spec.validate()  # invalid generated descriptors never reach compilation
    return {"descriptor": descriptor, "spec": spec}

  def check_correctness(self, prepared: Mapping[str, Any]) -> Mapping[str, Any]:
    descriptor, spec = prepared["descriptor"], prepared["spec"]
    try:
      evidence = dict(self.validator(self.words, self.xq, self.scales, self.reference, spec,
                                     timeout_seconds=self.timeout_seconds,
                                     candidate_id=descriptor.candidate_id))
    except BaseException as exc:
      return {"passed": False, "verdict": BLOCKED_FAIL_CLOSED,
              "blocker": f"{type(exc).__name__}: {exc}", "candidate_id": descriptor.candidate_id}
    # Every correctness result carries both generator and compiled-run identity.
    evidence["candidate_id"] = descriptor.candidate_id
    evidence["descriptor_identity"] = descriptor.canonical()
    evidence["verdict"] = "PASS" if evidence.get("gpu_correctness_claimable") is True else BLOCKED_FAIL_CLOSED
    evidence["passed"] = evidence["verdict"] == "PASS"
    return evidence

  def measure(self, prepared: Mapping[str, Any], *, warmups: int, rounds: int) -> Mapping[str, Any]:
    if self.candidate_timer is None: return {"passed": False, "verdict": BLOCKED_FAIL_CLOSED, "min_ms": None}
    try: return dict(self.candidate_timer(prepared["spec"], warmups=warmups, rounds=rounds))
    except BaseException as exc:
      return {"passed": False, "verdict": BLOCKED_FAIL_CLOSED, "min_ms": None,
              "blocker": f"{type(exc).__name__}: {exc}"}

  def measure_direct_packed(self, *, warmups: int, rounds: int) -> Mapping[str, Any]:
    if self.direct_timer is None: return {"passed": False, "verdict": BLOCKED_FAIL_CLOSED, "min_ms": None}
    try: return dict(self.direct_timer(warmups=warmups, rounds=rounds))
    except BaseException as exc:
      return {"passed": False, "verdict": BLOCKED_FAIL_CLOSED, "min_ms": None,
              "blocker": f"{type(exc).__name__}: {exc}"}


def evaluate_candidates(*, axes: Mapping[str, Any], base_spec: Q4KQ8MMQPrefillSpec,
                        words: np.ndarray, xq: np.ndarray, scales: np.ndarray,
                        reference: np.ndarray, policy: SearchPolicy = SearchPolicy(),
                        **session_kwargs: Any) -> dict[str, Any]:
  """Compile and evaluate generated candidates, correctness-before-timing."""
  def factory() -> CandidateSession:
    return CandidateSession(base_spec, words, xq, scales, reference, **session_kwargs)
  return run_search(axes=axes, session_factory=factory, policy=policy)


__all__ = ["BLOCKED_FAIL_CLOSED", "CandidateSession", "evaluate_candidates"]
