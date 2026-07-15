"""Small, pre-late-codegen regression probes for the generated MMQ graph.

These helpers intentionally inspect graph contracts only.  They do not select
routes, mutate descriptors, or compile/dispatch a device program.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from extra.qk.mmq_abi import Q4K_Q8_MMQ_ABI
from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS


MMQ_PROMOTION_EVIDENCE = ("correctness", "guard", "gpu_health", "resources", "identity", "fallback")


def validate_mmq_candidate_evidence_gate(evidence: Any) -> dict[str, Any]:
  """Return the fail-closed timing/promotion decision for one candidate.

  This is intentionally a consumer-side invariant.  Producers may use their
  existing evidence schemas, but a candidate cannot be timed or promoted from
  summary booleans unless every independent safety/provenance fact is present.
  ``fallback`` is satisfied only by explicit evidence that a usable fallback
  exists, or that no fallback is required.
  """
  if not isinstance(evidence, dict):
    return {"timing_allowed": False, "promotion_eligible": False,
            "blockers": ["candidate evidence must be a mapping"],
            "evidence": {name: False for name in MMQ_PROMOTION_EVIDENCE}}

  def passed(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("passed") is not True:
      return False
    status = value.get("status", "PASS")
    return isinstance(status, str) and status.upper() == "PASS"

  checks = {
    "correctness": passed(evidence.get("correctness")),
    "guard": passed(evidence.get("guard")),
    "gpu_health": passed(evidence.get("gpu_health")),
    "resources": passed(evidence.get("resources")),
    "identity": passed(evidence.get("identity")),
    "fallback": evidence.get("no_fallback") is True or passed(evidence.get("fallback")),
  }
  blockers = [f"missing or failed {name} evidence" for name, ok in checks.items() if not ok]
  allowed = not blockers
  return {"timing_allowed": allowed, "promotion_eligible": allowed,
          "blockers": blockers, "evidence": checks}


@dataclass(frozen=True)
class VectorPointerBase:
  index: Any
  base: Any
  dtype: Any


def vector_pointer_bases(root: Any) -> tuple[VectorPointerBase, ...]:
  """Find INDEX nodes whose buffer/base is vector-valued before late passes."""
  nodes = root.toposort() if hasattr(root, "toposort") else ()
  found = []
  for node in nodes:
    if getattr(getattr(node, "op", None), "name", node.op) != "INDEX" or not node.src:
      continue
    base = node.src[0]
    dtype = getattr(base, "dtype", None)
    if getattr(dtype, "count", 1) > 1:
      found.append(VectorPointerBase(node, base, dtype))
  return tuple(found)


def reject_vector_pointer_bases(root: Any) -> None:
  """Reject vector-valued pointer bases without rejecting vector WMMA carriers."""
  found = vector_pointer_bases(root)
  if found:
    raise ValueError(f"vector-valued pointer base(s): {len(found)}")


def validate_generated_mmq_abi(words: np.ndarray, xq: np.ndarray, scales: np.ndarray, *, m: int, n: int, k: int) -> None:
  """Validate the exact emitter input ABI, including per-Q8-block metadata."""
  blocks = n * (k // Q4_K_BLOCK_ELEMS)
  groups = m * (k // Q8_1_BLOCK_ELEMS)
  if words.dtype != np.uint32 or words.size != blocks * Q4K_Q8_MMQ_ABI.q4_words_per_block:
    raise ValueError("Q4 ABI requires 36 uint32 words per 256-value block")
  if xq.dtype != np.int8 or xq.size != groups * Q4K_Q8_MMQ_ABI.q8_values_per_block:
    raise ValueError("Q8 ABI requires 32 int8 values per block")
  if scales.dtype != np.float32 or scales.size != groups:
    raise ValueError("Q8 ABI requires one float32 scale per block")
  if words.size != blocks * Q4K_WORDS_PER_BLOCK:
    raise ValueError("generated MMQ Q4 word count does not match N*K block ownership")
  if xq.size != groups * Q8_1_BLOCK_ELEMS or scales.size != groups:
    raise ValueError("generated MMQ Q8 values/scales do not match M*K block ownership")
