"""Fail-closed structural admission for the staged Q6_K fp16-WMMA route.

The measured fixture identifies evidence for one candidate.  It is not model
authority: admission depends only on invocation, target, evidence, and memory
facts supplied by the selected model's actual inventory and runtime planner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

Q6K_WMMA_ROUTE = "staged_dequant_then_fp16_wmma"
Q6K_WMMA_CANDIDATE_ID = "q6k/staged-fp16-wmma/m512-n4096-k12288/gfx1100-wave32-v1"
Q6K_WMMA_EVIDENCE_ID = "experimental_q6k_8b_like_ffn_down_stage1"
Q6K_WMMA_FIXTURE_SHAPE = {"M": 512, "N": 4096, "K": 12288}
Q6K_WMMA_SHAPE = Q6K_WMMA_FIXTURE_SHAPE
Q6K_WMMA_TAILS = {"M": 0, "N": 0, "K": 0}
Q6K_WMMA_REQUIRED_LDS_BYTES = 32 * 1024
Q6K_WMMA_LIVE_BYTES = (Q6K_WMMA_SHAPE["N"] * Q6K_WMMA_SHAPE["K"] * 2 +
                         Q6K_WMMA_SHAPE["M"] * Q6K_WMMA_SHAPE["K"] * 2 +
                         Q6K_WMMA_SHAPE["M"] * Q6K_WMMA_SHAPE["N"] * 4)
Q6K_WMMA_ROLE_SHAPE_BOUNDARY = {"ffn_down": Q6K_WMMA_FIXTURE_SHAPE}


@dataclass(frozen=True)
class Q6KWMMAAdmission:
  admitted: bool
  route: str
  provenance: str
  rollback_route: str
  errors: tuple[str, ...] = ()
  candidate_id: str | None = None
  evidence_id: str | None = None
  required_live_bytes: int = Q6K_WMMA_LIVE_BYTES
  live_byte_budget: int | None = None

  def to_json(self) -> dict[str, Any]:
    return {"admitted": self.admitted, "route": self.route, "provenance": self.provenance,
            "rollback_route": self.rollback_route, "errors": list(self.errors),
            "candidate_id": self.candidate_id, "evidence_id": self.evidence_id,
            "required_live_bytes": self.required_live_bytes, "live_byte_budget": self.live_byte_budget}


def admit_q6k_wmma(*, role: str, shape: dict[str, int], phase: str | None = None, quant: str | None = None,
                   tails: dict[str, int] | None = None, backend: str | None = None, arch: str | None = None,
                   wave_size: int | None = None, lds_bytes: int | None = None,
                   candidate_id: str | None = None, evidence_id: str | None = None, evidence_valid: bool = False,
                   live_byte_budget: int | None = None, enabled: bool | None = None,
                   model_profile: str | None = None, model_path: str | None = None) -> Q6KWMMAAdmission:
  """Admit only the exact measured structural capability.

  ``model_profile`` and ``model_path`` are optional diagnostic provenance and
  deliberately do not participate in any admission or identity decision.
  """
  errors: list[str] = []
  if enabled is None: enabled = os.environ.get("PREFILL_Q6K_WMMA", "0").lower() in {"1", "true", "yes", "on"}
  if not enabled: errors.append("explicit Q6 WMMA opt-in missing")
  if phase != "prefill": errors.append("phase is not prefill")
  if quant != "Q6_K": errors.append("quant is not Q6_K")
  if role != "ffn_down": errors.append("role is not the validated ffn_down role")
  if shape != Q6K_WMMA_FIXTURE_SHAPE: errors.append("M/N/K are outside candidate capability")
  if tails != Q6K_WMMA_TAILS: errors.append("tail class is outside candidate capability")
  if backend != "AMD" or arch != "gfx1100" or wave_size != 32:
    errors.append("target backend/architecture/wave is outside candidate capability")
  if not isinstance(lds_bytes, int) or isinstance(lds_bytes, bool) or lds_bytes < Q6K_WMMA_REQUIRED_LDS_BYTES:
    errors.append("target LDS capability is insufficient")
  if candidate_id != Q6K_WMMA_CANDIDATE_ID: errors.append("candidate identity mismatch")
  if evidence_id != Q6K_WMMA_EVIDENCE_ID or not evidence_valid: errors.append("validated candidate evidence missing")
  if not isinstance(live_byte_budget, int) or isinstance(live_byte_budget, bool) or live_byte_budget < 0:
    errors.append("explicit live-byte budget missing or invalid")
  elif Q6K_WMMA_LIVE_BYTES > live_byte_budget:
    errors.append("staged live bytes exceed explicit byte budget")

  provenance = Q6K_WMMA_EVIDENCE_ID if not errors else "rollback"
  # Optional names/paths may aid logs, but are intentionally absent from the
  # result's semantic identity and cannot change the decision.
  _ = model_profile, model_path
  return Q6KWMMAAdmission(not errors, Q6K_WMMA_ROUTE if not errors else "direct_packed", provenance,
                          "direct_packed", tuple(errors),
                          Q6K_WMMA_CANDIDATE_ID if not errors else None,
                          Q6K_WMMA_EVIDENCE_ID if not errors else None,
                          Q6K_WMMA_LIVE_BYTES, live_byte_budget)
