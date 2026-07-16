"""Neutral fail-closed evidence contract for the blocked DS4 reuse probe."""
from __future__ import annotations

from typing import Any

COOP_128_REUSE_PROBE_ID = "q4k_q8_ds4_coop_128x128_k256_reuse_v0"


def cooperative_128_reuse_evidence() -> dict[str, Any]:
  return {
    "probe_id": COOP_128_REUSE_PROBE_ID, "status": "BLOCKED_FAIL_CLOSED",
    "default_off": True, "research_only": True,
    "candidate": {"tile": [128, 128, 256], "activation_reuse_panels": 8,
                  "lifecycle": "cooperative_multi_wave"},
    "compile": {"status": "NOT_RUN", "binary_identity": None},
    "correctness": {"status": "NOT_RUN", "full_output": False},
    "performance": {"status": "NOT_RUN", "same_session": False},
    "production_dispatch_changed": False, "rollback_route": "direct-packed",
    "exact_blockers": [
      "current DS4 cooperative atom rejects shape: bounded to 16x16x256",
      "current mapping requires one wave per workgroup; 128x128 needs multi-wave ownership",
      "emitted store_owner metadata is absent (observed 0; expected 16384 output owners)",
      "no compiler/resource evidence for LDS, VGPR, occupancy, or scratch at 128x128",
    ],
  }


__all__ = ["COOP_128_REUSE_PROBE_ID", "cooperative_128_reuse_evidence"]
