"""Bounded Q6_K/Q8_1 cooperative-tile probe, evidence-only.

The probe mirrors llama's semantic lifecycle (Q6 payload staged once, Q8_1
activation panels staged once, dot over the staged panels) but deliberately
does not emit or register a route.  ``enabled=True`` is fail-closed until a
real lowered program supplies correctness and resource facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from extra.qk.q6k_mmq_vocabulary import Q6K_BLOCK_BYTES, q6k_weight

SCHEMA = "tinygrad.q6k.coop_tile_probe.v1"
PROBE_ID = "q6k_q8_1_mmq_coop_m16_n16_k256_v0"


@dataclass(frozen=True)
class Q6KCoopTileProbe:
  tile_m: int = 16
  tile_n: int = 16
  tile_k: int = 256
  wave_size: int = 32
  workgroup_size: int = 64
  enabled: bool = False

  def validate(self) -> None:
    if (self.tile_m, self.tile_n, self.tile_k) != (16, 16, 256):
      raise ValueError("Q6 cooperative probe is bounded to M=16,N=16,K=256")
    if self.wave_size != 32 or self.workgroup_size != 64:
      raise ValueError("Q6 probe has no unverified physical mapping")
    if self.enabled:
      raise ValueError("Q6 cooperative probe has no lowered emitter; refusing dispatch")

  def resource_evidence(self) -> dict[str, Any]:
    self.validate()
    # Q6 q/scales/d plus Q8 values/scale/sum staged for one K tile.
    q6_stage = self.tile_n * (128 + 64 + 16 + 2)
    q8_stage = self.tile_m * (self.tile_k + (self.tile_k // 32) * 8)
    return {"status": "MODEL_ONLY", "q6_stage_bytes": q6_stage,
            "q8_stage_bytes": q8_stage, "lds_bytes_estimate": q6_stage + q8_stage,
            "scratch_bytes": "UNKNOWN", "vgpr": "UNKNOWN", "source": "bounded_formula"}

  def evidence(self, *, run_correctness: bool = False) -> dict[str, Any]:
    self.validate()
    report: dict[str, Any] = {
      "schema": SCHEMA, "probe_id": PROBE_ID, "default_off": True,
      "production_dispatch_changed": False, "rollback_route": "direct-packed",
      "candidate": {"quant": "Q6_K", "activation": "Q8_1", "tile": [16, 16, 256],
                     "lifecycle": "stage_q6_and_q8_once_then_cooperative_dot"},
      "correctness": {"status": "NOT_RUN", "full_output": False},
      "resources": self.resource_evidence(),
      "blocker": "no Q6-specific lowered cooperative emitter or final-program resource artifact",
    }
    if run_correctness:
      report["correctness"] = _bounded_correctness()
    return report


def _bounded_correctness() -> dict[str, Any]:
  rng = np.random.default_rng(6)
  packed = rng.integers(0, 256, size=(16, Q6K_BLOCK_BYTES), dtype=np.uint8)
  packed[:, 208:210] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
  x = rng.normal(size=(16, 256)).astype(np.float32)
  staged = np.empty((16, 256), dtype=np.float32)
  for n in range(16):
    for g in range(16):
      for p in range(16): staged[n, g * 16 + p] = q6k_weight(packed[n].tobytes(), g, p)
  direct = x @ staged.T
  # Q8_1 staging: one scale and sum per 32-element group, then reconstruct.
  q = np.empty_like(x, dtype=np.int8); scales = np.empty((16, 8), dtype=np.float32)
  for m in range(16):
    for b in range(8):
      chunk = x[m, b * 32:(b + 1) * 32]; scale = max(float(np.max(np.abs(chunk))) / 127.0, 1e-8)
      scales[m, b] = scale; q[m, b * 32:(b + 1) * 32] = np.rint(chunk / scale).clip(-127, 127).astype(np.int8)
  staged_x = q.astype(np.float32) * np.repeat(scales, 32, axis=1)
  got = staged_x @ staged.T
  error = float(np.max(np.abs(direct - got)))
  return {"status": "PASS" if np.isfinite(got).all() else "FAIL", "full_output": True,
          "max_abs_vs_float_activation": error, "finite": bool(np.isfinite(got).all()),
          "q8_groups": 8, "q6_blocks": 1}


def run_q6k_coop_tile_probe(*, enabled: bool = False, run_correctness: bool = False) -> dict[str, Any]:
  return Q6KCoopTileProbe(enabled=enabled).evidence(run_correctness=run_correctness)


__all__ = ["Q6KCoopTileProbe", "run_q6k_coop_tile_probe", "SCHEMA", "PROBE_ID"]
