"""EXP-only qualification adapter for the production packed-WMMA route.

The production owner is :mod:`tinygrad.llm.packed_wmma_prefill`.  This module
only turns a production ``PackedWmmaRoute`` into the old isolated GPU oracle
workload; it does not select or execute a model-forward route itself.
"""
from __future__ import annotations

import copy
import os
import tempfile

from tinygrad.llm import packed_wmma_prefill as production
from extra.llm_research.prefill.packed_wmma_correctness_canary import build_artifact, candidate_payload, run_canary

PROFILE = "qwen3_14b_q4k_m_gfx1100"


def _payload_for_production_row(row: production.PackedWmmaRoute) -> dict:
  """Rebind the oracle template to one exact production row and its geometry."""
  payload = copy.deepcopy(candidate_payload(PROFILE, row.role))
  if tuple(payload["workload"]["shape"][key] for key in ("m", "n", "k")) != row.shape:
    raise ValueError(f"oracle workload does not match production row {row}")
  g, schedule = row.geom, payload["schedule"]
  schedule["tile"] = {"m": g["tm"], "n": g["tn"], "k": g["tk"]}
  schedule["waves"] = {"m": g["wm"], "n": g["wn"]}
  schedule["threads"] = g["wm"] * g["wn"] * 32
  a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80
  schedule["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  schedule["lds"]["strides"] = {"a": 80, "b": 80}
  schedule["pipeline"]["buffer_count"] = g["bc"]
  return payload


def verify_production_row(row: production.PackedWmmaRoute, *, timeout_seconds: float = 90.0,
                           device: str = "AMD") -> tuple[bool, float | None]:
  """Run the isolated oracle against exactly the supplied production row."""
  fd, artifact_path = tempfile.mkstemp(prefix=f"packed_wmma_production_{row.quant}_{row.role}_", suffix=".npz")
  os.close(fd)
  try:
    build_artifact(row.quant, artifact_path, row.shape)
    outcome = run_canary(row.quant, artifact_path, timeout_seconds, base_payload=_payload_for_production_row(row),
                          device=device)
  finally:
    try: os.remove(artifact_path)
    except OSError: pass
  guarded = outcome.get("guarded", {})
  return bool(outcome.get("passed")), guarded.get("max_abs_error") if isinstance(guarded, dict) else None


def install_production_qualification_verifier(*, timeout_seconds: float = 90.0, device: str = "AMD") -> None:
  """Make production gate_combo use the isolated EXP oracle for this process."""
  production.set_packed_wmma_canary_verifier(
    lambda row: verify_production_row(row, timeout_seconds=timeout_seconds, device=device))
