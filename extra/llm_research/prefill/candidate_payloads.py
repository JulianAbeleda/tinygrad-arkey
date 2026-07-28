"""Shared access to the promoted prefill candidate-set payloads.

Both the packed-WMMA production candidates (packed_wmma_prefill_candidates.py) and the
correctness canary (packed_wmma_correctness_canary.py) need to read the promoted
candidate-set artifact and pick the per-role schedule template out of it. That load +
role lookup lived, byte-identical, in both places; this module is the single home for it.
"""
from __future__ import annotations

import json
from pathlib import Path

from extra.llm_research.route_manifest import promoted_prefill_candidate_policy


def load_candidate_payloads(candidate_set_path: str | None = None) -> list[dict]:
  """Return the payload dicts of the promoted (or explicitly given) candidate set."""
  path = candidate_set_path or promoted_prefill_candidate_policy()["candidate_set_path"]
  candidate_set = json.loads(Path(path).read_text())
  return [row["payload"] for row in candidate_set["entries"]]


def find_role_template(payloads: list[dict], role: str) -> dict:
  """First payload whose workload.role matches `role`; raise if the set has none."""
  template = next((p for p in payloads if p["workload"]["role"] == role), None)
  if template is None: raise ValueError(f"candidate set has no schedule template for role {role!r}")
  return template


def template_payload_for_role(role: str, candidate_set_path: str | None = None) -> dict:
  """Load the candidate set and return its schedule template for `role`."""
  return find_role_template(load_candidate_payloads(candidate_set_path), role)
