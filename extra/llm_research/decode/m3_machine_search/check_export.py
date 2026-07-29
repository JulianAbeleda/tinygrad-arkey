#!/usr/bin/env python3
"""Deterministic integrity checks for M3's symbolic topology-plan export."""
from __future__ import annotations
import json
from pathlib import Path
from export_search import REQUEST, EXPORT, ident

FORBIDDEN=("q4k_g3_lanemap_gemv_kernel", "emit_q6k_gemv_kernel", "q6k_coop_partial_kernel", "custom_kernel")

def validate(request, export):
  assert export["request_id"] == ident("search-request", request)
  assert export["ranking"]["kind"] == "deterministic_cpu_structural"
  assert export["promotion"]["status"] == "blocked"
  assert export["promotion"]["missing_primitive_lowerers"] == ["q4k_packed_block_dot", "q6k_packed_block_dot", "external_reduce"]
  assert export["promotion"]["missing_executor"] == "route-bound generic topology-plan executor"
  for route, result in export["routes"].items():
    rows=result["ranked_population"]
    assert result["population_size"] == len(rows) and result["selected"] == rows[0]
    assert [x["rank"] for x in rows] == list(range(1, len(rows)+1))
    assert rows == sorted(rows, key=lambda x:(x["structural_score"], x["candidate_id"]))
    for row in rows:
      text=json.dumps(row["plan"], sort_keys=True)
      assert not any(name in text for name in FORBIDDEN), (route, row["candidate_id"])
      assert row["plan_id"] == ident("generic-plan", row["plan"])

def main():
  validate(json.loads(REQUEST.read_text()), json.loads(EXPORT.read_text()))
  print("M3 export valid: finite populations, symbolic topology plans, and blocked promotion record")
if __name__ == "__main__": main()
