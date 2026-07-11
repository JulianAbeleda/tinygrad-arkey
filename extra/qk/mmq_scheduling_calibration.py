#!/usr/bin/env python3
"""Identity-bound scheduling calibration from independent generated microbenchmarks."""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping

SCHEMA = "tinygrad.mmq_scheduling_calibration.v1"


def selected_scheduling_cases():
  from extra.qk.mmq_calibration import (
    dependent_valu_case, independent_valu_case, issue_case, launch_case, resource_pressure_case,
  )
  return tuple([launch_case(wg) for wg in (1, 32, 64, 96, 128, 192)] +
               [resource_pressure_case(96, streams) for streams in (4, 8, 16, 32)] +
               [dependent_valu_case(96, 64), independent_valu_case(96, 64, 4)] +
               [issue_case(family) for family in ("dependent_valu_int", "dependent_salu", "mixed_salu_valu")])


def summarize_scheduling_relationships(samples:Iterable[Mapping[str, Any]]) -> dict[str, Any]:
  rows = list(samples)
  grouped: dict[str, list[Mapping[str, Any]]] = {}
  for row in rows: grouped.setdefault(str(row["case_id"]), []).append(row)
  summaries = {}
  for case_id, case_rows in grouped.items():
    counters = case_rows[0]["counters"]
    summaries[case_id] = {"samples": len(case_rows), "median_ms": statistics.median(row["median_ms"] for row in case_rows),
      "median_counters": {name: statistics.median(row["counters"][name] for row in case_rows) for name in counters}}
  grid = sorted((row for case_id, row in summaries.items() if case_id.startswith("launch.wg")),
                key=lambda row: row["median_counters"]["SQ_WAVES"])
  resource = sorted((row for case_id, row in summaries.items() if case_id.startswith("resource_pressure")),
                    key=lambda row: row["median_counters"]["SQ_INSTS_VALU"])
  return {"cases": summaries,
          "grid_relationship": {"waves": [row["median_counters"]["SQ_WAVES"] for row in grid],
                                "busy_cycles": [row["median_counters"]["SQ_BUSY_CYCLES"] for row in grid],
                                "wave_cycles": [row["median_counters"]["SQ_WAVE_CYCLES"] for row in grid]},
          "resource_relationship": {"valu_instructions": [row["median_counters"]["SQ_INSTS_VALU"] for row in resource],
                                    "wave_cycles": [row["median_counters"]["SQ_WAVE_CYCLES"] for row in resource],
                                    "wait_cycles": [row["median_counters"]["SQ_WAIT_ANY"] for row in resource]},
          "truth_status": "derived"}


def collect_scheduling_calibration(artifact_dir:Path, *, repetitions:int, seed:int,
                                   system_snapshot_id:str) -> dict[str, Any]:
  if repetitions < 3: raise ValueError("repetitions must be >= 3")
  if not system_snapshot_id.startswith("sha256:"): raise ValueError("system_snapshot_id must be content-addressed")
  from tinygrad.device import Compiled
  from extra.qk.mmq_amd_pmc import _decode_event
  from extra.qk.mmq_amd_telemetry import DEFAULT_SENSORS, read_sensor
  from extra.qk.mmq_calibration import run_calibration_case
  artifact_dir = Path(artifact_dir)
  cases, expected = selected_scheduling_cases(), {}
  for case in cases:
    source = json.loads((artifact_dir / f"{case.case_id}.json").read_text())
    expected[case.case_id] = source["hashes"]["binary_sha256"]
  rng, samples, orders = random.Random(seed), [], []
  for repetition in range(repetitions):
    order = list(cases); rng.shuffle(order); orders.append([case.case_id for case in order])
    for case in order:
      before = {name: read_sensor(path) for name, path in DEFAULT_SENSORS.items()}
      Compiled.profile_events.clear()
      result = run_calibration_case(case, warmups=1, rounds=3, system_snapshot_id=system_snapshot_id)
      events = [event for event in Compiled.profile_events if type(event).__name__ == "ProfilePMCEvent"]
      if not events: raise RuntimeError(f"{case.case_id} emitted no PMC event")
      if result["hashes"]["binary_sha256"] != expected[case.case_id]: raise RuntimeError(f"{case.case_id} binary identity mismatch")
      after = {name: read_sensor(path) for name, path in DEFAULT_SENSORS.items()}
      samples.append({"repetition": repetition, "case_id": case.case_id, "family": case.family,
                      "workgroups": case.workgroups, "independent_streams": case.independent_streams,
                      "binary_sha256": expected[case.case_id], "median_ms": result["median_ms"],
                      "counters": _decode_event(events[-1]), "telemetry_before": before, "telemetry_after": after})
  return {"schema": SCHEMA, "provenance_class": "generated_microbenchmark", "source_artifact_dir": str(artifact_dir),
          "system_snapshot_id": system_snapshot_id, "collector": "tinygrad_kfd_native_pmc",
          "counter_liveness": "live", "repetitions": repetitions, "seed": seed, "randomized_orders": orders,
          "samples": samples, "relationships": summarize_scheduling_relationships(samples),
          "candidate_timing_used_for_fit": False, "production_dispatch_changed": False}


def validate_scheduling_calibration(artifact:Mapping[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  if artifact.get("provenance_class") != "generated_microbenchmark": raise ValueError("generated provenance required")
  if artifact.get("candidate_timing_used_for_fit") is not False: raise ValueError("candidate timing must not be used")
  if artifact.get("counter_liveness") != "live": raise ValueError("SQ counters must be live")
  if not artifact.get("samples"): raise ValueError("samples are required")
