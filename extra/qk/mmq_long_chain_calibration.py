#!/usr/bin/env python3
"""Long generated issue chains for SQ-cycle to wall-time calibration."""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
from typing import Any, Mapping

SCHEMA = "tinygrad.mmq_long_chain_calibration.v1"


def long_chain_cases(include_controls:bool=True):
  from extra.qk.mmq_calibration import CalibrationCase, dependent_valu_case
  cases = [dependent_valu_case(96, length) for length in (1024, 4096, 16384)]
  if include_controls:
    cases += [CalibrationCase("dependent_salu.wg96.n1024", "dependent_salu", 96, 1024),
              CalibrationCase("mixed_salu_valu.wg96.n1024", "mixed_salu_valu", 96, 1024)]
  return tuple(cases)


def collect_long_chain_mode(*, mode:str, system_snapshot_id:str, seed:int=20260711,
                            include_controls:bool=True) -> dict[str, Any]:
  if mode not in ("auto", "profile_standard"): raise ValueError("mode must be auto or profile_standard")
  from tinygrad.device import Compiled, Device
  from extra.qk.mmq_amd_pmc import _decode_event
  from extra.qk.mmq_amd_telemetry import DEFAULT_SENSORS, read_sensor
  from extra.qk.mmq_calibration import run_calibration_case
  pmc_enabled = bool(getattr(Device["AMD"], "pmc_enabled", False))
  if mode == "profile_standard" and not pmc_enabled: raise RuntimeError("profile_standard collection requires native PMC")
  if mode == "auto" and pmc_enabled: raise RuntimeError("auto collection must not enable native PMC")
  cases = list(long_chain_cases(include_controls)); random.Random(seed).shuffle(cases)
  rows = []
  for case in cases:
    # Load and compile outside the measured protocol.
    warm = run_calibration_case(case, warmups=1, rounds=3, system_snapshot_id=system_snapshot_id)
    before = {name: read_sensor(path) for name, path in DEFAULT_SENSORS.items()}
    Compiled.profile_events.clear()
    result = run_calibration_case(case, warmups=5, rounds=30, system_snapshot_id=system_snapshot_id)
    after = {name: read_sensor(path) for name, path in DEFAULT_SENSORS.items()}
    if result["hashes"]["binary_sha256"] != warm["hashes"]["binary_sha256"]:
      raise RuntimeError(f"{case.case_id} binary changed between warmup and measurement")
    events = [event for event in Compiled.profile_events if type(event).__name__ == "ProfilePMCEvent"]
    counters = _decode_event(events[-1]) if events else None
    rows.append({"case_id": case.case_id, "family": case.family, "chain_length": case.chain_length,
                 "binary_sha256": result["hashes"]["binary_sha256"], "binary_bytes": result["binary_bytes"],
                 "protocol": {"warmups": 5, "rounds": 30}, "samples_ms": result["samples_ms"],
                 "median_ms": result["median_ms"], "counters": counters,
                 "telemetry_before": before, "telemetry_after": after})
  return {"schema": SCHEMA, "provenance_class": "generated_microbenchmark", "mode": mode,
          "system_snapshot_id": system_snapshot_id, "seed": seed, "randomized_order": [case.case_id for case in cases],
          "native_pmc_enabled": pmc_enabled, "sq_status": "live" if pmc_enabled else "blocked_profile_mode_precondition",
          "cases": rows, "candidate_timing_used_for_fit": False, "production_dispatch_changed": False}


def join_long_chain_modes(auto:Mapping[str, Any], profile:Mapping[str, Any]) -> dict[str, Any]:
  if auto.get("mode") != "auto" or profile.get("mode") != "profile_standard": raise ValueError("both modes are required")
  if auto.get("system_snapshot_id") != profile.get("system_snapshot_id"): raise ValueError("system snapshot mismatch")
  auto_rows, profile_rows = ({row["case_id"]: row for row in artifact["cases"]} for artifact in (auto, profile))
  if set(auto_rows) != set(profile_rows): raise ValueError("case identity mismatch")
  joined = []
  for case_id in auto_rows:
    a, p = auto_rows[case_id], profile_rows[case_id]
    if a["binary_sha256"] != p["binary_sha256"]: raise ValueError(f"{case_id} binary mismatch")
    cycles = p["counters"]["SQ_WAVE_CYCLES"] if p.get("counters") else None
    joined.append({"case_id": case_id, "family": a["family"], "chain_length": a["chain_length"],
                   "binary_sha256": a["binary_sha256"], "auto_median_ms": a["median_ms"],
                   "profile_median_ms": p["median_ms"], "profile_to_auto_ratio": p["median_ms"] / a["median_ms"],
                   "SQ_WAVES": p["counters"]["SQ_WAVES"] if p.get("counters") else None,
                   "SQ_WAVE_CYCLES": cycles, "SQ_BUSY_CYCLES": p["counters"]["SQ_BUSY_CYCLES"] if p.get("counters") else None,
                   "SQ_WAIT_ANY": p["counters"]["SQ_WAIT_ANY"] if p.get("counters") else None,
                   "aggregate_wave_cycles_per_wall_ns": cycles / (p["median_ms"] * 1e6) if cycles is not None else None})
  return {"schema": SCHEMA, "provenance_class": "generated_microbenchmark", "system_snapshot_id": auto["system_snapshot_id"],
          "modes": {"auto": auto, "profile_standard": profile}, "joined_cases": joined,
          "candidate_timing_used_for_fit": False, "production_dispatch_changed": False}


def validate_long_chain_calibration(artifact:Mapping[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  if artifact.get("provenance_class") != "generated_microbenchmark": raise ValueError("generated provenance required")
  if artifact.get("candidate_timing_used_for_fit") is not False: raise ValueError("candidate fitting is forbidden")
  if artifact.get("production_dispatch_changed") is not False: raise ValueError("production dispatch changed")
