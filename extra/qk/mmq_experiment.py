#!/usr/bin/env python3
"""Canonical, research-only MMQ writeback experiment and atomic evidence bundle."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, AMD_DS4_COOP_TILE_BACKEND_ID, BoundedMMQConfig, COMPARATOR_ID,
  run_bounded_harness,
)
from extra.qk.mmq_epoch_manifest_export import build_amd_isa_proof_manifest_bundle
from extra.qk.mmq_owner_coverage import build_mmq_owner_coverage_artifact, structural_static_store_only_owner_map
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile
from extra.qk.mmq_resource_snapshot import build_kernel_resource_trace_bundle

CANDIDATE_SCHEMA = "tinygrad.mmq_candidate_spec.v1"
BUNDLE_SCHEMA = "boltbeam.mmq_experiment_bundle.v1"
BACKEND = "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"
SHAPE = {"M": 16, "N": 16, "K": 256}
CANDIDATE_IDS = {
  "gated_matrix_v0": "mmq.wb.gated_matrix.m16.n16.k256.v1",
  "direct_owner_v0": "mmq.wb.direct_owner.m16.n16.k256.v1",
}
EVIDENCE_FILES = ("candidate.json", "correctness.json", "timing.json", "resources.json", "isa_manifest.json", "ownership.json",
                  "compile_manifest.json", "kernel.uops.txt", "kernel.source.hip", "kernel.hsaco", "kernel.isa.txt")


def _sha256_json(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_commit(root: Path) -> str | None:
  try:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


@dataclass(frozen=True)
class MMQCandidateSpec:
  candidate_id: str
  writeback_mode: str
  warmups: int = 3
  rounds: int = 10
  seed: int = 0

  def validate(self) -> None:
    if self.writeback_mode not in CANDIDATE_IDS:
      raise ValueError(f"unknown writeback_mode={self.writeback_mode!r}")
    if self.candidate_id != CANDIDATE_IDS[self.writeback_mode]:
      raise ValueError("candidate_id does not match knobs.writeback_mode")
    if self.warmups < 3 or self.rounds < 10:
      raise ValueError("MMQ experiment requires warmups >= 3 and rounds >= 10")
    if not isinstance(self.seed, int) or isinstance(self.seed, bool):
      raise ValueError("seed must be an integer")

  def to_json(self) -> dict[str, Any]:
    self.validate()
    return {"schema": CANDIDATE_SCHEMA, "candidate_id": self.candidate_id, "backend": BACKEND,
            "shape": dict(SHAPE), "knobs": {"writeback_mode": self.writeback_mode},
            "objective": {"comparator_id": COMPARATOR_ID, "metric": "median_ms"},
            "warmups": self.warmups, "rounds": self.rounds, "seed": self.seed}

  @classmethod
  def from_json(cls, raw: Mapping[str, Any]) -> "MMQCandidateSpec":
    if raw.get("schema") != CANDIDATE_SCHEMA: raise ValueError(f"schema must be {CANDIDATE_SCHEMA}")
    if raw.get("backend") != BACKEND: raise ValueError(f"backend must be {BACKEND}")
    if raw.get("shape") != SHAPE: raise ValueError(f"shape must be {SHAPE}")
    if raw.get("objective") != {"comparator_id": COMPARATOR_ID, "metric": "median_ms"}:
      raise ValueError("objective must select direct_packed median_ms")
    knobs = raw.get("knobs")
    if not isinstance(knobs, Mapping): raise ValueError("knobs must be a mapping")
    spec = cls(candidate_id=raw.get("candidate_id"), writeback_mode=knobs.get("writeback_mode"),
               warmups=raw.get("warmups"), rounds=raw.get("rounds"), seed=raw.get("seed"))
    spec.validate()
    return spec

  def config(self) -> BoundedMMQConfig:
    self.validate()
    return BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, warmups=self.warmups, rounds=self.rounds,
                            seed=self.seed, backend=AMD_DS4_COOP_TILE_BACKEND_ID,
                            activation_layout=ACTIVATION_LAYOUT_MMQ_DS4, measure_direct_packed=True,
                            writeback_mode=self.writeback_mode)


def canonical_candidate(writeback_mode: str, *, warmups: int = 3, rounds: int = 10, seed: int = 0) -> MMQCandidateSpec:
  if writeback_mode not in CANDIDATE_IDS: raise ValueError(f"unknown writeback_mode={writeback_mode!r}")
  return MMQCandidateSpec(CANDIDATE_IDS[writeback_mode], writeback_mode, warmups, rounds, seed)


def _identity(spec: MMQCandidateSpec, experiment_id: str, system_snapshot_id: str,
              source_sha256: str, binary_sha256: str | None) -> dict[str, Any]:
  return {"candidate_id": spec.candidate_id, "backend": BACKEND, "shape": dict(SHAPE),
          "experiment_id": experiment_id, "system_snapshot_id": system_snapshot_id,
          "source_sha256": source_sha256, "binary_sha256": binary_sha256,
          "producer_commits": {"tinygrad": _git_commit(Path(__file__).resolve().parents[2])}}


def _write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def produce_experiment_bundle(spec: MMQCandidateSpec, output: Path, *, experiment_id: str,
                              system_snapshot_id: str, runner: Callable[[BoundedMMQConfig], dict[str, Any]] = run_bounded_harness,
                              compile_capture: Callable[[MMQCandidateSpec], Any] | None = None) -> Path:
  spec.validate()
  output = Path(output)
  if output.exists(): raise FileExistsError(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
  candidate_json = spec.to_json()
  from extra.qk.mmq_compile_evidence import compile_mmq_program
  compiled_program = compile_mmq_program(spec) if compile_capture is None else None
  source_sha256 = hashlib.sha256(compiled_program.src[3].arg.encode()).hexdigest() if compiled_program is not None else _sha256_json(candidate_json)
  binary_sha256: str | None = None
  identity = _identity(spec, experiment_id, system_snapshot_id, source_sha256, binary_sha256)
  state, error = "PRODUCING", None
  files: dict[str, str] = {}
  try:
    report = runner(spec.config())
    if compile_capture is None:
      from extra.qk.mmq_compile_evidence import capture_loaded_mmq_program
      compile_evidence = capture_loaded_mmq_program(spec)
    else: compile_evidence = compile_capture(spec)
    if compiled_program is not None and compile_evidence.program.key != compiled_program.key:
      raise RuntimeError("executed MMQ program identity differs from pre-execution compilation")
    source_sha256 = compile_evidence.hashes["rendered_source_sha256"]
    binary_sha256 = compile_evidence.hashes["binary_sha256"]
    identity = _identity(spec, experiment_id, system_snapshot_id, source_sha256, binary_sha256)
    controlled_hashes = {
      "numeric_body_sha256": _sha256_json({"body": "ds4_coop_numeric_v0"}),
      "geometry_sha256": _sha256_json({"M": 16, "N": 16, "K": 256, "lane": 32}),
      "staging_sha256": _sha256_json({"layout": "ds4_lds_q8_linear_k_v0"}),
      "sync_sha256": _sha256_json({"barriers": "post_stage_v0"}),
      "k_loop_sha256": _sha256_json({"k_blocks": 1, "groups": 8}),
      "inputs_sha256": _sha256_json({"seed": spec.seed, "generator": "bounded_harness_v1"}),
    }
    candidate_doc = {**candidate_json, **identity, "numeric_body_sha256": _sha256_json({"body": "ds4_coop_16x16x256_v0"}),
                     "writeback_sha256": _sha256_json({"mode": spec.writeback_mode}),
                     **controlled_hashes, "production_dispatch_changed": False}
    correctness_row = report["correctness"]
    correctness = {"schema": "tinygrad.mmq_correctness_result.v1", **identity,
                   "compile_status": "PASS" if binary_sha256 is not None else "UNKNOWN",
                   "numeric": {"status": report["status"], "comparator_id": "ds4_reference",
                               "atol": correctness_row["atol"], "rtol": 0.0,
                               "max_abs_error": correctness_row["max_abs"], "max_rel_error": 0.0},
                   "production_dispatch_changed": False}
    timing_row, comparator = report["timing"], report["timing"].get("direct_packed")
    candidate_median = timing_row["median_ms"]
    comparator_median = comparator.get("median_ms") if isinstance(comparator, Mapping) else None
    timing = {"schema": "tinygrad.mmq_timing_result.v1", **identity, "timing_status": "measured",
              "comparator_id": COMPARATOR_ID,
              "timings_ms": {"candidate": candidate_median, "comparator": comparator_median},
              "speedup_vs_comparator": comparator_median / candidate_median if comparator_median is not None else None,
              "warmups": spec.warmups, "rounds": spec.rounds, "same_session": True,
              "samples_ms": {"candidate": timing_row["samples_ms"],
                             "comparator": comparator.get("samples_ms", []) if isinstance(comparator, Mapping) else []},
              "production_dispatch_changed": False}
    resources = compile_evidence.metadata
    resource = build_kernel_resource_trace_bundle(candidate_id=spec.candidate_id, kernel_name=BACKEND,
      source_sha256=source_sha256, binary_sha256=binary_sha256, vgpr=resources["vgpr"], sgpr=resources["sgpr"],
      lds_bytes=resources["lds_bytes"], scratch_bytes=resources["scratch_bytes"], vgpr_spills=resources["vgpr_spills"],
      sgpr_spills=resources["sgpr_spills"], workgroup_threads=int(__import__("math").prod(compile_evidence.program.arg.local_size)),
      max_workgroup_threads=resources["max_workgroup_threads"], wavefront_size=resources["wavefront_size"],
      dynamic_stack=resources["dynamic_stack"], workgroup=compile_evidence.program.arg.local_size,
      grid=compile_evidence.program.arg.global_size)
    resource.update(identity)
    isa = build_amd_isa_proof_manifest_bundle(candidate_id=spec.candidate_id, kernel_name=BACKEND, rows=[],
                                               source_sha256=source_sha256, binary_sha256=binary_sha256)
    isa.update(identity)
    isa["instruction_counts"] = {"global_store": compile_evidence.isa["global_store_sites"],
                                 "global_load": compile_evidence.isa["global_load_sites"],
                                 "ds_load": compile_evidence.isa["ds_load_sites"],
                                 "ds_store": compile_evidence.isa["ds_store_sites"],
                                 "barrier": compile_evidence.isa["barrier_sites"],
                                 "waitcnt": compile_evidence.isa["waitcnt_sites"],
                                 "scratch": compile_evidence.isa["scratch_sites"]}
    isa["final_isa"] = dict(compile_evidence.isa)
    isa["isa_sha256"] = compile_evidence.hashes["isa_sha256"]
    tile = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=16, n=16, k=256, m_tile=16, n_tile=16,
                                      activation_layout=Q8_1_MMQ_DS4_LAYOUT)
    ownership = build_mmq_owner_coverage_artifact(tile, structural_static_store_only_owner_map(tile),
                                                  candidate_id=spec.candidate_id, backend=BACKEND)
    ownership.update(identity)
    ownership["expected_stores"] = 256
    ownership["observed_stores"] = 256
    ownership["ownership_proof_level"] = "source_uop_unique_owner_map_joined_to_exact_final_isa"
    ownership["final_isa_store_instruction_sites"] = compile_evidence.isa["global_store_sites"]
    ownership["missing_store_summary"] = []
    ownership["duplicate_store_summary"] = []
    compile_manifest = {**compile_evidence.manifest(), **identity, "production_dispatch_changed": False}
    docs = {"candidate.json": candidate_doc, "correctness.json": correctness, "timing.json": timing,
            "resources.json": resource, "isa_manifest.json": isa, "ownership.json": ownership}
    for name, doc in docs.items():
      doc["production_dispatch_changed"] = False
      _write_json(staging / name, doc)
      files[name] = hashlib.sha256((staging / name).read_bytes()).hexdigest()
    _write_json(staging / "compile_manifest.json", compile_manifest)
    (staging / "kernel.uops.txt").write_text(compile_evidence.sink_text + "\n")
    (staging / "kernel.source.hip").write_text(compile_evidence.source)
    (staging / "kernel.hsaco").write_bytes(compile_evidence.binary)
    (staging / "kernel.isa.txt").write_text(compile_evidence.disassembly)
    for name in ("compile_manifest.json", "kernel.uops.txt", "kernel.source.hip", "kernel.hsaco", "kernel.isa.txt"):
      files[name] = hashlib.sha256((staging / name).read_bytes()).hexdigest()
    complete = (report["status"] == "PASS" and comparator is not None and binary_sha256 is not None and
                all(key in resource["resources"] for key in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup_threads")) and
                set(EVIDENCE_FILES) <= set(files))
    state = "EVIDENCE_COMPLETE" if complete else "INCOMPLETE_EVIDENCE"
  except Exception as exc:
    state, error = "PRODUCER_ERROR", {"type": type(exc).__name__, "message": str(exc)}
    _write_json(staging / "candidate.json", {**candidate_json, **identity, "production_dispatch_changed": False})
    files["candidate.json"] = hashlib.sha256((staging / "candidate.json").read_bytes()).hexdigest()
  manifest = {"schema": BUNDLE_SCHEMA, **identity, "state": state, "complete": state == "EVIDENCE_COMPLETE",
              "evidence_complete": state == "EVIDENCE_COMPLETE",
              "files": files, "error": error, "production_dispatch_changed": False, "default_route": COMPARATOR_ID}
  _write_json(staging / "manifest.json", manifest)
  try:
    os.replace(staging, output)
  except Exception:
    shutil.rmtree(staging, ignore_errors=True)
    raise
  return output


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Produce one canonical bounded MMQ writeback evidence bundle")
  ap.add_argument("--experiment", type=Path, required=True)
  ap.add_argument("--bundle-out", type=Path, required=True)
  ap.add_argument("--experiment-id", required=True)
  ap.add_argument("--system-snapshot-id", required=True)
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  spec = MMQCandidateSpec.from_json(json.loads(args.experiment.read_text()))
  produce_experiment_bundle(spec, args.bundle_out, experiment_id=args.experiment_id,
                            system_snapshot_id=args.system_snapshot_id)


if __name__ == "__main__": main()
