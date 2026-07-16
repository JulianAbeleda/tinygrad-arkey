#!/usr/bin/env python3
"""Guarded Phase 3 evidence runner for one exact Q4_K real-role workload."""
from __future__ import annotations

import argparse, json, platform, subprocess, sys, time
from pathlib import Path
from typing import Any

import numpy as np

from extra.qk.prefill.current_prefill_execution_adapter import ADAPTER_ID as DIRECT_ADAPTER_ID, CurrentPrefillAdapter
from extra.qk.prefill.execution_bridge_contracts import (CorrectnessProtocol, ExecutionRequest, GuardProtocol,
  TimingProtocol, TransportPlan, canonical_digest)
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry, execute_session
from extra.qk.prefill.q4k_q8_five_buffer_artifact import build_q4k_q8_five_buffer_artifact
from extra.qk.prefill.q4k_q8_five_buffer_execution_adapter import (ADAPTER_ID as PIPELINE_ADAPTER_ID,
  Q4KQ8FiveBufferAdapter, prepare_q4k_q8_five_buffer_pipeline_compile)
from extra.qk.prefill.q4k_q8_five_buffer_role_gate import admitted_q4k_non_fitting_roles
from extra.qk.runtime_specs import (FullKernelCandidateSet, capability_transport, full_kernel_candidate_capability,
  full_kernel_workload)

SCHEMA = "tinygrad.q4k_q8_phase3_role_evidence.v1"
DEFAULT_INVENTORY = "bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json"
DEFAULT_ROLE, DEFAULT_SHAPE = "attn_kv", (512, 1024, 5120)


def _health() -> dict[str, Any]:
  command = ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuse", "--showtemp"]
  started = time.monotonic()
  try: completed = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
  except (OSError, subprocess.TimeoutExpired) as exc:
    return {"command": command, "ok": False, "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started}
  return {"command": command, "ok": completed.returncode == 0, "returncode": completed.returncode,
          "stdout": completed.stdout, "stderr": completed.stderr, "elapsed_seconds": time.monotonic() - started}


def _select(inventory: dict[str, Any], role: str, shape: tuple[int, int, int]):
  matches = [(entry, admission) for entry, admission in admitted_q4k_non_fitting_roles(inventory)
             if (full_kernel_workload(admission.normalized_payload).role,
                 full_kernel_workload(admission.normalized_payload).shape) == (role, shape)]
  if len(matches) != 1: raise ValueError(f"expected one admitted Q4_K obligation for {role} {shape}, got {len(matches)}")
  physical, admission = matches[0]
  raw = FullKernelCandidateSet.from_json(inventory["candidate_sets"]["Q4_K"])
  original = [entry for entry in raw.entries if full_kernel_workload(entry.payload).role == role and
              full_kernel_workload(entry.payload).shape == shape]
  if len(original) != 1: raise ValueError(f"expected one direct-packed comparator for {role} {shape}, got {len(original)}")
  return physical, admission, original[0]


def _fixtures(directory: Path, shape: tuple[int, int, int], seed: int) -> tuple[Path, Path, dict[str, Any]]:
  directory.mkdir(parents=True, exist_ok=True)
  m, n, k = shape
  artifact = build_q4k_q8_five_buffer_artifact(m, n, k, seed=seed)
  positions = np.asarray(artifact.metadata["selected_positions"], dtype=np.int64)
  coefficients = np.asarray(artifact.metadata["coefficients_fp32"], dtype=np.float32)
  activation = np.zeros((m, k), dtype=np.float32)
  activation[np.arange(m), positions] = coefficients
  pipeline = directory / "pipeline-input.npz"
  direct = directory / "direct-packed-input.npz"
  np.savez(pipeline, q4_packed_words=artifact.q4_packed_words, activation=activation.reshape(-1),
           reference=artifact.reference)
  np.savez(direct, a=activation.astype(np.float16), b=artifact.q4_packed_words,
           reference=artifact.reference.astype(np.float16))
  return pipeline, direct, artifact.metadata


def _request(*, candidate_id: str, comparator_id: str, adapter_id: str, entry, input_npz: Path,
             workload_digest: str, session_id: str, timeout_ms: int, warmups: int, rounds: int,
             input_format: str | None = None) -> ExecutionRequest:
  workload = full_kernel_workload(entry.payload)
  schedule_digest = canonical_digest(entry.payload["schedule"], "schedule")
  transport = capability_transport(full_kernel_candidate_capability(entry.payload))
  compiler = {"adapter_id": adapter_id, "candidate_payload": entry.payload,
              "canonical_identity": entry.canonical_identity, "input_npz": str(input_npz.resolve())}
  if input_format is not None: compiler["input_format"] = input_format
  return ExecutionRequest(experiment_id="q4k-q8-phase3-role-gate", candidate_id=candidate_id,
    comparator_id=comparator_id, workload_digest=workload_digest, schedule_digest=schedule_digest,
    transport_plan=TransportPlan(transport, schedule_digest),
    target_context={"session_id": session_id, "target_id": "AMD:gfx1100:wave32",
      "system_snapshot_id": platform.node(), "workload": {"role": workload.role, "shape": list(workload.shape)}},
    compiler_context=compiler, correctness=CorrectnessProtocol("sparse_selected_position", atol=0.0, rtol=0.0),
    guard=GuardProtocol(timeout_ms), timing=TimingProtocol(warmups, rounds, 0, same_session=True))


def run(args: argparse.Namespace) -> dict[str, Any]:
  command = [sys.executable, *sys.argv]
  inventory_path = Path(args.inventory)
  inventory = json.loads(inventory_path.read_text())
  shape = (args.m, args.n, args.k)
  physical, _, direct = _select(inventory, args.role, shape)
  health_before = _health()
  if not health_before["ok"]: raise RuntimeError("GPU health preflight failed")
  # Compile before allocating role-sized fixtures or permitting dispatch.
  _, compile_gate = prepare_q4k_q8_five_buffer_pipeline_compile(physical.payload, physical.canonical_identity)
  pipeline_npz, direct_npz, fixture = _fixtures(Path(args.workdir), shape, args.seed)
  workload_digest = canonical_digest({"role": args.role, "shape": list(shape), "quant_format": "Q4_K"}, "workload")
  session_id = f"phase3-{args.role}-{args.m}x{args.n}x{args.k}-seed{args.seed}"
  candidate = _request(candidate_id="five_buffer_activation_prep_inclusive", comparator_id="direct_packed",
    adapter_id=PIPELINE_ADAPTER_ID, entry=physical, input_npz=pipeline_npz, input_format="fp32_activation",
    workload_digest=workload_digest, session_id=session_id, timeout_ms=args.timeout_ms,
    warmups=args.warmups, rounds=args.rounds)
  comparator = _request(candidate_id="direct_packed", comparator_id="five_buffer_activation_prep_inclusive",
    adapter_id=DIRECT_ADAPTER_ID, entry=direct, input_npz=direct_npz, workload_digest=workload_digest,
    session_id=session_id, timeout_ms=args.timeout_ms, warmups=args.warmups, rounds=args.rounds)
  registry = AdapterRegistry(); registry.register(PIPELINE_ADAPTER_ID, Q4KQ8FiveBufferAdapter())
  registry.register(DIRECT_ADAPTER_ID, CurrentPrefillAdapter())
  results = [row.to_dict() for row in execute_session((candidate, comparator), registry=registry, session_id=session_id)]
  health_after = _health()
  passed = health_after["ok"] and all(all(phase["status"] == "passed" for phase in row["phases"]) for row in results)
  return {"schema": SCHEMA, "status": "pass" if passed else "fail", "passed": passed, "command": command,
    "git": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "status_short": subprocess.check_output(["git", "status", "--short"], text=True).splitlines()},
    "session_id": session_id, "role": args.role, "shape": {"M": args.m, "N": args.n, "K": args.k},
    "measurement_definition": {"candidate": "physical DS4 activation preparation + five-buffer MMQ dispatch",
      "comparator": "direct-packed contraction dispatch", "same_session": True,
      "warmups": args.warmups, "rounds": args.rounds, "statistic": "median"},
    "health": {"before": health_before, "after": health_after}, "compile_gate": compile_gate,
    "fixture": fixture, "results": results}


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--inventory", default=DEFAULT_INVENTORY)
  parser.add_argument("--role", default=DEFAULT_ROLE); parser.add_argument("--m", type=int, default=DEFAULT_SHAPE[0])
  parser.add_argument("--n", type=int, default=DEFAULT_SHAPE[1]); parser.add_argument("--k", type=int, default=DEFAULT_SHAPE[2])
  parser.add_argument("--seed", type=int, default=0); parser.add_argument("--timeout-ms", type=int, default=120000)
  parser.add_argument("--warmups", type=int, default=1); parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--workdir", default="/tmp/tinygrad-q4k-q8-phase3-attn-kv")
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  try: report = run(args)
  except BaseException as exc:
    report = {"schema": SCHEMA, "status": "blocked", "passed": False, "command": [sys.executable, *sys.argv],
              "blocker": {"type": type(exc).__name__, "message": str(exc)}, "health_after_blocker": _health()}
  target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, sort_keys=True, separators=(",", ":")))
  return 0 if report.get("passed") else 1


if __name__ == "__main__": raise SystemExit(main())
