#!/usr/bin/env python3
"""Bounded guarded stage timing for an explicit two-program five-buffer artifact."""
from __future__ import annotations

import argparse, json, math, statistics, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from extra.qk.prefill.guarded_execution import GuardPolicy, make_tinygrad_guarded_hooks, run_guarded_execution
from extra.qk.prefill.host_safety_canary import make_tiny_health_probe
from extra.qk.prefill.isolated_guarded_executor import ExecutableBundle
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import AMD_ISA_TARGET, admitted_buffer_descriptors
from extra.qk.prefill.q4k_q8_five_buffer_execution_adapter import (
  load_q4k_q8_five_buffer_pipeline_npz, prepare_q4k_q8_five_buffer_pipeline_compile)
from tinygrad.runtime.process_isolated import run_isolated

SCHEMA = "tinygrad.q4k_q8_exact_binary_stage_timing.v1"
RUNTIME_DEVICE = "AMD"
MAX_WARMUPS, MAX_ROUNDS = 8, 32


def _finite_time(value: Any, name: str) -> float:
  if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
    raise ValueError(f"{name} dispatch did not return a finite non-negative device time")
  return float(value)


@dataclass
class _StageTimedExecutable:
  producer: Any
  mmq: Any
  values: Any
  scales: Any
  sums: Any
  device: str
  last_stage_times: dict[str, float] | None = None

  def dispatch(self, output, q4_packed_words, activation) -> float:
    args = lambda *buffers: tuple(buffer.get_buf(self.device) for buffer in buffers)
    producer = _finite_time(self.producer.dispatch(*args(self.values, self.scales, self.sums, activation)), "producer")
    mmq = _finite_time(self.mmq.dispatch(*args(output, q4_packed_words, self.values, self.scales, self.sums)), "mmq")
    total = producer + mmq
    self.last_stage_times = {"producer": producer, "mmq": mmq, "total": total}
    return total

  def close(self) -> None:
    for buffer in (self.values, self.scales, self.sums):
      if buffer.is_allocated(): buffer.deallocate()
    self.producer.close(); self.mmq.close()


def build_stage_timed_bundle(*, payload: dict[str, Any], canonical_identity: str,
                             compile_evidence: Mapping[str, Any], compile_target: str = AMD_ISA_TARGET,
                             runtime_device: str = RUNTIME_DEVICE) -> ExecutableBundle:
  """Child-only builder that rejects binary/resource drift before either dispatch."""
  pipeline, child = prepare_q4k_q8_five_buffer_pipeline_compile(payload, canonical_identity, target=compile_target)
  contract = compile_evidence.get("child_recompile_binary_identity_contract")
  required = ("canonical_identity", "abi_digest", "compile_target", "target", "source_sha256", "binary_sha256",
    "producer_source_sha256", "producer_binary_sha256", "producer_resource_summary",
    "pipeline_binary_sha256", "program_count", "execution_input_format")
  if not isinstance(contract, Mapping) or contract.get("enabled") is not True or \
     contract.get("reject_sha256_mismatch_before_dispatch") is not True:
    raise ValueError("exact-binary child recompile contract is missing")
  for field in required:
    if child.get(field) != compile_evidence.get(field) or contract.get(field) != compile_evidence.get(field):
      raise ValueError(f"exact-binary child {field} differs from admitted parent compile")
  if child.get("resource_summary") != compile_evidence.get("resource_summary"):
    raise ValueError("exact-binary child resource_summary differs from admitted parent compile")

  from tinygrad.device import Buffer
  from tinygrad.runtime.bridge import prepare_executable
  descriptors = {row.name: row for row in admitted_buffer_descriptors(pipeline.admission)}
  buffers = [Buffer(runtime_device, math.prod(descriptors[name].flat_shape), descriptors[name].dtype, preallocate=True)
             for name in ("q8_ds4_values", "q8_scales", "q8_weighted_sums")]
  producer_evidence = {"passed": True, "binary_sha256": child["producer_binary_sha256"]}
  executable = _StageTimedExecutable(prepare_executable(pipeline.producer, producer_evidence, device=runtime_device),
    prepare_executable(pipeline.mmq, child, device=runtime_device), *buffers, runtime_device)

  def dispatch(target, guarded):
    try: values = tuple(guarded[name].resource["payload"] for name in ("output", "q4_packed_words", "activation"))
    except KeyError as exc: raise ValueError(f"pipeline ABI buffer is missing: {exc.args[0]}") from exc
    return target.dispatch(*values)
  return ExecutableBundle(executable, make_tinygrad_guarded_hooks(runtime_device, dispatch, lambda: True))


def _child_run(payload: dict[str, Any], canonical_identity: str, compile_evidence: Mapping[str, Any],
               inputs: Mapping[str, np.ndarray], reference: np.ndarray, policy: GuardPolicy) -> dict[str, Any]:
  try: bundle = build_stage_timed_bundle(payload=payload, canonical_identity=canonical_identity,
                                          compile_evidence=compile_evidence)
  except BaseException as exc:
    return {"status": "failed", "passed": False, "dispatch_performed": False, "device_fault": False,
            "errors": [f"runtime construction failed: {type(exc).__name__}: {exc}"]}
  try:
    result = run_guarded_execution(executable=bundle.executable, inputs=inputs, reference=reference,
      hooks=bundle.hooks, policy=policy, identity={"canonical_identity": canonical_identity}, output_dtype=np.float32)
    if result.get("passed"):
      stages = bundle.executable.last_stage_times
      if not isinstance(stages, dict) or stages.get("total") != stages.get("producer", 0) + stages.get("mmq", 0):
        result = {**result, "status": "failed", "passed": False,
                  "errors": [*result.get("errors", ()), "stage timing attribution is unavailable or inconsistent"]}
      else: result = {**result, "stage_device_seconds": dict(stages)}
    return result
  finally:
    try: bundle.executable.close()
    except Exception: pass


def _isolated_launch(*, payload: dict[str, Any], canonical_identity: str, compile_evidence: Mapping[str, Any],
                     inputs: Mapping[str, np.ndarray], reference: np.ndarray, policy: GuardPolicy,
                     health_probe: Any = None) -> dict[str, Any]:
  child = run_isolated(_child_run, args=(payload, canonical_identity, compile_evidence, inputs, reference, policy),
                       timeout_seconds=policy.timeout_seconds, start_method="spawn")
  if child.timed_out or child.status != "passed" or not isinstance(child.result, Mapping):
    return {"passed": False, "dispatch_state": "timed_out" if child.timed_out else "device_lost",
            "errors": [child.error or "isolated child produced no result"]}
  guarded = dict(child.result)
  healthy = True
  if health_probe is not None:
    probe = run_isolated(lambda_probe, args=(health_probe,), timeout_seconds=10, start_method="spawn")
    healthy = probe.status == "passed" and probe.result is True
  passed = guarded.get("passed") is True and healthy
  return {"passed": passed, "dispatch_state": "completed" if passed else ("device_lost" if not healthy else "failed"),
          "errors": list(guarded.get("errors", ())), "guarded": guarded, "health_after": healthy}


def lambda_probe(probe: Any) -> bool:
  """Module-level spawn target for the independent health primitive."""
  return bool(probe())


def _candidate(path: Path) -> tuple[dict[str, Any], str]:
  row = json.loads(path.read_text())
  if set(row) != {"payload", "canonical_identity"} or not isinstance(row["payload"], dict) or \
     not isinstance(row["canonical_identity"], str):
    raise ValueError("candidate JSON must contain exactly payload and canonical_identity")
  return row["payload"], row["canonical_identity"]


def run(args: argparse.Namespace, *, launch=_isolated_launch) -> dict[str, Any]:
  if not 0 <= args.warmups <= MAX_WARMUPS or not 1 <= args.rounds <= MAX_ROUNDS:
    raise ValueError(f"bounded timing requires warmups <= {MAX_WARMUPS} and rounds in 1..{MAX_ROUNDS}")
  payload, identity = _candidate(Path(args.candidate))
  pipeline, evidence = prepare_q4k_q8_five_buffer_pipeline_compile(payload, identity)
  inputs, reference, detail = load_q4k_q8_five_buffer_pipeline_npz(args.input_npz, pipeline.admission)
  evidence = dict(evidence)
  evidence.update(input_identity="sha256:" + detail["input_artifact_sha256"],
                  reference_identity="sha256:" + detail["reference_sha256"],
                  content_identities=dict(detail["content_sha256"]), input_identity_detail=detail)
  policy = GuardPolicy(timeout_seconds=args.timeout_seconds, rtol=args.rtol, atol=args.atol)
  health = make_tiny_health_probe(device=RUNTIME_DEVICE)
  launches = []
  for index in range(1 + args.warmups + args.rounds):
    outcome = launch(payload=payload, canonical_identity=identity, compile_evidence=evidence,
      inputs=inputs, reference=reference, policy=policy, health_probe=health)
    if not outcome.get("passed"):
      return {"schema": SCHEMA, "status": "failed", "passed": False, "failed_launch": index,
              "compile_evidence": evidence, "outcome": outcome}
    if index: launches.append(dict(outcome["guarded"]["stage_device_seconds"]))
  samples = launches[args.warmups:]
  for row in samples:
    if row["total"] != row["producer"] + row["mmq"]: raise ValueError("stage total is not the stage sum")
  summary = {name: statistics.median(row[name] for row in samples) for name in ("producer", "mmq", "total")}
  return {"schema": SCHEMA, "status": "passed", "passed": True,
    "identity": {"canonical_identity": identity, "pipeline_binary_sha256": evidence["pipeline_binary_sha256"],
      "producer_binary_sha256": evidence["producer_binary_sha256"], "mmq_binary_sha256": evidence["binary_sha256"],
      "input_identity": evidence["input_identity"], "reference_identity": evidence["reference_identity"]},
    "resources": {"producer": evidence["producer_resource_summary"], "mmq": evidence["resource_summary"]},
    "measurement": {"units": "s", "source": "synchronized_device_dispatch", "warmups": args.warmups,
      "rounds": args.rounds, "statistic": "median", "samples": samples, "median": summary,
      "total_definition": "producer + mmq per measured launch"}}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", required=True); parser.add_argument("--input-npz", required=True)
  parser.add_argument("--output", required=True); parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--rounds", type=int, default=3); parser.add_argument("--timeout-seconds", type=float, default=120)
  parser.add_argument("--rtol", type=float, default=0.0); parser.add_argument("--atol", type=float, default=0.0)
  args = parser.parse_args()
  try: report = run(args)
  except BaseException as exc:
    report = {"schema": SCHEMA, "status": "blocked", "passed": False,
              "blocker": {"type": type(exc).__name__, "message": str(exc)}}
  target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, sort_keys=True, separators=(",", ":")))
  return 0 if report.get("passed") else 1


if __name__ == "__main__": raise SystemExit(main())
