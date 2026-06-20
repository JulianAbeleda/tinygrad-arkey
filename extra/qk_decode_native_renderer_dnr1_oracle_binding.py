#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinygrad.renderer.amd.schedule import (
  DecodeMMVQArtifactLaunchContract, DecodeMMVQInstructionContract, DecodeMMVQArtifactOracleBinding,
  decode_mmvq_artifact_oracle_binding_summary,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr1_oracle_binding_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def launch(row: dict[str, Any]) -> DecodeMMVQArtifactLaunchContract:
  return DecodeMMVQArtifactLaunchContract(
    runtime_name=str(row["runtime_name"]),
    global_size=tuple(int(x) for x in row["global_size"]),  # type: ignore[arg-type]
    local_size=tuple(int(x) for x in row["local_size"]),  # type: ignore[arg-type]
    kernarg_size=int(row["kernarg_size"]),
    group_segment_size=int(row["group_segment_size"]),
    private_segment_size=int(row["private_segment_size"]),
  )


def instruction(row: dict[str, Any]) -> DecodeMMVQInstructionContract:
  return DecodeMMVQInstructionContract(
    dot4=int(row["dot4"]),
    fma=int(row["fma"]),
    convert=int(row["convert"]),
    valu=int(row["valu"]),
    salu=int(row["salu"]),
    ds=int(row["ds"]),
    barrier=int(row["barrier"]),
    global_load=int(row["global_load"]),
    global_store=int(row["global_store"]),
    shuffle=int(row["shuffle"]),
    branch=int(row["branch"]),
    waitcnt=int(row["waitcnt"]),
  )


def main() -> int:
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  scope = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_project_scope_result.json")

  gates = loader.get("gates", {})
  binding = DecodeMMVQArtifactOracleBinding(
    producer=launch(loader["loader"]["producer"]),
    gateup=launch(loader["loader"]["gateup"]),
    work_decomposition=str(oracle["launch_contract"]["work_decomposition"]),
    instruction_contract=instruction(oracle["instruction_contract"]["oracle_grouped"]),
    correctness_passed=bool(gates.get("producer_correct") and gates.get("gate_correct") and gates.get("up_correct")),
    default_changed=bool(loader.get("default_changed")),
  )
  summary = decode_mmvq_artifact_oracle_binding_summary(binding)
  gate = binding.structural_gate()
  result = {
    "date": "2026-06-20",
    "phase": "DNR-1_DECODE_NATIVE_RENDERER_ORACLE_BINDING",
    "schema": "decode_native_renderer_dnr1_oracle_binding_v1",
    "verdict": "PASS_DNR1_DECODE_Q8_ORACLE_BINDING_STRUCTURAL" if gate["passed"] else "BLOCKED_DNR1_DECODE_Q8_ORACLE_BINDING_STRUCTURAL",
    "gate_pass": gate["passed"],
    "default_behavior_changed": False,
    "performance_claim": False,
    "binding": binding.to_dict(),
    "summary": summary,
    "input_artifacts": [
      "bench/q8-ffn-amd-scheduler-project/artifact_loader.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_project_scope_result.json",
    ],
    "project_scope_verdict": scope.get("verdict"),
    "next_action": "DNR-2 address/data-format lowering: emit a native runnable q8 gate/up candidate only after this binding remains green.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "gate_pass": gate["passed"], "summary": summary, "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0 if gate["passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
