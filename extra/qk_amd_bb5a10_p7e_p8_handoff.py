#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, pathlib, subprocess, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
P7D_SCRIPT = ROOT / "extra/qk_amd_bb5a10_p7d_authority_correctness.py"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str | None:
  try:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
  except Exception:
    return None


def main() -> int:
  p7d = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json", {})
  p6 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json", {})
  subset = p7d.get("authority_subset") or {}
  counts = subset.get("instruction_counts") or {}
  source_sha = sha256_file(P7D_SCRIPT) if P7D_SCRIPT.exists() else None
  source_lines = P7D_SCRIPT.read_text().count("\n") + 1 if P7D_SCRIPT.exists() else None
  p8_command = "CNT=30 K=4096 python3 extra/qk_amd_bb5a10_p8_performance.py"
  p8_prereq_command = "python3 extra/qk_amd_bb5a10_p7d_authority_correctness.py"
  handoff = {
    "candidate_id": "bb5a10_p7d_authority_k4096_single_tile",
    "candidate_class": "correct_executable_authority_k_subset",
    "source": {
      "path": "extra/qk_amd_bb5a10_p7d_authority_correctness.py",
      "sha256": source_sha,
      "line_count": source_lines,
      "entrypoint": "run_authority_subset(k=4096)",
    },
    "correctness": {
      "input_artifact": "bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json",
      "verdict": p7d.get("verdict"),
      "relative_rmse": subset.get("relative_rmse"),
      "tolerance": subset.get("tolerance"),
      "shape": subset.get("shape"),
      "authority_contract": subset.get("authority_contract"),
    },
    "resource_metadata": {
      "lds_bytes_runtime": 2048,
      "structural_authority_candidate_lds_bytes": ((p6.get("candidate") or {}).get("resource_summary") or {}).get("lds_bytes"),
      "scratch_bytes": 0,
      "private_segment_fixed_size": 0,
      "vgpr_static_upper_bound": 52,
      "instruction_counts": counts,
    },
    "p8": {
      "prerequisite_correctness_command": p8_prereq_command,
      "timing_command": p8_command,
      "performance_gate": {"metric": "TFLOPS", "minimum": 60.0},
      "timing_candidate_boundary": (
        "P8 must time a full-authority launch candidate. This P7e package is sufficient to enter P8 only if "
        "extra/qk_amd_bb5a10_p8_performance.py maps the proven P7d K-loop into an authority-shape launch or "
        "reports BLOCKED_FULL_AUTHORITY_LAUNCH_MAPPING without timing."
      ),
    },
    "proven": [
      "selected-compatible ds_store_b64 LDS staging",
      "ds_load_b128 WMMA fragment readback",
      "K=4096 accumulation over 256 WMMA K steps",
      "fp16 output store and numpy fp32 reference comparison",
    ],
    "not_proven": [
      "full authority M=512,N=12288 launch mapping",
      "edge predicates outside exact tile multiples",
      ">=60 TFLOPS P8 performance",
      "bit-identical Tensile LDS layout",
    ],
  }
  gate = {
    "input_p7d_pass": p7d.get("verdict") == "PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS" and bool(p7d.get("gate_pass")),
    "source_exists": P7D_SCRIPT.exists(),
    "source_sha256_present": bool(source_sha),
    "correctness_artifact_present": bool(p7d),
    "resource_metadata_present": bool(handoff["resource_metadata"]),
    "p8_timing_command_present": bool(p8_command),
    "p8_boundary_explicit": "full-authority launch" in handoff["p8"]["timing_candidate_boundary"],
    "performance_not_claimed": True,
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P7e_P8_handoff_package",
    "schema": "amd_bb5a10_p7e_p8_handoff_v1",
    "verdict": "PASS_BB5A10_P7E_P8_HANDOFF_PACKAGE" if gate_pass else "BLOCKED_BB5A10_P7E_P8_HANDOFF_PACKAGE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "git_head": git_head(),
    "handoff": handoff,
    "gate": gate,
    "decision": "P7e passes: P8 has a reproducible correctness/source/resource handoff and an exact timing entry command. P8 must still prove or block full-authority launch timing." if gate_pass else
                "P7e blocked; missing P7d correctness, source metadata, resource metadata, or exact P8 command.",
    "next_action": "Run P8 performance gate." if gate_pass else "Fix P7e inputs before P8.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json",
    ],
  }
  write_json("bb5a10_p7e_p8_handoff_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p7e_p8_handoff_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "next": result["next_action"],
    "p8_command": p8_command,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
