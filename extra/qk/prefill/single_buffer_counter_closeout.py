#!/usr/bin/env python3
"""Same-binary native-PMC closeout for the pure single-buffer candidate."""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys
from contextlib import contextmanager
from typing import Any, Callable

from extra.qk.mmq_amd_pmc import collect_kernel_pmc, _decode_event
from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "prefill-single-buffer-counter-closeout.v1"
GROUPS = {
  "compute_wait": ("SQ_BUSY_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_WAIT_ANY"),
  "lds": ("SQC_LDS_IDX_ACTIVE", "SQC_LDS_BANK_CONFLICT", "SQ_INSTS_LDS", "SQ_WAIT_INST_LDS"),
  "l2_hbm_proxy": ("GL2C_HIT", "GL2C_MISS", "GL2C_MC_RDREQ", "GL2C_MC_WRREQ"),
  "vram_wavefront_proxy": ("TA_BUFFER_LOAD_WAVEFRONTS", "TA_BUFFER_STORE_WAVEFRONTS"),
}
POWER = pathlib.Path("/sys/class/drm/card0/device/power_dpm_force_performance_level")


def _git() -> dict[str, Any]:
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
  return {"revision": revision, "dirty": dirty}


@contextmanager
def _restore_gpu_policy(path: pathlib.Path = POWER):
  original = path.read_text().strip() if path.exists() and os.access(path, os.R_OK) else None
  try: yield
  finally:
    if original is not None and os.access(path, os.W_OK): path.write_text(original + "\n")


def _resource_join(row: dict[str, Any], identity: str) -> tuple[str, str]:
  if row.get("canonical_identity") != identity or row.get("passed") is not True:
    raise ValueError("compiled-resource artifact is not a passing exact-candidate authority")
  binary = (row.get("program") or {}).get("binary_sha256")
  commit = (row.get("git") or {}).get("revision")
  if not isinstance(binary, str) or len(binary) != 64: raise ValueError("resource binary identity is absent")
  if not isinstance(commit, str) or len(commit) != 40: raise ValueError("resource commit identity is absent")
  return binary, commit


def build_report(payload: dict[str, Any], candidate_hash: str, resource: dict[str, Any], *, repetitions: int = 3,
                 collector: Callable[..., dict[str, Any]] = collect_kernel_pmc,
                 command: list[str] | None = None, git_state: dict[str, Any] | None = None) -> dict[str, Any]:
  identity = canonical_candidate_hash(payload)
  if candidate_hash != identity: raise ValueError("candidate hash does not match payload")
  binary, commit = _resource_join(resource, identity)
  git = git_state or _git()
  blockers = []
  if git.get("revision") != commit: blockers.append("current source commit does not match compiled-resource authority")
  if git.get("dirty") is not False: blockers.append("current worktree is dirty")
  rows = []
  if not blockers:
    if command is None: raise ValueError("candidate PMC child command is required")
    child = command
    for category, counters in GROUPS.items():
      row = collector({"candidate_id": identity, "backend": "AMD", "shape": payload["workload"]["shape"]},
                      counters, repetitions, command=child, system_snapshot_id=commit,
                      binary_sha256=binary)
      samples = row.get("samples", [])
      live = bool(samples) and all(s.get("status") == "live" and s.get("binary_sha256") == binary and
                                   all(isinstance(s.get("counters", {}).get(c), int) for c in counters) for s in samples)
      rows.append({"category": category, "status": "live" if live else "blocked", "authority": row})
      if not live: blockers.append(f"{category} counters are unavailable, incomplete, or binary-mismatched")
  unavailable = [
    {"category": "wmma", "status": "blocked", "reason": "gfx1100 native PMC exposes no direct WMMA instruction counter"},
    {"category": "occupancy", "status": "blocked", "reason": "native PMC exposes waves/cycles, not a calibrated occupancy metric"},
  ]
  blockers += [f"{x['category']} counter authority unavailable" for x in unavailable]
  return {"schema": SCHEMA, "status": "pass" if not blockers else "blocked",
          "joins": {"canonical_identity": identity, "binary_sha256": binary, "commit": commit},
          "gpu_policy": {"restoration": "original performance policy restored in finally"},
          "groups": rows, "unavailable": unavailable, "blockers": blockers}


def _child(payload_path: pathlib.Path, candidate_hash: str) -> int:
  from tinygrad.device import Compiled, Device
  from extra.qk.prefill.single_buffer_execution_authority import run
  payload = json.loads(payload_path.read_text())
  Compiled.profile_events.clear()
  authority = run(payload, candidate_hash, case="constant")
  Device[Device.DEFAULT].synchronize()
  events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"]
  result = {"status": "live" if authority.get("passed") and events else "blocked",
            "binary_sha256": (authority.get("program") or {}).get("binary_sha256"),
            "counters": _decode_event(events[-1]) if events else {}}
  print("MMQ_PMC_JSON=" + json.dumps(result, sort_keys=True))
  return 0 if result["status"] == "live" else 1


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--payload", type=pathlib.Path, required=True); ap.add_argument("--candidate-hash", required=True)
  ap.add_argument("--resource-artifact", type=pathlib.Path); ap.add_argument("--repetitions", type=int, default=3)
  ap.add_argument("--output", type=pathlib.Path); ap.add_argument("--child", action="store_true")
  args = ap.parse_args()
  with _restore_gpu_policy():
    if args.child: return _child(args.payload, args.candidate_hash)
    if args.resource_artifact is None: ap.error("--resource-artifact is required")
    payload = json.loads(args.payload.read_text()); resource = json.loads(args.resource_artifact.read_text())
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child", "--payload", str(args.payload),
               "--candidate-hash", args.candidate_hash]
    report = build_report(payload, args.candidate_hash, resource, repetitions=args.repetitions, command=command)
    text = json.dumps(report, indent=2) + "\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
