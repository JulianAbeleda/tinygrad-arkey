#!/usr/bin/env python3
"""Collect fail-closed sidecar manifests for bounded decode/ROCm runs.

Importing this module has no host or device side effects.  Collection occurs only when
``collect_run_manifest`` is called or the CLI is invoked.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, subprocess, time
from typing import Any, Callable

SCHEMA = "14b-decode-run-manifest.v1"
REQUIRED = ("schema", "task_id", "created_unix_ns", "branch", "commit", "worktree", "git_dirty_paths",
            "command_argv", "environment_overrides", "model_path", "model_size_bytes", "model_mtime_ns",
            "model_identity_sha256", "backend", "device", "architecture", "boot_id", "lock_path",
            "lock_owner_pid", "power_before", "power_after", "start_time", "end_time", "exit_code",
            "classification", "positive_control", "stdout_path", "stderr_path", "primary_artifact_path",
            "kernel_or_route_identity", "notes")
ROUTE_ENV_PREFIXES = ("TINYGRAD_", "AMD_", "HIP_", "HSA_", "ROCR_", "ROCPROF", "GGML_", "LLAMA_")

def _run(argv: list[str], cwd: str) -> str:
  return subprocess.run(argv, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE).stdout.strip()

def _sha256(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
  return digest.hexdigest()

def allowed_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
  environ = os.environ if environ is None else environ
  return {key: value for key, value in sorted(environ.items()) if key.startswith(ROUTE_ENV_PREFIXES)}

def git_facts(worktree: str, run: Callable[[list[str], str], str] = _run) -> dict[str, Any]:
  status = run(["git", "status", "--porcelain=v1"], worktree)
  dirty = [line[3:] for line in status.splitlines() if len(line) >= 4]
  return {"branch": run(["git", "branch", "--show-current"], worktree),
          "commit": run(["git", "rev-parse", "HEAD"], worktree), "worktree": str(pathlib.Path(worktree).resolve()),
          "git_dirty_paths": dirty}

def read_power(run: Callable[[list[str], str], str] = _run, cwd: str = ".") -> dict[str, Any]:
  try: return json.loads(run(["rocm-smi", "--json", "--showperflevel", "--showclocks"], cwd))
  except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError): return {"available": False}

def validate_manifest(manifest: dict[str, Any]) -> None:
  missing = [key for key in REQUIRED if key not in manifest]
  if missing: raise ValueError(f"run manifest missing required fields: {missing}")
  if not manifest["task_id"] or not manifest["command_argv"]: raise ValueError("task_id and command_argv are required")
  if not isinstance(manifest["positive_control"], dict) or not manifest["positive_control"]:
    raise ValueError("positive_control must be a non-empty mapping")
  if not manifest["branch"] or not manifest["commit"] or not manifest["worktree"]:
    raise ValueError("branch, commit, and worktree positive controls are required")

def collect_run_manifest(*, task_id: str, command_argv: list[str], model_path: str, backend: str, device: str,
                         architecture: str, positive_control: dict[str, Any], classification: str,
                         stdout_path: str, stderr_path: str, primary_artifact_path: str,
                         kernel_or_route_identity: str, notes: str, worktree: str = ".", lock_path: str = "/tmp/gpu-bench.lock",
                         lock_owner_pid: int | None = None, power_before: dict[str, Any] | None = None,
                         power_after: dict[str, Any] | None = None, start_time: float | None = None,
                         end_time: float | None = None, exit_code: int = 0,
                         environ: dict[str, str] | None = None, run: Callable[[list[str], str], str] = _run) -> dict[str, Any]:
  model = pathlib.Path(model_path)
  if not model.is_file(): raise ValueError(f"model path does not exist: {model}")
  now = time.time()
  stat = model.stat()
  manifest = {"schema": SCHEMA, "task_id": task_id, "created_unix_ns": time.time_ns(), **git_facts(worktree, run),
    "command_argv": list(command_argv), "environment_overrides": allowed_environment(environ),
    "model_path": str(model.resolve()), "model_size_bytes": stat.st_size, "model_mtime_ns": stat.st_mtime_ns,
    "model_identity_sha256": _sha256(model), "backend": backend, "device": device, "architecture": architecture,
    "boot_id": pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "lock_path": lock_path,
    "lock_owner_pid": lock_owner_pid, "power_before": read_power(run, worktree) if power_before is None else power_before,
    "power_after": read_power(run, worktree) if power_after is None else power_after, "start_time": now if start_time is None else start_time,
    "end_time": now if end_time is None else end_time, "exit_code": exit_code, "classification": classification,
    "positive_control": positive_control, "stdout_path": stdout_path, "stderr_path": stderr_path,
    "primary_artifact_path": primary_artifact_path, "kernel_or_route_identity": kernel_or_route_identity, "notes": notes}
  validate_manifest(manifest)
  return manifest

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True); parser.add_argument("--task-id", required=True); parser.add_argument("--model", required=True)
  parser.add_argument("--backend", required=True); parser.add_argument("--device", required=True); parser.add_argument("--architecture", required=True)
  parser.add_argument("--positive-control", required=True, help="non-empty JSON object")
  parser.add_argument("--classification", default="COLLECTED"); parser.add_argument("--worktree", default=".")
  parser.add_argument("command", nargs=argparse.REMAINDER)
  args = parser.parse_args()
  manifest = collect_run_manifest(task_id=args.task_id, command_argv=args.command, model_path=args.model, backend=args.backend,
    device=args.device, architecture=args.architecture, positive_control=json.loads(args.positive_control), classification=args.classification,
    stdout_path="", stderr_path="", primary_artifact_path=args.out, kernel_or_route_identity="", notes="", worktree=args.worktree)
  out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__": main()
