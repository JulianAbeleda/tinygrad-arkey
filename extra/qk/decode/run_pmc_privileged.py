#!/usr/bin/env python3
"""Run one explicitly requested PMC diagnostic under bounded root scope.

This wrapper does not modify capabilities or system policy. It uses the existing
passwordless sudo policy for the requested command, isolates root's HOME, and
refuses graph PMC unless the caller explicitly opts into that unsafe path.
"""
from __future__ import annotations

import argparse, os, pathlib, subprocess, sys


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--allow-unsafe-graph", action="store_true",
                  help="allow PMC_GRAPH=1; this is not approved for decode")
  ap.add_argument("command", nargs=argparse.REMAINDER,
                  help="command after -- to run as the privileged diagnostic")
  args = ap.parse_args(argv)
  command = args.command[1:] if args.command and args.command[0] == "--" else args.command
  if not command: ap.error("provide a command after --")
  if os.environ.get("PMC", "1") != "1": ap.error("PMC must be 1 for this wrapper")
  if os.environ.get("PMC_GRAPH", "0") == "1" and not args.allow_unsafe_graph:
    ap.error("PMC_GRAPH=1 is refused; pass --allow-unsafe-graph only for an approved experiment")
  if os.geteuid() == 0:
    return subprocess.call(command)

  repo = pathlib.Path(__file__).resolve().parents[3]
  pythonpath = os.environ.get("PYTHONPATH", str(repo))
  user_site = "/home/ubuntu/.local/lib/python3.12/site-packages"
  if pathlib.Path(user_site).is_dir() and user_site not in pythonpath: pythonpath = f"{pythonpath}:{user_site}"
  env = {
    "HOME": os.environ.get("TINYGRAD_PMC_ROOT_HOME", "/tmp/tinygrad-pmc-root"),
    "DEV": os.environ.get("DEV", "AMD"), "PROFILE": os.environ.get("PROFILE", "1"),
    "PMC": "1", "PMC_GRAPH": os.environ.get("PMC_GRAPH", "0"),
    "PYTHONPATH": pythonpath,
  }
  for key in ("PREFILL_V2", "PREFILL_CHUNKED", "PREFILL_GRAPH_GEMM", "TINYGRAD_LAUNCH_SIDECAR",
              "TINYGRAD_OBSERVATION_BINARY_DIR", "TINYGRAD_OBSERVATION_CANDIDATE_ID",
              "TINYGRAD_OBSERVATION_SOURCE_SHA256", "TINYGRAD_OBSERVATION_SOURCE_COMMIT",
              "TINYGRAD_OBSERVATION_SOURCE_TREE_SHA256", "TINYGRAD_OBSERVATION_TARGET_ID",
              "TINYGRAD_OBSERVATION_RUNTIME_ID", "TINYGRAD_OBSERVATION_RUN_ID",
              "TINYGRAD_OBSERVATION_ATTEMPT_ID", "TINYGRAD_OBSERVATION_SYNC",
              "TINYGRAD_OBSERVATION_FLUSH_EVERY"):
    if key in os.environ: env[key] = os.environ[key]
  return subprocess.call(["sudo", "-n", "env", *[f"{k}={v}" for k, v in env.items()], *command], cwd=repo)


if __name__ == "__main__": raise SystemExit(main())
