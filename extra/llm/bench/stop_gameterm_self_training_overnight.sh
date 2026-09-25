#!/usr/bin/env bash
set -euo pipefail

out="${1:-/home/ubuntu/storage/self-training/overnight}"
pid_file="$out/launcher.pid"
if [[ ! -f "$pid_file" ]]; then
  echo "no launcher PID file at $pid_file"
  exit 0
fi
pid="$(cat "$pid_file")"
if kill -0 "$pid" 2>/dev/null; then
  kill -INT -- "-$pid"
  echo "requested a clean stop for controller process group $pid"
else
  echo "controller PID $pid is no longer running"
fi
