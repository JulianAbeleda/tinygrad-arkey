#!/usr/bin/env python3
"""Run one command while holding the local eGPU qualification lock.

The lock is advisory and intentionally machine-local.  It serializes eGPU
observations made by this worktree; it is not an authorization mechanism.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid


DEFAULT_LOCK_PATH = pathlib.Path("/tmp/gpu-bench.lock")
LOCK_SCHEMA = "tinygrad.gpu.lock.v1"
LOCK_UNAVAILABLE_EXIT = 75  # EX_TEMPFAIL, without relying on the Unix-only `sysexits` module.


def _git_head(cwd: pathlib.Path) -> str | None:
  try:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _write_all(fd: int, payload: bytes) -> None:
  offset = 0
  while offset < len(payload):
    written = os.write(fd, payload[offset:])
    if written == 0: raise OSError("short write while recording GPU lock metadata")
    offset += written


def _write_metadata(fd: int, metadata: dict) -> None:
  payload = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
  os.ftruncate(fd, 0)
  os.lseek(fd, 0, os.SEEK_SET)
  _write_all(fd, payload)
  os.fsync(fd)


def _read_metadata(path: pathlib.Path) -> str:
  try:
    return path.read_text(encoding="utf-8").strip() or "unavailable"
  except OSError:
    return "unavailable"


def _open_lock(path: pathlib.Path) -> int:
  flags = os.O_CREAT | os.O_RDWR
  flags |= getattr(os, "O_CLOEXEC", 0)
  flags |= getattr(os, "O_NOFOLLOW", 0)
  return os.open(path, flags, 0o600)


def _acquire(fd: int, wait_s: float) -> bool:
  deadline = time.monotonic() + wait_s
  while True:
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
      return True
    except BlockingIOError:
      if time.monotonic() >= deadline: return False
      time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _normalized_returncode(returncode: int) -> int:
  return returncode if returncode >= 0 else 128 + -returncode


def run_locked(command: list[str], *, lock_path: pathlib.Path = DEFAULT_LOCK_PATH, wait_s: float = 0.0,
               cwd: pathlib.Path | None = None) -> int:
  if not command: raise ValueError("a command is required after --")
  if wait_s < 0: raise ValueError("--wait-s must be non-negative")
  cwd = (cwd or pathlib.Path.cwd()).resolve()
  lock_path = lock_path.expanduser().resolve()
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  fd = _open_lock(lock_path)
  try:
    if not _acquire(fd, wait_s):
      print(f"eGPU lock unavailable: {lock_path}; owner metadata: {_read_metadata(lock_path)}", file=sys.stderr)
      return LOCK_UNAVAILABLE_EXIT

    nonce = uuid.uuid4().hex
    metadata = {
      "schema": LOCK_SCHEMA,
      "pid": os.getpid(),
      "started_monotonic_ns": time.monotonic_ns(),
      "started_unix_ns": time.time_ns(),
      "cwd": str(cwd),
      "git_head": _git_head(cwd),
      "argv": command,
      "nonce": nonce,
    }
    _write_metadata(fd, metadata)
    os.set_inheritable(fd, True)
    env = os.environ | {
      "TINYGRAD_GPU_LOCK_FD": str(fd),
      "TINYGRAD_GPU_LOCK_PATH": str(lock_path),
      "TINYGRAD_GPU_LOCK_NONCE": nonce,
    }
    child = None
    previous = None
    try:
      child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True, pass_fds=(fd,))

      def forward(signum: int, _frame) -> None:
        try:
          os.killpg(child.pid, signum)
        except ProcessLookupError:
          pass

      previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
      return _normalized_returncode(child.wait())
    finally:
      if previous is not None:
        for sig, handler in previous.items(): signal.signal(sig, handler)
      # Do not erase a newer holder's metadata: the advisory lock is still ours here.
      _write_metadata(fd, {})
      fcntl.flock(fd, fcntl.LOCK_UN)
  finally:
    os.close(fd)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lock-path", type=pathlib.Path, default=DEFAULT_LOCK_PATH,
                      help=f"advisory lock path (default: {DEFAULT_LOCK_PATH})")
  parser.add_argument("--wait-s", type=float, default=0.0,
                      help="bounded lock acquisition wait; default is nonblocking")
  parser.add_argument("command", nargs=argparse.REMAINDER, help="command to execute; prefix it with --")
  args = parser.parse_args(argv)
  command = args.command
  if command[:1] == ["--"]: command = command[1:]
  try:
    return run_locked(command, lock_path=args.lock_path, wait_s=args.wait_s)
  except (OSError, ValueError) as exc:
    print(f"with_gpu_lock: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
