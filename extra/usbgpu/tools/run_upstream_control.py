#!/usr/bin/env python3
"""Run a workload against the pinned, matched upstream TinyGPU control stack.

This runner never installs, activates, deactivates, or replaces a system
extension.  It refuses to run unless macOS reports the upstream extension as
enabled and the arkey extension as not enabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import plistlib
import subprocess
import sys
import time
import uuid

from with_gpu_lock import run_locked


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = pathlib.Path(os.environ.get("TINYGPU_UPSTREAM_ROOT", REPO_ROOT.parent / "tinygrad-upstream-control"))
UPSTREAM_HEAD = "6ea7d366fa92842c0bc8b7b080e26e83a7406252"
UPSTREAM_RELEASE = "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"
UPSTREAM_APP = pathlib.Path(os.environ.get("TINYGPU_UPSTREAM_APP", "/Applications/TinyGPU-Upstream.app"))
STANDARD_APP = pathlib.Path("/Applications/TinyGPU.app")
UPSTREAM_APP_ID = "org.tinygrad.tinygpu.installer"
UPSTREAM_DEXT_ID = "org.tinygrad.tinygpu.driver2"
ARKEY_APP_ID = "org.tinygrad.arkey.tinygpu.installer"
ARKEY_DEXT_ID = "org.tinygrad.arkey.tinygpu.driver2"
UPSTREAM_ZIP_SHA256 = "0c47285e2232643210555cf30ce08289b9e55da261c300e0c82e8448a359a21f"
UPSTREAM_APP_SHA256 = "3ed8bbd9ec8e14e7cf0047fbe7fb6169242394e3e5e75f5428d3edcb70254409"
UPSTREAM_DEXT_SHA256 = "236035427b9b182ad5f9eb3c16d4a3e5804f84bb864a933d6c7aa8e9c6f3f198"
CACHE_ZIP = pathlib.Path.home() / "Library/Caches/tinygrad/downloads" / f"TinyGPU_{UPSTREAM_RELEASE}.zip"


def _sha256(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024): digest.update(chunk)
  return digest.hexdigest()


def _plist_id(path:pathlib.Path) -> str:
  with path.open("rb") as stream: return plistlib.load(stream)["CFBundleIdentifier"]


def _output(argv:list[str], cwd:pathlib.Path|None=None) -> str:
  return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def registration_rows(output:str, bundle_id:str) -> list[str]:
  return [line.strip() for line in output.splitlines() if bundle_id in line]


def registration_ready(output:str) -> bool:
  upstream, arkey = registration_rows(output, UPSTREAM_DEXT_ID), registration_rows(output, ARKEY_DEXT_ID)
  return len(upstream) == 1 and "[activated enabled]" in upstream[0] and \
    not any("[activated enabled]" in row for row in arkey)


def inspect_control() -> dict:
  errors:list[str] = []
  app_exec = UPSTREAM_APP / "Contents/MacOS/TinyGPU"
  app_dext = UPSTREAM_APP / "Contents/Library/SystemExtensions" / f"{UPSTREAM_DEXT_ID}.dext" / UPSTREAM_DEXT_ID
  standard_plist = STANDARD_APP / "Contents/Info.plist"

  def expect(label:str, actual, expected) -> None:
    if actual != expected: errors.append(f"{label}: expected {expected!r}, got {actual!r}")

  try: expect("upstream worktree HEAD", _output(["git", "rev-parse", "HEAD"], UPSTREAM_ROOT), UPSTREAM_HEAD)
  except (OSError, subprocess.CalledProcessError) as exc: errors.append(f"upstream worktree unavailable: {exc}")
  try: expect("upstream worktree status", _output(["git", "status", "--porcelain=v1", "--untracked-files=all"], UPSTREAM_ROOT), "")
  except (OSError, subprocess.CalledProcessError) as exc: errors.append(f"upstream worktree status unavailable: {exc}")
  try: expect("upstream app bundle id", _plist_id(UPSTREAM_APP / "Contents/Info.plist"), UPSTREAM_APP_ID)
  except (OSError, KeyError, plistlib.InvalidFileException) as exc: errors.append(f"upstream app unavailable: {exc}")
  try: expect("upstream app executable", _sha256(app_exec), UPSTREAM_APP_SHA256)
  except OSError as exc: errors.append(f"upstream app executable unavailable: {exc}")
  try: expect("upstream app DEXT", _sha256(app_dext), UPSTREAM_DEXT_SHA256)
  except OSError as exc: errors.append(f"upstream app DEXT unavailable: {exc}")
  try: expect("upstream release cache", _sha256(CACHE_ZIP), UPSTREAM_ZIP_SHA256)
  except OSError as exc: errors.append(f"upstream release cache unavailable: {exc}")
  try: expect("standard app remains arkey", _plist_id(standard_plist), ARKEY_APP_ID)
  except (OSError, KeyError, plistlib.InvalidFileException) as exc: errors.append(f"standard arkey app unavailable: {exc}")
  try: subprocess.run(["codesign", "--verify", "--deep", "--strict", str(UPSTREAM_APP)], check=True, capture_output=True)
  except (OSError, subprocess.CalledProcessError) as exc: errors.append(f"upstream app signature invalid: {exc}")

  registered = sorted(pathlib.Path("/Library/SystemExtensions").glob(f"*/{UPSTREAM_DEXT_ID}.dext/{UPSTREAM_DEXT_ID}"))
  registered_hashes = []
  for path in registered:
    try: registered_hashes.append(_sha256(path))
    except OSError as exc: errors.append(f"registered upstream DEXT unavailable: {exc}")
  if registered_hashes != [UPSTREAM_DEXT_SHA256]:
    errors.append(f"registered upstream DEXT: expected one {UPSTREAM_DEXT_SHA256}, got {registered_hashes}")

  extension_output = ""
  try: extension_output = _output(["systemextensionsctl", "list"])
  except (OSError, subprocess.CalledProcessError) as exc: errors.append(f"system extension state unavailable: {exc}")
  ready = registration_ready(extension_output) if extension_output else False
  return {
    "schema":"tinygpu.upstream-control.v1", "artifact_ready":not errors, "activation_ready":ready,
    "upstream_head":UPSTREAM_HEAD, "upstream_release":UPSTREAM_RELEASE,
    "upstream_registration":registration_rows(extension_output, UPSTREAM_DEXT_ID),
    "arkey_registration":registration_rows(extension_output, ARKEY_DEXT_ID), "errors":errors,
  }


def _run_inside_lock(command:list[str]) -> int:
  if not os.environ.get("TINYGRAD_GPU_LOCK_FD"):
    print("upstream control refused: missing inherited eGPU lock", file=sys.stderr)
    return 2
  state = inspect_control()
  if not state["artifact_ready"] or not state["activation_ready"]:
    print(json.dumps(state, indent=2, sort_keys=True), file=sys.stderr)
    print("upstream control refused: enable upstream TinyGPU and disable arkey TinyGPU in Driver Extensions", file=sys.stderr)
    return 2

  socket_path = pathlib.Path(f"/tmp/tinygpu-upstream-control-{os.getpid()}-{uuid.uuid4().hex}.sock")
  app_exec = UPSTREAM_APP / "Contents/MacOS/TinyGPU"
  server = subprocess.Popen([str(app_exec), "server", str(socket_path)])
  try:
    deadline = time.monotonic() + 5.0
    while not socket_path.exists():
      if server.poll() is not None:
        print(f"upstream TinyGPU server exited with {server.returncode}", file=sys.stderr)
        return server.returncode or 2
      if time.monotonic() >= deadline:
        print("upstream TinyGPU server did not create its Unix socket", file=sys.stderr)
        return 2
      time.sleep(0.05)

    env = os.environ.copy()
    env.update({"APL_REMOTE_SOCK":str(socket_path), "PYTHONPATH":str(UPSTREAM_ROOT),
                "TINYGPU_CONTROL_HEAD":UPSTREAM_HEAD, "TINYGPU_CONTROL_RELEASE":UPSTREAM_RELEASE})
    return subprocess.run(command, cwd=UPSTREAM_ROOT, env=env, check=False).returncode
  finally:
    if server.poll() is None:
      server.terminate()
      try: server.wait(timeout=3)
      except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    try: socket_path.unlink()
    except FileNotFoundError: pass


def main(argv:list[str]|None=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, epilog="a command is required after -- unless --check is used")
  parser.add_argument("--check", action="store_true", help="report artifact and activation state without touching the GPU")
  parser.add_argument("--inside-lock", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run from the pinned upstream worktree; prefix it with --")
  args = parser.parse_args(argv)
  command = args.command[1:] if args.command[:1] == ["--"] else args.command
  if args.check:
    if command: parser.error("--check does not accept a command")
    print(json.dumps(inspect_control(), indent=2, sort_keys=True))
    return 0
  if not command: parser.error("a command is required after --")
  if args.inside_lock: return _run_inside_lock(command)
  child = [sys.executable, str(pathlib.Path(__file__).resolve()), "--inside-lock", "--", *command]
  return run_locked(child, cwd=REPO_ROOT)


if __name__ == "__main__":
  raise SystemExit(main())
