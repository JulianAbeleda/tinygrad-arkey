#!/usr/bin/env python3
"""Capture early TinyGPU idle evidence before a USB4/PCIe link disappears."""
from __future__ import annotations

import argparse
import copy
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path: sys.path.insert(0, str(TESTS_DIR))

from qualify import (DEFAULT_APP, DEFAULT_LOCK, DEFAULT_SOCKET, ROOT, QualificationError,
                     QualificationInterrupted, atomic_json, decode_json, default_process_reader,
                     exact_server_pids, socket_reachable, validate_continuity, validate_lock,
                     validate_power_continuity, validate_power_status, validate_status)


CAPTURE_SCHEMA = "tinygpu.tunnel-idle-capture.v1"


def validate_handshake(value: dict) -> None:
  fields = {"schema", "protocol_major", "protocol_minor", "capabilities", "server_build_id"}
  if set(value) != fields or value.get("schema") != "tinygpu.handshake.v1" or \
     value.get("protocol_major") != 1 or value.get("protocol_minor") != 0 or \
     type(value.get("capabilities")) is not int or value["capabilities"] & 11 != 11 or \
     type(value.get("server_build_id")) is not str or \
     re.fullmatch(r"[A-Za-z0-9._+-]{1,128}", value["server_build_id"]) is None or \
     value["server_build_id"] != "tinygrad-arkey-native-v12":
    raise QualificationError("invalid diagnostic handshake")


def validate_v12_registration(text: str) -> None:
  rows = [line.split() for line in text.splitlines() if "org.tinygrad.arkey.tinygpu.driver2" in line]
  if len(rows) != 1 or rows[0][:5] != ["*", "*", "-", "org.tinygrad.arkey.tinygpu.driver2", "(1.0.0/12)"] or \
     rows[0][-2:] != ["[activated", "enabled]"]:
    raise QualificationError("exactly one active and enabled arkey v12 registration is required")


def capture_idle(*, handshake_reader, status_reader, power_reader, registry_reader=lambda: {},
                 sleeper=time.sleep, clock=time.monotonic, wall_clock_ns=time.time_ns,
                 duration_s: float = 300, interval_s: float = 1,
                 checkpoint=lambda evidence: None) -> dict:
  if duration_s <= 0 or interval_s <= 0: raise ValueError("capture duration and interval must be positive")
  evidence = {"schema":CAPTURE_SCHEMA, "status":"running", "duration_s":duration_s,
              "interval_s":interval_s, "handshake":None, "registry":None, "samples":[],
              "first_failure":None}

  def save() -> None:
    checkpoint(copy.deepcopy(evidence))

  def sample(label: str) -> None:
    status = status_reader(); validate_status(status)
    power = power_reader(); validate_power_status(power)
    if power["provider_generation"] != status["provider_generation"]:
      raise QualificationError("status payload provider generations differ")
    if power["last_canary_identity_dword"] != status["last_identity_dword"] or \
       power["last_canary_success_monotonic_ns"] < status["last_success_monotonic_ns"]:
      raise QualificationError("power-residency canary does not cover keepalive sample")
    if evidence["samples"]:
      previous = evidence["samples"][-1]
      validate_continuity(previous["keepalive"], status)
      validate_power_continuity(previous["power"], power)
    evidence["samples"].append({"label":label, "unix_ns":wall_clock_ns(),
                                "keepalive":status, "power":power})
    save()

  try:
    evidence["handshake"] = handshake_reader(); validate_handshake(evidence["handshake"]); save()
    sample("initial")
    evidence["registry"] = registry_reader(); save()
    start = clock(); deadline = start + duration_s
    while clock() < deadline:
      sleeper(min(interval_s, max(0, deadline - clock())))
      sample(f"sample-{len(evidence['samples'])}")
    if len(evidence["samples"]) < 2:
      raise QualificationError("capture produced fewer than two samples")
    validate_continuity(evidence["samples"][0]["keepalive"], evidence["samples"][-1]["keepalive"], require_advance=True)
    validate_power_continuity(evidence["samples"][0]["power"], evidence["samples"][-1]["power"], require_canary_advance=True)
    evidence["status"] = "passed"
  except BaseException as exc:
    evidence["status"] = "failed"
    evidence["first_failure"] = {"type":type(exc).__name__, "message":str(exc), "unix_ns":wall_clock_ns()}
  save()
  return evidence


def run_command(argv: list[str], records: list[dict], label: str, *, max_output: int = 2 << 20) -> subprocess.CompletedProcess:
  result = subprocess.run(argv, check=False, capture_output=True)
  records.append({"label":label, "argv":argv, "returncode":result.returncode,
                  "stdout":result.stdout[-max_output:].decode("utf-8", "replace"),
                  "stderr":result.stderr[-max_output:].decode("utf-8", "replace")})
  return result


def json_command(argv: list[str], records: list[dict], label: str) -> dict:
  result = run_command(argv, records, label, max_output=65536)
  if result.returncode:
    raise QualificationError(f"{label} failed: {result.returncode}: {result.stderr.decode('utf-8', 'replace')[:4096]}")
  return decode_json(result.stdout.strip(), max_bytes=4096 if label != "handshake" else 65536)


def required_text_command(argv: list[str], records: list[dict], label: str) -> str:
  result = run_command(argv, records, label)
  if result.returncode:
    raise QualificationError(f"{label} failed: {result.returncode}: {result.stderr.decode('utf-8', 'replace')[:4096]}")
  return result.stdout.decode("utf-8", "replace")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--duration-s", type=float, default=300)
  ap.add_argument("--interval-s", type=float, default=1)
  args = ap.parse_args(argv)
  created = time.time_ns()
  out = ROOT / "docs/task_workflow/output" / \
    f"egpu-usb4-tunnel-idle-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}.json"
  records: list[dict] = []
  envelope = {"schema":CAPTURE_SCHEMA, "created_unix_ns":created, "lock":None,
              "preflight":{}, "commands":records, "capture":None, "post_capture":{}}
  previous_handlers = {}
  locked = False

  def write(_capture=None) -> None:
    if _capture is not None: envelope["capture"] = _capture
    atomic_json(out, envelope)

  try:
    lock = validate_lock(); envelope["lock"] = lock
    if pathlib.Path(os.environ["TINYGRAD_GPU_LOCK_PATH"]).resolve() != DEFAULT_LOCK.resolve():
      raise QualificationError(f"capture requires {DEFAULT_LOCK}")
    if pathlib.Path(lock.get("cwd", "")).resolve() != ROOT:
      raise QualificationError("lock runner cwd is not the feature worktree")
    locked = True
    processes = default_process_reader()
    server_pids = exact_server_pids(DEFAULT_APP, processes)
    if server_pids or socket_reachable(DEFAULT_SOCKET):
      raise QualificationError(f"TinyGPU workload server must be absent: pids={server_pids}")
    for sig in (signal.SIGINT, signal.SIGTERM):
      previous_handlers[sig] = signal.signal(sig, lambda signum, _frame: (_ for _ in ()).throw(
        QualificationInterrupted(f"signal {signum}")))

    registration = required_text_command(["/usr/bin/systemextensionsctl", "list"], records, "system extensions")
    validate_v12_registration(registration)
    envelope["preflight"] = {
      "system_extensions":registration,
      "boot_time":required_text_command(["/usr/sbin/sysctl", "-n", "kern.boottime"], records, "boot time").strip(),
      "app":str(DEFAULT_APP),
    }
    write()

    def registry_reader():
      return {
        "tinygpu":required_text_command(["/usr/sbin/ioreg", "-p", "IOService", "-r", "-n", "tinygpu", "-l", "-w", "0"],
                                         records, "ioreg tinygpu"),
        "pci_devices":required_text_command(["/usr/sbin/ioreg", "-p", "IOService", "-r", "-c", "IOPCIDevice", "-l", "-w", "0"],
                                             records, "ioreg pci devices"),
      }

    envelope["capture"] = capture_idle(
      handshake_reader=lambda:json_command([str(DEFAULT_APP), "keepalive", "handshake"], records, "handshake"),
      status_reader=lambda:json_command([str(DEFAULT_APP), "keepalive", "status"], records, "keepalive status"),
      power_reader=lambda:json_command([str(DEFAULT_APP), "power", "status"], records, "power status"),
      registry_reader=registry_reader, duration_s=args.duration_s, interval_s=args.interval_s, checkpoint=write)
  except BaseException as exc:
    envelope["capture"] = {"schema":CAPTURE_SCHEMA, "status":"failed", "samples":[],
                           "first_failure":{"type":type(exc).__name__, "message":str(exc), "unix_ns":time.time_ns()}}
  finally:
    for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
    if locked:
      predicate = '(process == "kernel") AND (eventMessage CONTAINS[c] "ACIO" OR eventMessage CONTAINS[c] "linkStatus" OR eventMessage CONTAINS[c] "dead child" OR eventMessage CONTAINS[c] "stopUsingTunnel" OR eventMessage CONTAINS[c] "tinygpu")'
      envelope["post_capture"] = {
        "relevant_log":run_command(["/usr/bin/log", "show", "--last", "10m", "--style", "compact", "--predicate", predicate],
                                   records, "relevant kernel log").stdout.decode("utf-8", "replace"),
        "system_profiler":run_command(["/usr/sbin/system_profiler", "SPThunderboltDataType", "SPPCIDataType"],
                                      records, "post-capture topology").stdout.decode("utf-8", "replace"),
      }
    write()
  print(out)
  return 0 if envelope["capture"] and envelope["capture"].get("status") == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
