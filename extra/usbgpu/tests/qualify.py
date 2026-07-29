#!/usr/bin/env python3
"""Lock-held evidence runner for the TinyGPU USB4 eGPU acceptance matrix."""
from __future__ import annotations

import argparse, fcntl, hashlib, json, os, pathlib, platform, re, signal, socket, stat, subprocess, sys, tempfile, time

GATES = tuple(f"A{i}" for i in range(12))
CLASSIFICATION_GATES = {"A10", "A11"}
STATUS_FIELDS = {"schema", "provider_generation", "state", "enabled", "policy_id", "interval_ms", "maximum_timer_leeway_ms",
                 "expected_identity", "last_identity_dword", "attempts", "successes", "failures", "consecutive_failures",
                 "last_attempt_monotonic_ns", "last_success_monotonic_ns", "success_gap_over_leeway_count", "max_success_gap_ms",
                 "timer_error", "counter_saturated", "active_workload_leases", "active_bar_mappings", "active_dma_allocations"}
POWER_FIELDS = {"schema", "provider_generation", "policy_id", "override_requested", "override_active", "full_power_requested",
                "power_request_accepted", "power_request_confirmed", "power_release_attempted", "desired_power_flags",
                "last_observed_power_flags", "override_request_error", "power_request_error", "power_release_error",
                "override_release_error", "transition_count", "unexpected_downgrade_count", "last_transition_monotonic_ns",
                "last_canary_identity_dword", "last_canary_success_monotonic_ns", "publishable"}
MONOTONIC_FIELDS = ("attempts", "successes", "failures", "consecutive_failures", "last_attempt_monotonic_ns",
                    "last_success_monotonic_ns", "success_gap_over_leeway_count", "max_success_gap_ms")
REQUIRED_ENV = {"DEV":"AMD", "JIT":"1", "PYTHONPATH":".", "AM_REMOTE_DISCOVERY_PROFILE":"gfx1100_744c",
                "AM_REMOTE_SKIP_RESIZE_BAR":"1"}
FORBIDDEN_ENV = ("REMOTE_KEEPALIVE_S", "AM_REMOTE_SMALL_BAR_DISCOVERY")
ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_APP = pathlib.Path("/Applications/TinyGPU.app/Contents/MacOS/TinyGPU")
DEFAULT_SOCKET = pathlib.Path(os.environ.get("APL_REMOTE_SOCK", "/tmp/tinygpu.sock"))
DEFAULT_LOCK = pathlib.Path("/tmp/gpu-bench.lock")
DEFAULT_INSTALL_PROVENANCE = ROOT / "docs/task_workflow/output/tinygpu-development-install-provenance.txt"


class QualificationError(RuntimeError): pass
class QualificationInterrupted(QualificationError): pass


def atomic_json(path:pathlib.Path, payload:dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(payload, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def sha256(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda:f.read(1<<20), b""): digest.update(block)
  return digest.hexdigest()


def decode_json(payload:bytes, *, max_bytes:int) -> dict:
  if len(payload) > max_bytes: raise QualificationError("JSON payload is too large")
  def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
      if key in result: raise ValueError(f"duplicate JSON key: {key}")
      result[key] = value
    return result
  try: value = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates)
  except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc: raise QualificationError("malformed JSON payload") from exc
  if type(value) is not dict: raise QualificationError("JSON object required")
  return value


def validate_lock(env=os.environ, *, parent_pid:int|None=None) -> dict:
  try:
    fd = int(env["TINYGRAD_GPU_LOCK_FD"]); path = pathlib.Path(env["TINYGRAD_GPU_LOCK_PATH"]).resolve()
    nonce = env["TINYGRAD_GPU_LOCK_NONCE"]
    descriptor, target = os.fstat(fd), path.stat()
    payload = decode_json(path.read_bytes(), max_bytes=16384)
  except (KeyError, OSError, ValueError) as exc: raise QualificationError("valid inherited GPU lock metadata is required") from exc
  # macOS presents inherited descriptors through a synthetic /dev/fd device with a different st_dev;
  # the canonical lock path and inode are the stable identity checks.
  if not stat.S_ISREG(descriptor.st_mode) or descriptor.st_ino != target.st_ino:
    raise QualificationError("GPU lock descriptor does not name the metadata file")
  if payload.get("schema") != "tinygrad.gpu.lock.v1" or payload.get("nonce") != nonce: raise QualificationError("GPU lock metadata mismatch")
  if parent_pid is None: parent_pid = os.getppid()
  if payload.get("pid") != parent_pid: raise QualificationError("GPU lock was not inherited from the lock runner")
  try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as exc: raise QualificationError("inherited descriptor does not own the GPU lock") from exc
  return payload


def validate_environment(env=os.environ) -> dict:
  mismatches = {key:{"expected": expected, "actual": env.get(key)} for key, expected in REQUIRED_ENV.items() if env.get(key) != expected}
  forbidden = {key:env[key] for key in FORBIDDEN_ENV if env.get(key) not in (None, "", "0")}
  if mismatches: raise QualificationError(f"qualification environment mismatch: {mismatches}")
  if forbidden: raise QualificationError(f"qualification forbids legacy or small-BAR controls: {forbidden}")
  return {key:env[key] for key in REQUIRED_ENV}


def validate_install_provenance(path:pathlib.Path, installed_executable:pathlib.Path, *, source_commit:str|None=None) -> dict:
  """Bind the live app/dext bytes to the install transcript and current feature commit."""
  app = installed_executable.resolve()
  app_root = app.parents[2]
  dext_root = app_root / "Contents/Library/SystemExtensions/org.tinygrad.arkey.tinygpu.driver2.dext"
  dext = dext_root / "org.tinygrad.arkey.tinygpu.driver2"
  try:
    raw = path.read_bytes()
    if not raw or len(raw) > 8 << 20: raise QualificationError("install provenance size is invalid")
    text = raw.decode("utf-8")
    if source_commit is None:
      source_commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    app_hash, dext_hash = sha256(app), sha256(dext)
  except (OSError, UnicodeDecodeError, subprocess.CalledProcessError) as exc:
    raise QualificationError("audited install provenance and live binaries are required") from exc
  required = (
    "schema=tinygpu.development-install.provenance.v1", f"source_commit={source_commit}", "=== activated ===",
    f"Identifier=org.tinygrad.arkey.tinygpu.installer", f"Identifier=org.tinygrad.arkey.tinygpu.driver2",
    "Signature=adhoc", "[activated enabled]", f"{app_hash}  {app}", f"{dext_hash}  {dext}",
  )
  missing = [item for item in required if item not in text]
  if missing: raise QualificationError(f"install provenance does not match the live feature build: {missing}")
  return {"schema":"tinygpu.development-install.provenance.v1", "path":str(path.resolve()), "sha256":hashlib.sha256(raw).hexdigest(),
          "source_commit":source_commit, "app_sha256":app_hash, "dext_sha256":dext_hash}


def validate_status(value:dict) -> None:
  if set(value) != STATUS_FIELDS or value.get("schema") != "tinygpu.keepalive.v1": raise QualificationError("malformed keepalive status")
  u64 = ("provider_generation", "attempts", "successes", "failures", "consecutive_failures", "last_attempt_monotonic_ns",
         "last_success_monotonic_ns", "success_gap_over_leeway_count", "max_success_gap_ms")
  u32 = ("interval_ms", "maximum_timer_leeway_ms", "active_workload_leases", "active_bar_mappings", "active_dma_allocations")
  if any(type(value[k]) is not int or not 0 <= value[k] < 1<<64 for k in u64): raise QualificationError("invalid status counter")
  if any(type(value[k]) is not int or not 0 <= value[k] < 1<<32 for k in u32): raise QualificationError("invalid status u32")
  if type(value["timer_error"]) is not int or not -(1<<31) <= value["timer_error"] < 1<<31: raise QualificationError("invalid timer error")
  if type(value["enabled"]) is not bool or type(value["counter_saturated"]) is not bool: raise QualificationError("invalid status boolean")
  if value["policy_id"] != "usb4_amd_744c_v1" or value["expected_identity"] != "1002:744c" or value["last_identity_dword"] != "0x744c1002":
    raise QualificationError("keeper identity or policy mismatch")
  if value["interval_ms"] != 1000 or value["maximum_timer_leeway_ms"] != 100: raise QualificationError("unexpected keepalive policy")
  if value["state"] != "active_healthy" or not value["enabled"] or value["timer_error"] != 0: raise QualificationError("keeper is not healthy")
  if value["counter_saturated"]: raise QualificationError("saturated status counter")
  if value["attempts"] != value["successes"] + value["failures"]: raise QualificationError("status counter invariant")
  if value["attempts"] == 0 or value["successes"] == 0 or value["last_attempt_monotonic_ns"] == 0 or value["last_success_monotonic_ns"] == 0:
    raise QualificationError("keeper has no successful tick")
  if value["failures"] != 0 or value["consecutive_failures"] != 0: raise QualificationError("keeper recorded a failed tick")
  if value["max_success_gap_ms"] > 2000: raise QualificationError("keeper missed a one-shot deadline")
  if any(value[k] != 0 for k in ("active_workload_leases", "active_bar_mappings", "active_dma_allocations")):
    raise QualificationError("workload resource leak")


def validate_continuity(first:dict, second:dict, *, require_advance:bool=False) -> None:
  validate_status(first); validate_status(second)
  if first["provider_generation"] != second["provider_generation"]: raise QualificationError("provider generation changed")
  if any(second[key] < first[key] for key in MONOTONIC_FIELDS): raise QualificationError("keeper counter regressed")
  if second["failures"] != first["failures"]: raise QualificationError("keeper failure count changed")
  if require_advance and second["successes"] <= first["successes"]: raise QualificationError("keeper did not advance")


def validate_power_status(value:dict) -> None:
  if set(value) != POWER_FIELDS or value.get("schema") != "tinygpu.power-residency.v1": raise QualificationError("malformed power-residency status")
  u64 = ("provider_generation", "transition_count", "unexpected_downgrade_count", "last_transition_monotonic_ns",
         "last_canary_success_monotonic_ns")
  u32 = ("desired_power_flags", "last_observed_power_flags")
  i32 = ("override_request_error", "power_request_error", "power_release_error", "override_release_error")
  boolean = ("override_requested", "override_active", "full_power_requested", "power_request_accepted", "power_request_confirmed",
             "power_release_attempted", "publishable")
  if any(type(value[k]) is not int or not 0 <= value[k] < 1<<64 for k in u64): raise QualificationError("invalid power-residency counter")
  if any(type(value[k]) is not int or not 0 <= value[k] < 1<<32 for k in u32): raise QualificationError("invalid power-residency flags")
  if any(type(value[k]) is not int or not -(1<<31) <= value[k] < 1<<31 for k in i32): raise QualificationError("invalid power-residency error")
  if any(type(value[k]) is not bool for k in boolean): raise QualificationError("invalid power-residency boolean")
  if value["policy_id"] != "driverkit_full_power_v1" or value["desired_power_flags"] != 2 or value["last_observed_power_flags"] != 2:
    raise QualificationError("power-residency policy or observed state mismatch")
  if not all(value[k] for k in ("override_requested", "override_active", "full_power_requested", "power_request_accepted",
                                "power_request_confirmed", "publishable")) or value["power_release_attempted"]:
    raise QualificationError("provider power residency is not active")
  if any(value[k] for k in i32) or value["unexpected_downgrade_count"]: raise QualificationError("power-residency request or transition failed")
  if value["transition_count"] == 0 or value["last_transition_monotonic_ns"] == 0 or value["last_canary_success_monotonic_ns"] == 0:
    raise QualificationError("power-residency evidence is incomplete")
  if value["last_canary_identity_dword"] != "0x744c1002": raise QualificationError("power-residency canary identity mismatch")


def validate_power_continuity(first:dict, second:dict, *, require_canary_advance:bool=False) -> None:
  validate_power_status(first); validate_power_status(second)
  if first["provider_generation"] != second["provider_generation"]: raise QualificationError("power-residency provider generation changed")
  for key in ("transition_count", "unexpected_downgrade_count", "last_transition_monotonic_ns", "last_canary_success_monotonic_ns"):
    if second[key] < first[key]: raise QualificationError("power-residency counter regressed")
  if require_canary_advance and second["last_canary_success_monotonic_ns"] <= first["last_canary_success_monotonic_ns"]:
    raise QualificationError("power-residency canary did not advance")


def validate_cadence(first:dict, second:dict) -> None:
  validate_continuity(first, second, require_advance=True)
  observed = second["successes"] - first["successes"]
  over = second["success_gap_over_leeway_count"] - first["success_gap_over_leeway_count"]
  if over / observed > .01: raise QualificationError("keeper cadence exceeded leeway")


def status_command(command:list[str]) -> dict:
  result = subprocess.run(command, check=False, capture_output=True)
  if result.returncode: raise QualificationError(f"status command failed: {result.returncode}: {result.stderr.decode('utf-8', 'replace')[:4096]}")
  return decode_json(result.stdout.strip(), max_bytes=4096)


def handshake_command(command:list[str]) -> dict:
  result = subprocess.run(command, check=False, capture_output=True)
  if result.returncode: raise QualificationError(f"handshake command failed: {result.returncode}")
  value = decode_json(result.stdout.strip(), max_bytes=65536)
  if set(value) != {"schema", "protocol_major", "protocol_minor", "capabilities", "server_build_id"} or \
     value.get("schema") != "tinygpu.handshake.v1" or value.get("protocol_major") != 1 or value.get("protocol_minor") != 0 or \
     type(value.get("capabilities")) is not int or value["capabilities"] & 11 != 11 or \
     type(value.get("server_build_id")) is not str or re.fullmatch(r"[A-Za-z0-9._+-]{1,128}", value["server_build_id"]) is None:
    raise QualificationError("invalid diagnostic handshake")
  return value


def default_endpoint_reader() -> bool:
  result = subprocess.run(["/usr/sbin/system_profiler", "-json", "SPPCIDataType"], check=False, capture_output=True)
  if result.returncode: return False
  try: root = decode_json(result.stdout, max_bytes=16<<20)
  except QualificationError: return False
  def normalize(value) -> str:
    match = re.search(r"0x([0-9a-fA-F]{4})", str(value))
    return match.group(1).lower() if match else ""
  def visit(value) -> bool:
    if isinstance(value, dict):
      lowered = {str(key).lower(): item for key, item in value.items()}
      vendor = next((normalize(item) for key, item in lowered.items() if "vendor" in key and "id" in key), "")
      device = next((normalize(item) for key, item in lowered.items() if "device" in key and "id" in key), "")
      if vendor == "1002" and device == "744c": return True
      return any(visit(item) for item in value.values())
    if isinstance(value, list): return any(visit(item) for item in value)
    return False
  return visit(root)


def socket_reachable(path:pathlib.Path) -> bool:
  client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  client.settimeout(.25)
  try: client.connect(str(path)); return True
  except OSError: return False
  finally: client.close()


def exact_server_pids(installed:pathlib.Path, processes:list[tuple[int, list[str]]]) -> list[int]:
  target = str(installed.resolve())
  return [pid for pid, argv in processes if len(argv) >= 2 and str(pathlib.Path(argv[0]).resolve()) == target and argv[1] == "server"]


def default_process_reader() -> list[tuple[int, list[str]]]:
  rows = subprocess.run(["ps", "-ww", "-axo", "pid=,command="], check=False, capture_output=True, text=True).stdout.splitlines()
  return [(int(parts[0]), parts[1:]) for row in rows if (parts:=row.strip().split()) and parts[0].isdigit()]


def common_context(app:pathlib.Path) -> dict:
  def output(command):
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {"argv":command, "returncode":result.returncode, "stdout":result.stdout[-65536:].strip(), "stderr":result.stderr[-65536:].strip()}
  app_root = app.parents[2] if len(app.parents) >= 3 else app
  dext_root = app_root / "Contents/Library/SystemExtensions/org.tinygrad.arkey.tinygpu.driver2.dext"
  dext_exe = dext_root / "org.tinygrad.arkey.tinygpu.driver2"
  file_info = lambda path: {"path":str(path), "size_bytes":path.stat().st_size if path.is_file() else None,
                            "sha256":sha256(path) if path.is_file() else None}
  return {
    "host":platform.platform(), "worktree":str(ROOT),
    "git":{"head":output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "branch":output(["git", "-C", str(ROOT), "branch", "--show-current"]),
           "status":output(["git", "-C", str(ROOT), "status", "--short", "--branch"])},
    "toolchain":{"xcode":output(["xcodebuild", "-version"]), "macos_sdk":output(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
                 "driverkit_sdk":output(["xcrun", "--sdk", "driverkit", "--show-sdk-version"])},
    "os":{"sw_vers":output(["sw_vers"]), "uname":output(["uname", "-a"]), "csrutil":output(["csrutil", "status"])},
    "app":{"file":file_info(app), "bundle_id":output(["plutil", "-extract", "CFBundleIdentifier", "raw", "-o", "-", str(app_root / "Contents/Info.plist")]),
           "codesign":output(["codesign", "-dvvv", str(app_root)]), "entitlements":output(["codesign", "-d", "--entitlements", ":-", str(app_root)])},
    "dext":{"file":file_info(dext_exe), "bundle_id":output(["plutil", "-extract", "CFBundleIdentifier", "raw", "-o", "-", str(dext_root / "Info.plist")]),
            "codesign":output(["codesign", "-dvvv", str(dext_root)]), "entitlements":output(["codesign", "-d", "--entitlements", ":-", str(dext_root)])},
    "system_extensions":output(["systemextensionsctl", "list"]),
    "topology":output(["system_profiler", "SPThunderboltDataType", "SPPCIDataType", "SPPowerDataType"]),
  }


def run_gate(gate:str, *, status_reader, endpoint_reader, process_reader=lambda:[], terminator=lambda pid:None, sleeper=time.sleep,
             installed_executable:pathlib.Path|None=None, model:pathlib.Path|None=None, manual_prompt=lambda _:None,
             runner=lambda command: subprocess.run(command, check=False, capture_output=True), minimal_command=None, bench_command=None,
             handshake_reader=None, duration_s=None, sample_interval_s=60, churn_count=25, churn_idle_s=5,
             idle_s=None, include_post_idle=False, clock=time.monotonic, socket_reader=lambda:False, install_provenance=None,
             power_status_reader=None) -> dict:
  evidence, first_failure = {"gate":gate, "samples":[], "power_samples":[], "endpoint_checks":[], "commands":[], "command_results":[]}, None
  try:
    if endpoint_reader is None: raise QualificationError("endpoint reader is required")
    if model is not None:
      if not model.is_file(): raise QualificationError("model does not exist")
      evidence["model"] = {"path":str(model.resolve()), "size_bytes":model.stat().st_size, "sha256":sha256(model)}
    def checked_status(label):
      visible = bool(endpoint_reader()); evidence["endpoint_checks"].append({"label":label, "visible":visible, "unix_ns":time.time_ns()})
      if not visible: raise QualificationError(f"endpoint disappeared at {label}")
      value = status_reader(); validate_status(value); evidence["samples"].append(value)
      if power_status_reader is not None:
        power = power_status_reader(); validate_power_status(power); evidence["power_samples"].append(power)
        if power["provider_generation"] != value["provider_generation"]: raise QualificationError("status payload provider generations differ")
        if power["last_canary_identity_dword"] != value["last_identity_dword"] or \
           power["last_canary_success_monotonic_ns"] < value["last_success_monotonic_ns"]:
          raise QualificationError("power-residency canary does not cover keepalive sample")
      elif gate in {"A0", "A1"}: raise QualificationError(f"{gate} requires power-residency status")
      return value
    def command(argv, label, *, classify=False):
      if not argv: raise QualificationError(f"{gate} requires {label} command")
      resolved = [str(item) for item in argv]; evidence["commands"].append(resolved)
      result = runner(resolved)
      if isinstance(result, int): returncode, stdout, stderr = result, b"", b""
      else:
        returncode, stdout, stderr = result.returncode, result.stdout or b"", result.stderr or b""
        if isinstance(stdout, str): stdout = stdout.encode()
        if isinstance(stderr, str): stderr = stderr.encode()
      record = {"label":label, "argv":resolved, "returncode":returncode, "stdout":stdout[-65536:].decode("utf-8", "replace"),
                "stderr":stderr[-65536:].decode("utf-8", "replace")}
      evidence["command_results"].append(record)
      if "minimal" in label.lower() and stdout and record["stdout"].strip().splitlines()[-1] != "[2.0, 5.0, 10.0, 17.0]":
        raise QualificationError("minimal harness did not emit the exact four-value result")
      if returncode and not classify: raise QualificationError(f"{label} command failed: {returncode}")
      return returncode
    def sample(seconds, label):
      start = clock(); first = checked_status(f"{label}:start")
      while clock() - start < seconds:
        sleeper(min(sample_interval_s, max(0, seconds - (clock() - start))))
        checked_status(f"{label}:sample-{len(evidence['samples'])}")
      validate_cadence(first, evidence["samples"][-1])
    if gate == "A0":
      if handshake_reader is None: raise QualificationError("A0 requires diagnostic handshake")
      if install_provenance is None: raise QualificationError("A0 requires audited install provenance")
      handshake = handshake_reader(); status = checked_status("A0"); power = evidence["power_samples"][-1]
      evidence["provenance"] = {"installed_executable":str(installed_executable.resolve()) if installed_executable else None,
                                "install":install_provenance, "handshake":handshake, "status":status,
                                "power_residency":power, "endpoint_visible":True}
    elif gate == "A1":
      if installed_executable is None: raise QualificationError("A1 requires --installed-executable")
      evidence["process_census_initial"] = process_reader(); evidence["socket_reachable_initial"] = socket_reader()
      for pid in exact_server_pids(installed_executable, evidence["process_census_initial"]): terminator(pid)
      deadline = clock() + 10
      while exact_server_pids(installed_executable, process_reader()) and clock() < deadline: sleeper(.1)
      evidence["process_census_before_first_status"] = process_reader()
      evidence["socket_reachable_before_first_status"] = socket_reader()
      if exact_server_pids(installed_executable, evidence["process_census_before_first_status"]): raise QualificationError("server remained after TERM")
      if evidence["socket_reachable_before_first_status"]: raise QualificationError("TinyGPU socket remained reachable")
      first = checked_status("A1:first"); first_power = evidence["power_samples"][-1]
      sleeper(120); second = checked_status("A1:second"); second_power = evidence["power_samples"][-1]
      validate_cadence(first, second); validate_power_continuity(first_power, second_power, require_canary_advance=True)
      evidence["process_census_after_second_status"] = process_reader(); evidence["socket_reachable_after_second_status"] = socket_reader()
      if exact_server_pids(installed_executable, evidence["process_census_after_second_status"]): raise QualificationError("server restarted during A1")
      if evidence["socket_reachable_after_second_status"]: raise QualificationError("TinyGPU socket became reachable during A1")
    elif gate == "A2":
      before = checked_status("A2:before"); command(minimal_command, "minimal"); after = checked_status("A2:after"); validate_continuity(before, after)
    elif gate == "A3":
      before = checked_status("A3:before")
      for index in range(churn_count):
        command(minimal_command, f"minimal-{index+1}"); sleeper(churn_idle_s); checked_status(f"A3:after-{index+1}")
      (validate_cadence if churn_count * churn_idle_s >= 2 else validate_continuity)(before, evidence["samples"][-1])
    elif gate == "A4":
      if not include_post_idle: raise QualificationError("A4 must include its immediate A5 post-idle compute")
      sample(5400 if idle_s is None else idle_s, "A4")
      before = evidence["samples"][-1]; command(minimal_command, "A5 post-idle minimal"); after = checked_status("A5:after"); validate_continuity(before, after)
      evidence["A5"] = "passed"
    elif gate == "A5":
      raise QualificationError("A5 is only the immediate final step of A4 --include-post-idle")
    elif gate == "A6":
      if model is None: raise QualificationError("A6 requires --model")
      before = checked_status("A6:before"); command([*bench_command, "--model", str(model), "--prefill", "--prefill-mode", "smoke"], "8B smoke")
      after = checked_status("A6:after"); validate_continuity(before, after)
    elif gate == "A7":
      if model is None: raise QualificationError("A7 requires --model")
      decode_s = 600 if duration_s is None else duration_s
      for index in range(5):
        before = checked_status(f"A7:decode-{index+1}:before")
        command([*bench_command, "--model", str(model), "--decode", "--decode-duration-s", str(decode_s)], f"decode-{index+1}")
        after = checked_status(f"A7:decode-{index+1}:after"); validate_continuity(before, after)
        sample(900 if idle_s is None else idle_s, f"A7:idle-{index+1}")
    elif gate == "A8":
      if model is None: raise QualificationError("A8 requires --model")
      sample(28800 if idle_s is None else idle_s, "A8")
      before = checked_status("A8:A2:before"); command(minimal_command, "A8 internal A2 minimal")
      after = checked_status("A8:A2:after"); validate_continuity(before, after)
      before = checked_status("A8:A6:before"); command([*bench_command, "--model", str(model), "--prefill", "--prefill-mode", "smoke"], "A8 internal A6 smoke")
      after = checked_status("A8:A6:after"); validate_continuity(before, after)
    elif gate == "A9":
      before = checked_status("A9:before"); manual_prompt("manual replug"); after = checked_status("A9:after-replug")
      if after["provider_generation"] <= before["provider_generation"]: raise QualificationError("replug did not rebind provider")
      command(minimal_command, "post-replug minimal"); final = checked_status("A9:after-minimal"); validate_continuity(after, final)
      evidence["manual_action_required"] = True
    elif gate == "A10":
      if model is None: raise QualificationError("A10 requires --model")
      before = checked_status("A10:before")
      outcome = command([*bench_command, "--model", str(model), "--decode", "--decode-duration-s", str(1800 if duration_s is None else duration_s)],
                        "classification decode", classify=True)
      after = checked_status("A10:after"); validate_continuity(before, after)
      evidence["classification"] = {"workload_returncode":outcome, "endpoint_visible_after":True}
    elif gate == "A11":
      before = checked_status("A11:before"); manual_prompt("manual sleep/wake"); after = checked_status("A11:after")
      evidence["classification"] = {"generation_before":before["provider_generation"], "generation_after":after["provider_generation"],
                                    "endpoint_visible_after":True}; evidence["manual_action_required"] = True
    else: raise QualificationError("unknown gate")
  except BaseException as exc:
    first_failure = {"type":type(exc).__name__, "message":str(exc)}
  evidence["status"] = "recorded" if gate in CLASSIFICATION_GATES else ("passed" if first_failure is None else "failed")
  evidence["first_failure"] = first_failure
  return evidence


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--gate", choices=GATES, required=True)
  ap.add_argument("--model", type=pathlib.Path)
  ap.add_argument("--include-post-idle", action="store_true")
  args = ap.parse_args(argv)
  created = time.time_ns(); lock = context = environment = provenance = None
  previous_handlers = {}
  try:
    lock = validate_lock()
    if pathlib.Path(os.environ["TINYGRAD_GPU_LOCK_PATH"]).resolve() != DEFAULT_LOCK.resolve(): raise QualificationError(f"acceptance requires {DEFAULT_LOCK}")
    if pathlib.Path(lock.get("cwd", "")).resolve() != ROOT: raise QualificationError("lock runner cwd is not the feature worktree")
    environment = validate_environment()
    provenance = validate_install_provenance(DEFAULT_INSTALL_PROVENANCE, DEFAULT_APP)
    for sig in (signal.SIGINT, signal.SIGTERM):
      previous_handlers[sig] = signal.signal(sig, lambda signum, _frame: (_ for _ in ()).throw(QualificationInterrupted(f"signal {signum}")))
    status_cmd = [str(DEFAULT_APP), "keepalive", "status"]
    power_cmd = [str(DEFAULT_APP), "power", "status"]
    hello_cmd = [str(DEFAULT_APP), "keepalive", "handshake"]
    minimal = [sys.executable, str(ROOT / "extra/usbgpu/tests/minimal_amd_compute.py")]
    bench = [sys.executable, str(ROOT / "extra/llm_research/bench.py")]
    evidence = run_gate(args.gate, status_reader=lambda:status_command(status_cmd), power_status_reader=lambda:status_command(power_cmd),
                        handshake_reader=lambda:handshake_command(hello_cmd),
                        installed_executable=DEFAULT_APP, model=args.model, minimal_command=minimal, bench_command=bench,
                        endpoint_reader=default_endpoint_reader, process_reader=default_process_reader, terminator=lambda pid:os.kill(pid, signal.SIGTERM),
                        socket_reader=lambda:socket_reachable(DEFAULT_SOCKET), include_post_idle=args.include_post_idle, install_provenance=provenance,
                        manual_prompt=lambda action:input(f"Operator action required: {action}. Press Enter after completion: "))
    context = common_context(DEFAULT_APP)
  except BaseException as exc:
    evidence = {"gate":args.gate, "status":"recorded" if args.gate in CLASSIFICATION_GATES else "failed", "samples":[], "power_samples":[], "endpoint_checks":[],
                "commands":[], "command_results":[], "first_failure":{"type":type(exc).__name__, "message":str(exc)}}
  finally:
    for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
  evidence["lock"], evidence["environment"], evidence["install_provenance"] = lock, environment, provenance
  evidence["context"], evidence["created_unix_ns"] = context, created
  out = ROOT / "docs/task_workflow/output" / f"egpu-usb4-persistent-pcie-{args.gate}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}.json"
  atomic_json(out, evidence); print(out)
  return 0 if evidence["status"] in {"passed", "recorded"} else 1


if __name__ == "__main__": raise SystemExit(main())
