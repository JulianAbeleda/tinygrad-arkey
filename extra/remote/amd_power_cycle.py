#!/usr/bin/env python3
import argparse, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 6667

def stamp(msg:str):
  print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

def parse_int(x:str) -> int:
  return int(x, 0)

def script_path(name:str) -> str:
  return str(Path(__file__).resolve().parent / name)

class PowerBackend:
  def off(self): raise NotImplementedError
  def on(self): raise NotImplementedError

class DryRunBackend(PowerBackend):
  def off(self): stamp("dry-run power off")
  def on(self): stamp("dry-run power on")

class ShellyBackend(PowerBackend):
  def __init__(self, host:str, timeout:float):
    self.host, self.timeout = host, timeout

  def _set(self, state:str):
    url = f"http://{self.host}/relay/0?turn={state}"
    stamp(f"shelly {state} {url}")
    with urllib.request.urlopen(url, timeout=self.timeout) as resp:
      body = resp.read(4096)
    stamp(f"shelly {state} ok status={resp.status} bytes={len(body)}")

  def off(self): self._set("off")
  def on(self): self._set("on")

def mac_gpu_visible(device_id:int) -> bool:
  if sys.platform != "darwin": return True
  out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, check=False).stdout
  return f"Device ID: {device_id:#06x}" in out

def wait_for_gpu(device_id:int, timeout:float, poll:float):
  deadline, attempt = time.monotonic() + timeout, 1
  while True:
    stamp(f"poll gpu device_id={device_id:#06x} attempt={attempt}")
    if mac_gpu_visible(device_id):
      stamp(f"gpu visible device_id={device_id:#06x}")
      return
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise TimeoutError(f"GPU {device_id:#06x} did not appear within {timeout:.1f}s")
    time.sleep(min(poll, remaining))
    attempt += 1

def bridge_pids(port:int) -> list[int]:
  res = subprocess.run(["pgrep", "-afil", "extra/remote/serve.py"], capture_output=True, text=True, check=False)
  pids = []
  for line in res.stdout.splitlines():
    parts = line.split(maxsplit=1)
    if len(parts) != 2: continue
    pid_s, cmd = parts
    if pid_s.isdigit() and f" {port}" in f" {cmd} ":
      pids.append(int(pid_s))
  return pids

def stop_bridge(port:int, timeout:float):
  pids = bridge_pids(port)
  if not pids:
    stamp(f"bridge not running port={port}")
    return
  stamp(f"stop bridge port={port} pids={','.join(map(str, pids))}")
  for pid in pids:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not bridge_pids(port):
      stamp("bridge stopped")
      return
    time.sleep(0.25)
  pids = bridge_pids(port)
  if not pids:
    stamp("bridge stopped")
    return
  stamp(f"bridge still running, kill pids={','.join(map(str, pids))}")
  for pid in pids:
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
  time.sleep(0.25)

def start_bridge(bridge_script:str, port:int, startup_seconds:float) -> subprocess.Popen:
  stamp(f"start bridge script={bridge_script} port={port}")
  env = os.environ.copy()
  env.setdefault("DEBUG", "1")
  proc = subprocess.Popen([sys.executable, bridge_script, str(port)], cwd=str(ROOT), env=env)
  time.sleep(startup_seconds)
  if proc.poll() is not None: raise RuntimeError(f"bridge exited early status={proc.returncode}")
  stamp(f"bridge started pid={proc.pid}")
  return proc

def run_check(cmd:list[str], env:dict[str, str]):
  stamp(f"run {' '.join(cmd)}")
  subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

def run_health_checks(port:int):
  remote = f"127.0.0.1:{port}"
  env = os.environ.copy()
  env.setdefault("REMOTE_TIMEOUT", "3")
  env.setdefault("REMOTE_RPC_TIMEOUT", env["REMOTE_TIMEOUT"])
  run_check([sys.executable, script_path("bench.py"), remote, "--skip-tensor"], env)
  env = os.environ.copy()
  env.setdefault("REMOTE_TIMEOUT", "5")
  env.setdefault("REMOTE_RPC_TIMEOUT", "10")
  run_check([sys.executable, script_path("amd_repro.py"), remote, "--stage", "psp-status"], env)

def build_backend(args) -> PowerBackend:
  if args.backend == "dry-run": return DryRunBackend()
  if args.backend == "shelly":
    if not args.host: raise SystemExit("--host is required with --backend shelly")
    return ShellyBackend(args.host, args.http_timeout)
  raise AssertionError(args.backend)

def main():
  parser = argparse.ArgumentParser(description="Power-cycle a macOS AMD eGPU test rig and restart the TinyGPU bridge")
  parser.add_argument("--backend", choices=("shelly", "dry-run"), required=True)
  parser.add_argument("--host", help="Shelly host or IP address")
  parser.add_argument("--bridge-port", type=parse_int, default=DEFAULT_PORT)
  parser.add_argument("--bridge-script", default=script_path("serve.py"))
  parser.add_argument("--device-id", type=parse_int, default=0x744c)
  parser.add_argument("--power-off-seconds", type=float, default=8.0)
  parser.add_argument("--enumerate-timeout", type=float, default=60.0)
  parser.add_argument("--poll-seconds", type=float, default=5.0)
  parser.add_argument("--startup-seconds", type=float, default=2.0)
  parser.add_argument("--stop-timeout", type=float, default=3.0)
  parser.add_argument("--http-timeout", type=float, default=5.0)
  parser.add_argument("--skip-power", action="store_true", help="only stop/start bridge and run checks")
  parser.add_argument("--no-health-check", action="store_true", help="skip bench.py and psp-status checks")
  args = parser.parse_args()

  backend, powered_off = build_backend(args), False
  if args.poll_seconds <= 0: raise SystemExit("--poll-seconds must be positive")
  if args.enumerate_timeout < 0: raise SystemExit("--enumerate-timeout must be non-negative")
  if not Path(args.bridge_script).is_file(): raise SystemExit(f"bridge script not found: {args.bridge_script}")

  try:
    stop_bridge(args.bridge_port, args.stop_timeout)
    if not args.skip_power:
      backend.off()
      powered_off = True
      stamp(f"wait power_off_seconds={args.power_off_seconds:.1f}")
      time.sleep(args.power_off_seconds)
      backend.on()
      powered_off = False
    else:
      stamp("skip power cycle")
    wait_for_gpu(args.device_id, args.enumerate_timeout, args.poll_seconds)
    start_bridge(args.bridge_script, args.bridge_port, args.startup_seconds)
    if not args.no_health_check: run_health_checks(args.bridge_port)
    else: stamp("skip health checks")
    stamp("power-cycle helper complete")
  except KeyboardInterrupt:
    stamp("interrupted")
    if powered_off:
      try:
        backend.on()
        powered_off = False
      except Exception as e:
        stamp(f"WARNING failed to restore power: {e}")
    raise SystemExit(130)
  except Exception as e:
    if powered_off:
      try:
        backend.on()
        powered_off = False
      except Exception as restore_e:
        stamp(f"WARNING failed to restore power: {restore_e}")
    stamp(f"failed: {e}")
    raise SystemExit(1)

if __name__ == "__main__":
  main()
