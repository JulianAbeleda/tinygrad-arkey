#!/usr/bin/env python3
"""Clock telemetry for current pp512 authority surfaces.

Measures:
  - tinygrad clean WMMA PREFILL_V2
  - tinygrad + Tensile research route
  - llama.cpp llama-bench pp512

The tinygrad path deliberately captures each TinyJit before changing any route
global and asserts captured PROGRAM identity. This avoids the old OFF-arm
Tensile flag leak.
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, signal, statistics, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "bench/qk-prefill-clock-threeway"
DEV = pathlib.Path("/sys/class/drm/card0/device")
HWMON = next((DEV / "hwmon").glob("hwmon*"), None) if (DEV / "hwmon").exists() else None
MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LLAMA_BENCH = "/home/ubuntu/env/llama.cpp/build/bin/llama-bench"


def read_text(path: pathlib.Path, default: str = "") -> str:
  try:
    return path.read_text().strip()
  except OSError:
    return default


def active_mhz(name: str) -> int:
  for line in read_text(DEV / name).splitlines():
    if "*" in line:
      match = re.search(r"(\d+)Mhz", line)
      return int(match.group(1)) if match else 0
  return 0


def rocm_sclk_mhz() -> int:
  try:
    out = subprocess.run(["rocm-smi", "--showgpuclocks"], capture_output=True, text=True, timeout=4).stdout
  except Exception:
    return 0
  for line in out.splitlines():
    if "sclk clock level" in line:
      match = re.search(r"\((\d+)Mhz\)", line)
      return int(match.group(1)) if match else 0
  return 0


def sample_clock() -> dict[str, Any]:
  row: dict[str, Any] = {
    "t": time.time(),
    "rocm_sclk": rocm_sclk_mhz(),
    "dpm_sclk": active_mhz("pp_dpm_sclk"),
    "dpm_mclk": active_mhz("pp_dpm_mclk"),
    "gpu_busy": int(read_text(DEV / "gpu_busy_percent", "0") or 0),
    "mem_busy": int(read_text(DEV / "mem_busy_percent", "0") or 0),
    "perf_level": read_text(DEV / "power_dpm_force_performance_level"),
  }
  if HWMON is not None:
    row["power_w"] = round(int(read_text(HWMON / "power1_average", "0") or 0) / 1e6, 1)
    row["temp_c"] = round(int(read_text(HWMON / "temp1_input", "0") or 0) / 1e3, 1)
  return row


class ClockSampler:
  def __init__(self, interval_s: float = 0.2):
    self.interval_s = interval_s
    self.log_path = pathlib.Path(f"/tmp/qk_clock_threeway_{os.getpid()}_{time.time_ns()}.log")
    self._file = None
    self._proc = None

  def start(self) -> None:
    hwpower = str(HWMON / "power1_average") if HWMON is not None else "/dev/null"
    hwtemp = str(HWMON / "temp1_input") if HWMON is not None else "/dev/null"
    cmd = f"""
while true; do
  t=$(date +%s.%N)
  s=$(rocm-smi --showgpuclocks 2>/dev/null | sed -n 's/.*sclk clock level.*(\\([0-9][0-9]*\\)Mhz).*/\\1/p' | head -n1)
  ds=$(awk '/\\*/ {{gsub(/Mhz/,"",$2); print $2}}' {DEV}/pp_dpm_sclk 2>/dev/null)
  dm=$(awk '/\\*/ {{gsub(/Mhz/,"",$2); print $2}}' {DEV}/pp_dpm_mclk 2>/dev/null)
  gb=$(cat {DEV}/gpu_busy_percent 2>/dev/null || echo 0)
  mb=$(cat {DEV}/mem_busy_percent 2>/dev/null || echo 0)
  pw=$(cat {hwpower} 2>/dev/null || echo 0)
  tc=$(cat {hwtemp} 2>/dev/null || echo 0)
  pl=$(cat {DEV}/power_dpm_force_performance_level 2>/dev/null || echo unknown)
  echo "$t ${{s:-0}} ${{ds:-0}} ${{dm:-0}} ${{gb:-0}} ${{mb:-0}} ${{pw:-0}} ${{tc:-0}} $pl"
  sleep {self.interval_s}
done
"""
    self._file = self.log_path.open("w")
    self._proc = subprocess.Popen(["bash", "-c", cmd], stdout=self._file, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

  def finish(self) -> list[dict[str, Any]]:
    if self._proc is not None:
      try:
        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
      except Exception:
        pass
      try:
        self._proc.wait(timeout=3)
      except subprocess.TimeoutExpired:
        pass
    if self._file is not None:
      self._file.close()
    rows = []
    try:
      lines = self.log_path.read_text().splitlines()
    except OSError:
      lines = []
    for line in lines:
      parts = line.split()
      if len(parts) < 9:
        continue
      try:
        rows.append({
          "t": float(parts[0]),
          "rocm_sclk": int(parts[1]),
          "dpm_sclk": int(parts[2]),
          "dpm_mclk": int(parts[3]),
          "gpu_busy": int(parts[4]),
          "mem_busy": int(parts[5]),
          "power_w": round(int(parts[6]) / 1e6, 1),
          "temp_c": round(int(parts[7]) / 1e3, 1),
          "perf_level": parts[8],
        })
      except ValueError:
        continue
    try:
      self.log_path.unlink()
    except OSError:
      pass
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  active = [r for r in rows if r.get("gpu_busy", 0) > 0 or r.get("rocm_sclk", 0) > 800 or r.get("dpm_sclk", 0) > 800]

  def stats(rs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [r.get(key, 0) for r in rs if isinstance(r.get(key, 0), (int, float))]
    return {
      "median": statistics.median(vals) if vals else 0,
      "min": min(vals) if vals else 0,
      "max": max(vals) if vals else 0,
    }

  def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
      "n": len(rs),
      "rocm_sclk_mhz": stats(rs, "rocm_sclk"),
      "dpm_sclk_mhz": stats(rs, "dpm_sclk"),
      "dpm_mclk_mhz": stats(rs, "dpm_mclk"),
      "gpu_busy_pct": stats(rs, "gpu_busy"),
      "mem_busy_pct": stats(rs, "mem_busy"),
      "power_w": stats(rs, "power_w"),
      "temp_c": stats(rs, "temp_c"),
      "perf_level_last": rs[-1].get("perf_level", "") if rs else "",
    }

  return {"all_samples": block(rows), "active_samples": block(active)}


def sudo(cmd: str) -> bool:
  return subprocess.run(["sudo", "-n", "bash", "-c", cmd], capture_output=True, text=True).returncode == 0


def set_lane(lane: str) -> bool:
  if lane == "auto":
    return sudo(f"echo auto > {DEV}/power_dpm_force_performance_level")
  if lane == "manual_peak":
    return sudo(
      f"echo manual > {DEV}/power_dpm_force_performance_level && "
      f"echo 2 > {DEV}/pp_dpm_sclk && echo 3 > {DEV}/pp_dpm_mclk"
    )
  if lane in ("high", "profile_peak"):
    return sudo(f"echo {lane} > {DEV}/power_dpm_force_performance_level")
  raise ValueError(f"unknown lane {lane}")


def restore_lane() -> None:
  sudo(f"echo auto > {DEV}/power_dpm_force_performance_level")


def write_result(name: str, data: dict[str, Any]) -> None:
  ART.mkdir(parents=True, exist_ok=True)
  (ART / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
  latest = ART / "latest.json"
  if latest.exists():
    try:
      cur = json.loads(latest.read_text())
    except json.JSONDecodeError:
      cur = {}
  else:
    cur = {}
  cur[f"{data['engine']}:{data['lane']}"] = data
  latest.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n")


def captured_program_names(jit: Any) -> list[str]:
  return [
    getattr(u.arg, "name", "")
    for u in jit.captured.linear.toposort()
    if getattr(u.op, "name", "") == "PROGRAM"
  ]


def run_tinygrad(engine: str, lane: str, n: int, warm: int) -> dict[str, Any]:
  import tinygrad.codegen.opt.postrange as pr
  import tinygrad.llm.model as Mod
  import extra.qk_tensile_inmodel as TI
  from tinygrad import Device, Tensor, TinyJit
  from tinygrad.llm.model import PREFILL_UBATCH, Transformer

  assert os.environ.get("PREFILL_V2"), "run with PREFILL_V2=1"
  flag = engine == "tensile"
  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(MODEL, 2048)
  TI.install(Device[Device.DEFAULT])
  toks = Tensor([5, 6, 7, 8, 9, 10] * 200 + [0] * (2048 - 1200), dtype="int32").reshape(1, 2048)
  chunk = toks[:, 0:PREFILL_UBATCH].contiguous()
  temp = Tensor([0.0])
  for block in model.blk:
    block._use_flash, block._prefill_v2 = False, True
  saved = pr._WARMSTART_OPTS

  def build() -> Any:
    Mod.PREFILL_TENSILE_GEMM = flag
    return TinyJit(model.forward)

  def run_once(jit: Any) -> None:
    Mod.PREFILL_TENSILE_GEMM = flag
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try:
      jit(chunk, 0, temp).realize()
      Device[Device.DEFAULT].synchronize()
    finally:
      pr._WARMSTART_OPTS = saved

  TI.ROUTE_COUNT.clear()
  jit = build()
  run_once(jit)  # trace
  run_once(jit)  # capture under the same flag
  names = captured_program_names(jit)
  has_tensile = any("tensile" in name for name in names)
  if flag:
    assert has_tensile, "Tensile run did not capture Tensile programs"
  else:
    assert not has_tensile, "WMMA run leaked Tensile programs"

  for _ in range(warm):
    run_once(jit)
  sampler = ClockSampler()
  sampler.start()
  times = []
  for _ in range(n):
    t0 = time.perf_counter()
    run_once(jit)
    times.append(time.perf_counter() - t0)
  rows = sampler.finish()
  result = {
    "date": "2026-06-20",
    "engine": engine,
    "lane": lane,
    "model": MODEL,
    "n_prompt": int(PREFILL_UBATCH),
    "n_meas": n,
    "warm": warm,
    "tok_s_median": PREFILL_UBATCH / statistics.median(times),
    "tok_s_best": PREFILL_UBATCH / min(times),
    "times_s": times,
    "route_count": dict(TI.ROUTE_COUNT),
    "captured_tensile_programs": [name for name in names if "tensile" in name],
    "telemetry": summarize(rows),
  }
  write_result(f"{engine}_{lane}.json", result)
  return result


def parse_json_array(text: str) -> Any:
  start, end = text.find("["), text.rfind("]")
  if start < 0 or end < start:
    raise ValueError(f"no JSON array in llama-bench output: {text[:500]}")
  return json.loads(text[start:end + 1])


def run_llama(lane: str, reps: int) -> dict[str, Any]:
  cmd = [LLAMA_BENCH, "-m", MODEL, "-p", "512", "-n", "0", "-r", str(reps), "-o", "json"]
  sampler = ClockSampler()
  sampler.start()
  proc = subprocess.run(cmd, capture_output=True, text=True)
  rows = sampler.finish()
  combined = proc.stdout + "\n" + proc.stderr
  bench = parse_json_array(combined)
  row = bench[0] if bench else {}
  samples_ts = row.get("samples_ts") or []
  result = {
    "date": "2026-06-20",
    "engine": "llama",
    "lane": lane,
    "model": MODEL,
    "command": cmd,
    "returncode": proc.returncode,
    "build_commit": row.get("build_commit"),
    "build_number": row.get("build_number"),
    "gpu_info": row.get("gpu_info"),
    "n_prompt": row.get("n_prompt"),
    "n_gen": row.get("n_gen"),
    "repetitions": reps,
    "tok_s_avg": row.get("avg_ts"),
    "tok_s_median": statistics.median(samples_ts) if samples_ts else row.get("avg_ts"),
    "tok_s_stddev": row.get("stddev_ts"),
    "samples_tok_s": samples_ts,
    "samples_ns": row.get("samples_ns"),
    "telemetry": summarize(rows),
    "stderr_tail": proc.stderr[-1000:],
  }
  if proc.returncode != 0:
    result["error"] = combined[-2000:]
  write_result(f"llama_{lane}.json", result)
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("engine", choices=["wmma", "tensile", "llama"])
  parser.add_argument("--lane", choices=["auto", "manual_peak", "high", "profile_peak"], default="auto")
  parser.add_argument("--n", type=int, default=25)
  parser.add_argument("--warm", type=int, default=8)
  parser.add_argument("--llama-reps", type=int, default=10)
  args = parser.parse_args()

  ok = set_lane(args.lane)
  time.sleep(0.5)
  try:
    if args.engine == "llama":
      result = run_llama(args.lane, args.llama_reps)
    else:
      result = run_tinygrad(args.engine, args.lane, args.n, args.warm)
  finally:
    if args.lane != "auto":
      restore_lane()

  print(json.dumps({
    "out": str(ART / f"{args.engine}_{args.lane}.json"),
    "lane_set_ok": ok,
    "engine": result["engine"],
    "lane": result["lane"],
    "tok_s_median": result.get("tok_s_median"),
    "tok_s_best": result.get("tok_s_best"),
    "telemetry_active": result["telemetry"]["active_samples"],
  }, indent=2, sort_keys=True))
  return 0 if result.get("returncode", 0) == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
