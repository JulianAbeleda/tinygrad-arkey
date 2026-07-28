#!/usr/bin/env python3
"""Canonical QK benchmark entry point.

Dispatches to the repo's blessed measurement authorities in isolated
subprocesses. Report throughput from this entry point, not from generate TTFT.

Entry-point hardening (see docs/decode-fix-and-fault-scope-20260726.md Phase 1):
a dispatch target that does not exist, or a sub-run that produces no parsable
throughput number, must fail loudly -- a non-zero/None result must never pass
silently through `main`'s OR-of-returncodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extra.qk.prefill.prefill_harness import (
  DEFAULT_MODEL_PROFILE, MODEL_HARNESS_ALIASES, MODEL_HARNESS_PROFILES, PREFILL_MODES, csv_ints,
  prefill_authority_argv, prefill_run_profile,
  prefill_subprocess_env, resolve_prefill_model_profile,
)
from extra.qk.decode.decode_harness import (
  csv_ints as decode_csv_ints, decode_authority_argv, decode_run_profile, decode_subprocess_env,
)
from extra.qk.timing_harness import add_clock_pin_arg

# decode_runtime_overhead.py prints: "ctx  512: W  8.67ms (115.28 tok/s) | D ..."
DECODE_THROUGHPUT_RE = re.compile(r"ctx\s*\d+:\s*W\s*[\d.]+ms\s*\(([\d.]+)\s*tok/s\)")
# prefill_whole_synced.py prints: "  WHOLE-PREFILL@512: 4017 tok/s"
PREFILL_THROUGHPUT_RE = re.compile(r"WHOLE-PREFILL@\d+:\s*([\d.]+)\s*tok/s")

# 14B authority was promoted and recorded at these flash-decode checkpoints. The generic ctx128
# checkpoint exercises a separate 14B prompt-finalization compiler defect (non-assignable float32
# vector store) before decode timing begins; keep it available only as an explicit diagnostic.
DECODE_DEFAULT_CKPTS_BY_PROFILE = {"qwen3_14b_q4k_m_gfx1100": (512, 1024, 4096)}
DECODE_DURATION_SCHEMA = "tinygrad.qk.decode.duration.v1"
DECODE_DURATION_DEFAULT_TIMEOUT_S = 900.0
DECODE_DURATION_KILL_GRACE_S = 5.0


def _decode_ckpts(raw:str|None, model_profile_id:str) -> tuple[int, ...]|None:
  return decode_csv_ints(raw) if raw else DECODE_DEFAULT_CKPTS_BY_PROFILE.get(model_profile_id)


class DispatchTargetMissing(RuntimeError):
  """A bench.py dispatch target (measurement core) does not exist on disk."""


class NoThroughputProduced(RuntimeError):
  """A sub-run exited without printing a single parsable throughput number."""


class BelowPerfFloor(RuntimeError):
  """A sub-run's measured throughput fell below the caller-supplied floor."""


class DurationInterrupted(RuntimeError):
  """A signal interrupted a duration-bounded decode run."""

  def __init__(self, signum: int):
    self.signum = signum
    super().__init__(f"interrupted by signal {signum}")


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.write("\n")
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary, path)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def _sha256_file(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _duration_output_path(raw: str | None) -> pathlib.Path:
  if raw:
    return pathlib.Path(raw).expanduser().resolve()
  return ROOT / "bench" / "qk-decode-duration" / f"run-{time.time_ns()}-{os.getpid()}.json"


def _duration_controls(env_extra: dict[str, str]) -> dict[str, str]:
  names = ("DEV", "JIT", "PYTHONPATH", "AM_REMOTE_DISCOVERY_PROFILE", "AM_REMOTE_SKIP_RESIZE_BAR",
           "REMOTE_KEEPALIVE_S", "FLASH_DECODE", "FLASH_DECODE_THRESHOLD", "QK_MODEL")
  resolved = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
  return {name: resolved[name] for name in names if name in resolved}


def _stop_process_group(proc: subprocess.Popen[str]) -> tuple[str, str, bool]:
  """Terminate a child group and return its final output plus whether SIGKILL was needed."""
  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except ProcessLookupError:
    pass
  try:
    stdout, stderr = proc.communicate(timeout=DECODE_DURATION_KILL_GRACE_S)
    return stdout, stderr, False
  except subprocess.TimeoutExpired:
    try:
      os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
    stdout, stderr = proc.communicate()
    return stdout, stderr, True


def _run_decode_duration(*, model: str, profile, reps: int, timeout_s: float, duration_s: float,
                         out_path: pathlib.Path, bench_argv: list[str], min_value: float | None = None) -> int:
  """Run complete decode-authority cycles until the monotonic duration deadline."""
  model_path = pathlib.Path(model).expanduser().resolve()
  model_stat = model_path.stat()
  model_sha256 = _sha256_file(model_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  cycle_dir = out_path.parent / f"{out_path.stem}.cycles"
  env_extra = decode_subprocess_env(str(model_path))
  started_monotonic_ns, started_unix_ns = time.monotonic_ns(), time.time_ns()
  deadline_ns = started_monotonic_ns + int(duration_s * 1e9)
  cycles, first_failure = [], None
  status, signal_number = "passed", None
  active_proc = None

  def interrupted(signum: int, _frame) -> None:
    raise DurationInterrupted(signum)

  previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
  try:
    while not cycles or time.monotonic_ns() < deadline_ns:
      sequence = len(cycles) + 1
      child_path = cycle_dir / f"cycle-{sequence:06d}.json"
      child_argv = decode_authority_argv(str(model_path), profile, out_path=child_path, reps=reps)
      # Keep this explicit rather than routing duration cycles through `_run`: a cycle needs a process group,
      # timeout, and an individually addressable authority artifact.
      _verify_dispatch_target("DECODE duration", child_argv)
      env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
      started_cycle_monotonic_ns, started_cycle_unix_ns = time.monotonic_ns(), time.time_ns()
      active_proc = subprocess.Popen([sys.executable, *child_argv], cwd=str(ROOT), env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
      timed_out = killed = False
      try:
        stdout, stderr = active_proc.communicate(timeout=timeout_s)
      except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr, killed = _stop_process_group(active_proc)
      ended_cycle_monotonic_ns, ended_cycle_unix_ns = time.monotonic_ns(), time.time_ns()
      sys.stdout.write(stdout); sys.stderr.write(stderr)
      sys.stdout.flush(); sys.stderr.flush()
      throughputs = [float(value) for value in DECODE_THROUGHPUT_RE.findall(stdout + stderr)]
      cycle = {
        "sequence": sequence,
        "argv": [sys.executable, *child_argv],
        "started_monotonic_ns": started_cycle_monotonic_ns,
        "ended_monotonic_ns": ended_cycle_monotonic_ns,
        "started_unix_ns": started_cycle_unix_ns,
        "ended_unix_ns": ended_cycle_unix_ns,
        "elapsed_s": (ended_cycle_monotonic_ns - started_cycle_monotonic_ns) / 1e9,
        "returncode": active_proc.returncode,
        "timed_out": timed_out,
        "killed": killed,
        "throughput_tok_s": throughputs,
      }
      active_proc = None
      if timed_out:
        # A timeout is the primary cause; a missing artifact is expected once the child is terminated.
        first_failure = first_failure or {"kind": "cycle_timeout", "cycle": sequence}
      if child_path.exists():
        child_bytes = child_path.read_bytes()
        cycle["artifact_path"] = os.path.relpath(child_path, out_path.parent)
        cycle["artifact_sha256"] = hashlib.sha256(child_bytes).hexdigest()
        try:
          child_payload = json.loads(child_bytes)
          if child_payload.get("schema") != "tinygrad.decode.fixed_depth.v2":
            raise ValueError(f"unexpected child schema {child_payload.get('schema')!r}")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
          first_failure = first_failure or {"kind": "invalid_child_artifact", "message": str(exc), "cycle": sequence}
      else:
        first_failure = first_failure or {"kind": "missing_child_artifact", "cycle": sequence}
      cycles.append(cycle)
      if timed_out:
        status = "cycle_timeout"
        break
      if active_proc is None and cycle["returncode"] != 0:
        status = "child_failed"
        first_failure = first_failure or {"kind": "child_exit", "cycle": sequence, "returncode": cycle["returncode"]}
        break
      if not throughputs:
        status = "child_failed"
        first_failure = first_failure or {"kind": "no_throughput", "cycle": sequence}
        break
      if min_value is not None and min(throughputs) < min_value:
        status = "below_performance_floor"
        first_failure = first_failure or {
          "kind": "below_performance_floor", "cycle": sequence,
          "minimum_required_tok_s": min_value, "observed_minimum_tok_s": min(throughputs),
        }
        break
      if first_failure is not None:
        status = "child_failed"
        break
  except DurationInterrupted as exc:
    status, signal_number = "interrupted", exc.signum
    first_failure = first_failure or {"kind": "signal", "signal": exc.signum}
    if active_proc is not None and active_proc.poll() is None:
      _stop_process_group(active_proc)
  except BaseException as exc:
    status = "internal_error"
    first_failure = first_failure or {"kind": type(exc).__name__, "message": str(exc)}
    if active_proc is not None and active_proc.poll() is None:
      _stop_process_group(active_proc)
  finally:
    for sig, handler in previous.items(): signal.signal(sig, handler)
    ended_monotonic_ns, ended_unix_ns = time.monotonic_ns(), time.time_ns()
    aggregate = {
      "schema": DECODE_DURATION_SCHEMA,
      "artifact_version": 1,
      "status": status,
      "first_failure": first_failure,
      "bench_argv": bench_argv,
      "environment_controls": _duration_controls(env_extra),
      "model": {"path": str(model_path), "size_bytes": model_stat.st_size, "sha256": model_sha256},
      "requested_duration_s": duration_s,
      "minimum_decode_tok_s": min_value,
      "cycle_timeout_s": timeout_s,
      "kill_grace_s": DECODE_DURATION_KILL_GRACE_S,
      "started_monotonic_ns": started_monotonic_ns,
      "ended_monotonic_ns": ended_monotonic_ns,
      "started_unix_ns": started_unix_ns,
      "ended_unix_ns": ended_unix_ns,
      "actual_duration_s": (ended_monotonic_ns - started_monotonic_ns) / 1e9,
      "cycles": cycles,
    }
    _atomic_json(out_path, aggregate)
  return 0 if status == "passed" else (128 + signal_number if signal_number is not None else 1)


def _verify_dispatch_target(desc: str, argv: list[str]) -> None:
  target = ROOT / argv[0]
  if not target.exists():
    raise DispatchTargetMissing(
      f"{desc}: dispatch target does not exist: {target} "
      f"(argv[0]={argv[0]!r} -- the measurement core was deleted or moved without updating the argv builder)")


def _run(desc: str, argv: list[str], env_extra: dict[str, str], label: str = "authority",
         throughput_re: re.Pattern | None = None, min_value: float | None = None) -> int:
  _verify_dispatch_target(desc, argv)
  print(f"\n===== {desc} ({label}) =====", flush=True)
  env = {**os.environ, "PYTHONPATH": str(ROOT), **env_extra}
  proc = subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env, check=False,
                        capture_output=True, text=True)
  sys.stdout.write(proc.stdout)
  sys.stderr.write(proc.stderr)
  sys.stdout.flush(); sys.stderr.flush()

  if throughput_re is None:
    return proc.returncode

  # Scan BOTH streams. decode_runtime_overhead.py:203-207 prints its "ctx N: W ..ms (.. tok/s)" rows to stderr, so a
  # stdout-only scan made the decode authority path raise NoThroughputProduced on every successful run.
  values = [float(m) for m in throughput_re.findall(proc.stdout + proc.stderr)]
  if not values:
    raise NoThroughputProduced(
      f"{desc} ({label}) produced no parsable throughput number (rc={proc.returncode}) -- "
      f"treating as failure even though the child may have exited 0. Expected a match for "
      f"{throughput_re.pattern!r} in its stdout.")

  if min_value is not None:
    worst = min(values)
    if worst < min_value:
      raise BelowPerfFloor(
        f"{desc} ({label}) throughput {worst:.2f} is below the required floor {min_value:.2f} "
        f"(all measured values: {values})")

  return proc.returncode


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True, help="GGUF path")
  ap.add_argument("--model-profile", default="", choices=("", *MODEL_HARNESS_PROFILES.keys(), *MODEL_HARNESS_ALIASES),
                  help=f"prefill model/profile defaults; default infers from --model or uses {DEFAULT_MODEL_PROFILE}")
  ap.add_argument("--prefill", action="store_true", help="prefill authority only")
  ap.add_argument("--decode", action="store_true", help="decode authority only")
  ap.add_argument("--prefill-mode", choices=PREFILL_MODES, default="authority")
  ap.add_argument("--prefill-K", type=int, default=None)
  ap.add_argument("--prefill-warmups", type=int, default=None)
  ap.add_argument("--prefill-rounds", type=int, default=None)
  ap.add_argument("--prefill-start-positions", default=None)
  ap.add_argument("--prefill-whole-lengths", default=None)
  ap.add_argument("--prefill-no-artifact", action="store_true", help="do not write prefill-whole-synced/latest.json")
  ap.add_argument("--prefill-artifact", default="", help="explicit prefill artifact path instead of latest.json")
  ap.add_argument("--prefill-require-route", default="", help="fail unless this exact prefill route is attributed")
  ap.add_argument("--decode-ckpts", default=None, help="comma-separated decode checkpoint contexts")
  ap.add_argument("--decode-nmeas", type=int, default=None, help="override decode measurements per context")
  ap.add_argument("--decode-max-context", type=int, default=None, help="override decode model max_context")
  ap.add_argument("--decode-reps", type=int, default=5, help="independent fixed-depth repetitions")
  ap.add_argument("--decode-out", default=None, help="decode artifact path (default: unique per invocation)")
  ap.add_argument("--decode-duration-s", type=float, default=None,
                  help="run complete decode-authority cycles for this many seconds (requires --decode)")
  ap.add_argument("--decode-cycle-timeout-s", type=float, default=DECODE_DURATION_DEFAULT_TIMEOUT_S,
                  help=f"per-cycle duration-mode timeout (default: {DECODE_DURATION_DEFAULT_TIMEOUT_S:g})")
  ap.add_argument("--decode-duration-out", default=None, help="duration aggregate JSON path")
  ap.add_argument("--min-decode", type=float, default=None, help="fail if measured decode tok/s falls below this")
  ap.add_argument("--min-prefill", type=float, default=None, help="fail if measured prefill tok/s falls below this")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)

  if args.decode_duration_s is not None:
    if not args.decode or args.prefill:
      ap.error("--decode-duration-s requires --decode and cannot be combined with --prefill")
    if args.decode_duration_s <= 0:
      ap.error("--decode-duration-s must be positive")
    if args.decode_cycle_timeout_s <= 0:
      ap.error("--decode-cycle-timeout-s must be positive")
    if args.decode_out is not None:
      ap.error("--decode-duration-s cannot be combined with --decode-out; use --decode-duration-out")
  elif args.decode_duration_out is not None:
    ap.error("--decode-duration-out requires --decode-duration-s")

  both = not (args.prefill or args.decode)
  model_profile = resolve_prefill_model_profile(args.model_profile or None, model_path=args.model)
  if args.decode_duration_s is not None:
    profile = decode_run_profile(ckpts=_decode_ckpts(args.decode_ckpts, model_profile.id),
                                 max_context=args.decode_max_context, nmeas=args.decode_nmeas)
    invocation_args = list(sys.argv[1:] if argv is None else argv)
    resolved_argv = [sys.executable, str(pathlib.Path(__file__).resolve()), *invocation_args]
    return _run_decode_duration(model=args.model, profile=profile, reps=args.decode_reps,
                                timeout_s=args.decode_cycle_timeout_s, duration_s=args.decode_duration_s,
                                out_path=_duration_output_path(args.decode_duration_out),
                                bench_argv=resolved_argv, min_value=args.min_decode)
  rc = 0
  if args.prefill or both:
    profile = prefill_run_profile(args.prefill_mode, K=args.prefill_K, warmups=args.prefill_warmups,
                                  rounds=args.prefill_rounds,
                                  start_positions=csv_ints(args.prefill_start_positions) if args.prefill_start_positions else None,
                                  whole_lengths=csv_ints(args.prefill_whole_lengths) if args.prefill_whole_lengths else None)
    rc = _run("PREFILL pp@L", prefill_authority_argv(args.model, profile, model_profile_id=model_profile.id, pin_clock=args.pin_clock,
                                                     artifact=not args.prefill_no_artifact,
                                                     require_route=args.prefill_require_route or None,
                                                     artifact_path=args.prefill_artifact or None),
              prefill_subprocess_env(model_profile_id=model_profile.id, model_path=args.model), label=f"{profile.mode}:{model_profile.id}",
              throughput_re=PREFILL_THROUGHPUT_RE, min_value=args.min_prefill) or rc
  if args.decode or both:
    profile = decode_run_profile(ckpts=_decode_ckpts(args.decode_ckpts, model_profile.id),
                                 max_context=args.decode_max_context, nmeas=args.decode_nmeas)
    decode_out = args.decode_out or str(ROOT / "bench" / "qk-decode-runtime-overhead" /
                                        f"run-{time.time_ns()}-{os.getpid()}.json")
    rc = _run("DECODE W==D", decode_authority_argv(args.model, profile, out_path=decode_out, reps=args.decode_reps),
              decode_subprocess_env(args.model), throughput_re=DECODE_THROUGHPUT_RE, min_value=args.min_decode) or rc
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
