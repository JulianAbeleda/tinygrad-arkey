#!/usr/bin/env python3
"""Native eager AMD PMC collection and liveness classification for MMQ research."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

SCHEMA = "tinygrad.amd_pmc_result.v1"
STATUSES = frozenset(("advertised", "live", "zero_suspect", "unsupported", "blocked"))
DEFAULT_GROUPS = (
  ("SQ_BUSY_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_WAIT_ANY"),
  ("SQC_LDS_IDX_ACTIVE", "SQC_LDS_BANK_CONFLICT", "SQ_INSTS_LDS", "SQ_WAIT_INST_LDS"),
  ("GL2C_HIT", "GL2C_MISS", "GL2C_MC_RDREQ", "GL2C_MC_WRREQ"),
  ("TA_BUFFER_LOAD_WAVEFRONTS", "TA_BUFFER_STORE_WAVEFRONTS"),
)


def _decode_event(event: Any) -> dict[str, int]:
  view, ptr, out = memoryview(event.blob).cast("Q"), 0, {}
  for sample in event.sched:
    count = sample.xcc * sample.inst * sample.se * sample.sa * sample.wgp
    out[sample.name] = sum(int(view[ptr+i]) for i in range(count))
    ptr += count
  return out


def _child_control(kind: str, size: int) -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.device import Compiled, Device
  if kind == "compute":
    a = Tensor(np.arange(size, dtype=np.float32), device="AMD").realize()
    Compiled.profile_events.clear()
    (a * 3.0 + 7.0).realize()
  elif kind == "memory":
    a = Tensor(np.arange(size, dtype=np.float32), device="AMD").realize()
    b = Tensor(np.arange(size, dtype=np.float32), device="AMD").realize()
    Compiled.profile_events.clear()
    (a + b).realize()
  elif kind in ("lds_free", "lds_conflict"):
    from tinygrad import dtypes
    from tinygrad.dtype import AddrSpace
    from tinygrad.uop.ops import KernelInfo, UOp
    a = Tensor(np.arange(32, dtype=np.int32), device="AMD").realize()
    stride = 1 if kind == "lds_free" else 32
    def kernel(out: UOp, src: UOp) -> UOp:
      lane = UOp.special(32, "lidx0")
      lds = UOp.placeholder((32 * stride,), dtypes.int32, 210, addrspace=AddrSpace.LOCAL)
      stage = lds[lane * stride].store(src[lane])
      bar = UOp.barrier(UOp.group(stage))
      return out[lane].store(lds.after(bar)[lane * stride]).sink(arg=KernelInfo(name=f"pmc_{kind}", opts_to_apply=()))
    Compiled.profile_events.clear()
    Tensor.empty(32, dtype=dtypes.int32, device="AMD").custom_kernel(a, fxn=kernel)[0].realize()
  else: raise ValueError(f"unknown control {kind!r}")
  Device["AMD"].synchronize()
  events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"]
  return {"status": "live" if events else "blocked", "event_count": len(events),
          "counters": _decode_event(events[-1]) if events else {}}


def _child_mmq(writeback_mode: str, seed: int) -> dict[str, Any]:
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Compiled, Device
  from extra.qk.mmq_bounded_harness import ACTIVATION_LAYOUT_MMQ_DS4, _finite_q4k_bytes, _q8_activation_inputs
  from extra.qk.mmq_q4k_q8_atom import _as_u32_words, _ds4_tensors, _q4k_q8_1_bounded_ds4_coop_tile_kernel
  q4 = _finite_q4k_bytes(16, 256, seed)
  activation = _q8_activation_inputs(16, 256, seed + 1, ACTIVATION_LAYOUT_MMQ_DS4)
  assert activation.ds4_activation is not None
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device="AMD").realize()
  values, scales, sums = _ds4_tensors(activation.ds4_activation, "AMD")
  Compiled.profile_events.clear()
  out = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(
    words, values, scales, sums,
    fxn=_q4k_q8_1_bounded_ds4_coop_tile_kernel(16, 16, 256, "ffn_gate_up", writeback_mode))[0].realize()
  Device["AMD"].synchronize()
  events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"]
  return {"status": "live" if events else "blocked", "event_count": len(events),
          "kernel": "q4k_q8_1_mmq_ds4_coop_tile", "writeback_mode": writeback_mode,
          "counters": _decode_event(events[-1]) if events else {}, "output_device": out.device}


def _extract_json(stdout: str) -> dict[str, Any]:
  marker = "MMQ_PMC_JSON="
  line = next((line for line in reversed(stdout.splitlines()) if line.startswith(marker)), None)
  if line is None: raise ValueError("PMC child produced no result marker")
  return json.loads(line[len(marker):])


def _run_control(counters: tuple[str, ...], kind: str, size: int, timeout: int) -> dict[str, Any]:
  env = dict(os.environ, PROFILE="1", PMC="1", PMC_COUNTERS=",".join(counters), VIZ="0")
  root = str(Path(__file__).resolve().parents[2])
  env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  argv = [sys.executable, str(Path(__file__).resolve()), "--child", kind, str(size)]
  try:
    proc = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
      return {"status": "blocked", "returncode": proc.returncode, "stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]}
    result = _extract_json(proc.stdout)
    result["stderr"] = proc.stderr[-4000:]
    return result
  except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as exc:
    return {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}


def classify_liveness(negative: list[int], positive: list[int]) -> tuple[str, str]:
  if not negative or not positive: return "blocked", "control samples missing"
  if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in negative + positive):
    return "blocked", "counter samples are invalid"
  if max(negative + positive) == 0: return "zero_suspect", "positive and negative controls are all zero"
  if len(set(positive)) == 1 and len(set(negative)) == 1 and positive[0] <= negative[0]:
    return "zero_suspect", "positive control is non-directional and degenerate"
  neg_med, pos_med = sorted(negative)[len(negative)//2], sorted(positive)[len(positive)//2]
  if pos_med <= neg_med: return "zero_suspect", "positive control did not exceed negative control"
  return "live", "repeated positive control exceeds negative control"


def run_pmc_liveness_suite(counter_groups: Iterable[Iterable[str]] = DEFAULT_GROUPS, *, repetitions: int = 3,
                           negative_size: int = 1024, positive_size: int = 1 << 20, timeout: int = 45,
                           system_snapshot_id: str | None = None) -> dict[str, Any]:
  if repetitions < 2: raise ValueError("repetitions must be >= 2")
  passes = []
  for pass_index, raw_group in enumerate(counter_groups):
    group = tuple(dict.fromkeys(raw_group))
    controls = {"negative": [], "positive": []}
    is_lds = any("LDS" in name for name in group)
    is_memory = any(name.startswith(("GL2C_", "TA_")) for name in group)
    negative_kind, positive_kind = (("lds_free", "lds_conflict") if is_lds else
                                    (("memory", "memory") if is_memory else ("compute", "compute")))
    for _ in range(repetitions):
      controls["negative"].append(_run_control(group, negative_kind, 32 if is_lds else negative_size, timeout))
      controls["positive"].append(_run_control(group, positive_kind, 32 if is_lds else positive_size, timeout))
    rows = []
    for counter in group:
      neg = [r.get("counters", {}).get(counter) for r in controls["negative"] if r.get("status") == "live"]
      pos = [r.get("counters", {}).get(counter) for r in controls["positive"] if r.get("status") == "live"]
      neg_i, pos_i = [v for v in neg if isinstance(v, int)], [v for v in pos if isinstance(v, int)]
      status, reason = classify_liveness(neg_i, pos_i)
      rows.append({"name": counter, "status": status, "reason": reason,
                   "negative_samples": neg_i, "positive_samples": pos_i})
    passes.append({"pass": pass_index, "counters": list(group), "controls": controls, "metrics": rows})
  return {"schema": SCHEMA, "kind": "liveness_suite", "collector": "tinygrad_kfd_native_pmc",
          "system_snapshot_id": system_snapshot_id, "repetitions": repetitions,
          "controls": {"negative_size": negative_size, "positive_size": positive_size,
                       "selection": "SQ=compute scaling; GL2/TA=memory scaling; LDS=conflict-free vs conflict"}, "passes": passes,
          "notes": ["each control runs in a fresh eager process", "zero_suspect is not a measured zero"]}


def collect_kernel_pmc(candidate: Mapping[str, Any], counters: Iterable[str], repetitions: int, *,
                       command: list[str], system_snapshot_id: str, binary_sha256: str,
                       timeout: int = 120) -> dict[str, Any]:
  """Run an identity-bound eager command. The command must print MMQ_PMC_JSON output."""
  if repetitions < 1: raise ValueError("repetitions must be positive")
  candidate_id = candidate.get("candidate_id")
  if not isinstance(candidate_id, str) or not candidate_id: raise ValueError("candidate_id is required")
  if len(binary_sha256) != 64 or any(c not in "0123456789abcdef" for c in binary_sha256):
    raise ValueError("binary_sha256 must be lowercase SHA-256")
  group, samples = tuple(dict.fromkeys(counters)), []
  env = dict(os.environ, PROFILE="1", PMC="1", PMC_COUNTERS=",".join(group), VIZ="0")
  root = str(Path(__file__).resolve().parents[2])
  env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  for _ in range(repetitions):
    try:
      proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
      result = _extract_json(proc.stdout) if proc.returncode == 0 else {"status": "blocked", "returncode": proc.returncode}
      result["stderr"] = proc.stderr[-4000:]
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
      result = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    samples.append(result)
  return {"schema": SCHEMA, "kind": "candidate_pmc", "collector": "tinygrad_kfd_native_pmc",
          "candidate_id": candidate_id, "backend": candidate.get("backend"), "shape": candidate.get("shape"),
          "system_snapshot_id": system_snapshot_id, "binary_sha256": binary_sha256,
          "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
          "counters": list(group), "repetitions": repetitions, "samples": samples}


def collect_mmq_pmc(candidate: Mapping[str, Any], counters: Iterable[str], repetitions: int, *,
                    system_snapshot_id: str, binary_sha256: str, seed: int = 0, timeout: int = 60) -> dict[str, Any]:
  mode = candidate.get("knobs", {}).get("writeback_mode") if isinstance(candidate.get("knobs"), Mapping) else None
  if mode not in ("gated_matrix_v0", "direct_owner_v0"): raise ValueError("candidate writeback_mode is invalid")
  command = [sys.executable, str(Path(__file__).resolve()), "--mmq-child", mode, str(seed)]
  return collect_kernel_pmc(candidate, counters, repetitions, command=command,
                            system_snapshot_id=system_snapshot_id, binary_sha256=binary_sha256, timeout=timeout)


def probe_rocprof_fallback(command: list[str], counters: Iterable[str], *,
                           rocprof: str = "/opt/rocm-7.2.4/bin/rocprofv3", timeout: int = 120) -> dict[str, Any]:
  """Attempt ROCProfiler collection and report absence of output without treating it as zero."""
  group = tuple(dict.fromkeys(counters))
  with tempfile.TemporaryDirectory(prefix="mmq-rocprof-") as directory:
    argv = [rocprof, "--pmc", *group, "-d", directory, "-f", "json", "--", *command]
    try:
      proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
      files = sorted(str(path.relative_to(directory)) for path in Path(directory).rglob("*") if path.is_file())
      status = "advertised" if proc.returncode == 0 and files else "blocked"
      reason = ("rocprofiler emitted output requiring metric parsing" if files else
                "rocprofiler emitted no counter artifact; direct KFD/PM4 dispatch is not intercepted")
      return {"schema": SCHEMA, "kind": "rocprof_fallback_probe", "collector": "rocprofv3",
              "status": status, "reason": reason, "counters": list(group), "command": command,
              "returncode": proc.returncode, "output_files": files, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    except FileNotFoundError as exc:
      return {"schema": SCHEMA, "kind": "rocprof_fallback_probe", "collector": "rocprofv3",
              "status": "unsupported", "reason": str(exc), "counters": list(group), "command": command}
    except (OSError, subprocess.TimeoutExpired) as exc:
      return {"schema": SCHEMA, "kind": "rocprof_fallback_probe", "collector": "rocprofv3",
              "status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "counters": list(group), "command": command}


def validate_pmc_result(artifact: Mapping[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  for pidx, p in enumerate(artifact.get("passes", [])):
    for midx, metric in enumerate(p.get("metrics", [])):
      if metric.get("status") not in STATUSES: raise ValueError(f"passes[{pidx}].metrics[{midx}].status is invalid")
      for field in ("negative_samples", "positive_samples"):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in metric.get(field, [])):
          raise ValueError(f"passes[{pidx}].metrics[{midx}].{field} is invalid")


def _main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--child", nargs=2, metavar=("KIND", "SIZE"))
  ap.add_argument("--mmq-child", nargs=2, metavar=("WRITEBACK_MODE", "SEED"))
  ap.add_argument("--liveness", action="store_true")
  args = ap.parse_args()
  if args.child:
    print("MMQ_PMC_JSON=" + json.dumps(_child_control(args.child[0], int(args.child[1])), sort_keys=True))
  elif args.mmq_child:
    print("MMQ_PMC_JSON=" + json.dumps(_child_mmq(args.mmq_child[0], int(args.mmq_child[1])), sort_keys=True))
  elif args.liveness: print(json.dumps(run_pmc_liveness_suite(), indent=2, sort_keys=True))
  else: ap.error("choose --child or --liveness")


if __name__ == "__main__": _main()
