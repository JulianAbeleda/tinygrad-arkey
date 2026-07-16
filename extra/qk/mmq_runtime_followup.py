#!/usr/bin/env python3
"""Compile-once MMQ runtime decomposition with preallocated, correct inputs."""
from __future__ import annotations

import hashlib, json, statistics, time
from pathlib import Path
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.device import Device

from extra.qk.mmq_bounded_harness import run_bounded_harness
from extra.qk.q4k_q8_fixture import ACTIVATION_LAYOUT_MMQ_DS4, make_finite_q4k_bytes, make_q8_activation_inputs
from extra.qk.mmq_compile_evidence import capture_loaded_mmq_program, compile_mmq_program
from extra.qk.mmq_experiment import BACKEND, canonical_candidate
from extra.qk.mmq_q4k_q8_atom import _as_u32_words, _ds4_tensors, _q4k_q8_1_bounded_ds4_coop_tile_kernel

SCHEMA = "tinygrad.mmq_runtime_followup.v3"


def _median(values:list[float]) -> float: return statistics.median(values)


def run_runtime_followup(writeback_mode:str, *, repeats:tuple[int, ...]=(1, 16, 64, 256), warmups:int=5,
                         rounds:int=30, seed:int=0, system_snapshot_id:str, collect_pmc:bool=False) -> dict[str, Any]:
  if not repeats or any(n <= 0 for n in repeats): raise ValueError("repeat counts must be positive")
  if warmups < 1 or rounds < 3: raise ValueError("warmups >= 1 and rounds >= 3 are required")
  spec = canonical_candidate(writeback_mode, seed=seed)
  compile_start = time.perf_counter(); program = compile_mmq_program(spec); compile_ms = (time.perf_counter()-compile_start)*1e3
  setup_start = time.perf_counter()
  q4 = make_finite_q4k_bytes(16, 256, seed)
  activation = make_q8_activation_inputs(16, 256, seed + 1, ACTIVATION_LAYOUT_MMQ_DS4)
  assert activation.ds4_activation is not None
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device="AMD").realize()
  values, scales, sums = _ds4_tensors(activation.ds4_activation, "AMD")
  fxn = _q4k_q8_1_bounded_ds4_coop_tile_kernel(16, 16, 256, "ffn_gate_up", writeback_mode)
  out = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(words, values, scales, sums, fxn=fxn)[0].realize()
  Device["AMD"].synchronize(); setup_ms = (time.perf_counter()-setup_start)*1e3
  evidence = capture_loaded_mmq_program(spec)
  if evidence.program.key != program.key: raise RuntimeError("follow-up program differs from compiled candidate")
  from tinygrad.engine.realize import runtime_cache
  runtime = runtime_cache[(program.key, "AMD")]
  buffers = {0: out.uop.buffer._buf, 1: words.uop.buffer._buf, 2: values.uop.buffer._buf,
             3: scales.uop.buffer._buf, 4: sums.uop.buffer._buf}
  args = tuple(buffers[index] for index in program.arg.globals)
  gs, ls = program.arg.global_size, program.arg.local_size
  for _ in range(warmups): runtime(*args, global_size=gs, local_size=ls, wait=True)
  points = []
  for repeat in repeats:
    gpu_samples, enqueue_sync_samples = [], []
    for _ in range(rounds):
      gpu_samples.append(sum(float(runtime(*args, global_size=gs, local_size=ls, wait=True))*1e3 for _ in range(repeat)))
    for _ in range(rounds):
      start = time.perf_counter()
      for _ in range(repeat): runtime(*args, global_size=gs, local_size=ls, wait=False)
      Device["AMD"].synchronize(); enqueue_sync_samples.append((time.perf_counter()-start)*1e3)
    points.append({"repeats": repeat, "gpu_timestamp_sum_ms": gpu_samples,
                   "gpu_timestamp_per_launch_ms": _median(gpu_samples)/repeat,
                   "enqueue_sync_wall_ms": enqueue_sync_samples,
                   "enqueue_sync_per_launch_ms": _median(enqueue_sync_samples)/repeat})
  readback_samples = []
  for _ in range(rounds):
    start=time.perf_counter(); out.numpy(); readback_samples.append((time.perf_counter()-start)*1e3)
  harness_start=time.perf_counter(); harness=run_bounded_harness(spec.config()); harness_wall_ms=(time.perf_counter()-harness_start)*1e3
  result = {"schema": SCHEMA, "candidate_id": spec.candidate_id, "backend": BACKEND, "writeback_mode": writeback_mode,
    "system_snapshot_id": system_snapshot_id, "binary_sha256": evidence.hashes["binary_sha256"],
    "source_sha256": evidence.hashes["rendered_source_sha256"], "program_key": program.key.hex(),
    "protocol": {"warmups": warmups, "rounds": rounds, "repeats": list(repeats), "source_unrolled": False,
                 "compile_once": True, "preallocated": True},
    "decomposition": {"compile_ms": compile_ms, "setup_materialization_ms": setup_ms, "points": points,
                      "correctness_readback_ms": readback_samples, "correctness_readback_median_ms": _median(readback_samples),
                      "existing_harness_internal_candidate_median_ms": harness["timing"]["median_ms"],
                      "existing_harness_total_wall_ms": harness_wall_ms},
    "correctness": {"status": harness["status"], **harness["correctness"]},
    "harness_scope_finding": "existing cooperative full_runner constructs and realizes input/output tensors and returns out.numpy on every timed call",
    "production_dispatch_changed": False}
  if collect_pmc:
    from extra.qk.mmq_amd_pmc import collect_mmq_pmc
    result["pmc"] = collect_mmq_pmc(spec.to_json(), ("SQ_BUSY_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_WAIT_ANY"),
      3, system_snapshot_id=system_snapshot_id, binary_sha256=evidence.hashes["binary_sha256"], seed=seed)
    samples = result["pmc"]["samples"]
    result["pmc_summary"] = {name: {"status": "live" if any(s.get("counters", {}).get(name, 0) > 0 for s in samples) else "zero_suspect",
                                    "samples": [s.get("counters", {}).get(name) for s in samples]}
                             for name in result["pmc"]["counters"]}
  return result


def write_runtime_followup(result:dict[str, Any], path:Path) -> None:
  path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
