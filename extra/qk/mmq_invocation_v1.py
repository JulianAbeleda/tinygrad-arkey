#!/usr/bin/env python3
"""MMQ-shaped, noncandidate host invocation factorial for 16x16x256 orchestration."""
from __future__ import annotations

import argparse, hashlib, json, platform, random, statistics, subprocess, time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear, get_runtime
from tinygrad.uop.ops import KernelInfo, UOp

from extra.qk.q4k_q8_fixture import ACTIVATION_LAYOUT_MMQ_DS4, make_finite_q4k_bytes, make_q8_activation_inputs
from extra.qk.mmq_compile_evidence import analyze_final_isa, disassemble_amdgpu
from extra.qk.mmq_q4k_q8_atom import _as_u32_words, _staged_ds4_lifecycle_for_spec
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec, Q8_1_MMQ_DS4_LAYOUT

SCHEMA = "tinygrad.mmq_invocation_v1.generated_host_factorial.v1"
POINTS = (0, 64, 128, 256)
PHASES = ("numpy_views", "q4_construct_realize_transfer", "ds4_construct_realize_transfer", "output_allocation",
          "uop_construction", "schedule_creation", "warmed_compile_cache_lookup", "enqueue_sync_host_residual",
          "lifecycle_construction", "output_readback")
SHAPE = {"M": 16, "N": 16, "K": 256}


def generated_id(false_sites:int) -> str:
  if false_sites not in POINTS: raise ValueError(f"false_sites must be one of {POINTS}")
  return f"generated_noncandidate.mmq_invocation_v1.m16n16k256.false{false_sites}"


def _kernel(false_sites:int) -> Callable[..., UOp]:
  identity = generated_id(false_sites).replace(".", "_")
  def kernel(out:UOp, words:UOp, values:UOp, scales:UOp, sums:UOp) -> UOp:
    row, batch, lane = UOp.special(16, "gidx0"), UOp.special(16, "gidx1"), UOp.special(32, "lidx0")
    value = words[row].cast(dtypes.float32) + values[batch * 256 + lane].cast(dtypes.float32)
    value = value + scales[batch * 8].cast(dtypes.float32) + sums[batch * 8].cast(dtypes.float32)
    stores = [out[batch, row].store(value, gate=lane.eq(0))]
    for site in range(false_sites):
      # gidx0 is in [0,15], so these sites are static but cannot execute.
      stores.append(out[site % 16, (site // 16) % 16].store(value, gate=row.eq(16 + site) & lane.eq(0)))
    return UOp.group(*stores).sink(arg=KernelInfo(name=identity, opts_to_apply=()))
  return kernel


def _sink(false_sites:int) -> UOp:
  return _kernel(false_sites)(UOp.placeholder((16, 16), dtypes.float32, 0),
    UOp.placeholder((16 * 18,), dtypes.uint32, 1), UOp.placeholder((16 * 256,), dtypes.int8, 2),
    UOp.placeholder((16 * 8,), dtypes.float32, 3), UOp.placeholder((16 * 8,), dtypes.float32, 4))


def _clock(fxn:Callable[[], Any]) -> tuple[Any, int]:
  start = time.perf_counter_ns(); value = fxn(); return value, time.perf_counter_ns() - start


def _empty_clock() -> int:
  start = time.perf_counter_ns(); return time.perf_counter_ns() - start


def _summarize(samples:list[int], overhead:int) -> dict[str, Any]:
  median = float(statistics.median(samples))
  return {"samples_ns": samples, "median_ns": median, "min_ns": min(samples), "max_ns": max(samples),
          "overhead_corrected_median_ns": max(0.0, median - overhead)}


def _fit(rows:list[dict[str, Any]], phase:str) -> dict[str, Any]:
  x = np.asarray([[1, row["false_sites"]] for row in rows], dtype=np.float64)
  y = np.asarray([row["phases"][phase]["overhead_corrected_median_ns"] for row in rows])
  coef = np.linalg.lstsq(x, y, rcond=None)[0]; predicted = x @ coef
  denom = float(np.sum((y - y.mean()) ** 2))
  return {"intercept_ns": float(coef[0]), "per_false_site_ns": float(coef[1]),
          "r2": 1.0 if denom == 0 else 1.0 - float(np.sum((y - predicted) ** 2)) / denom}


def _identity(system_snapshot_id:str | None) -> dict[str, Any]:
  try: revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError): revision = None
  facts = {"hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version(), "tinygrad_git_revision": revision,
           "device": "AMD", "device_class": type(Device["AMD"]).__name__, "renderer_class": type(Device["AMD"].renderer).__name__}
  digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
  return {"system_snapshot_id": system_snapshot_id or f"sha256:{digest}",
          "system_snapshot_source": "supplied" if system_snapshot_id else "host_facts", "host_facts": facts}


def _prepare_point(false_sites:int, q4:np.ndarray, ds4:Any) -> dict[str, Any]:
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device="AMD").realize()
  values = Tensor(np.ascontiguousarray(ds4.values.reshape(-1)), dtype=dtypes.int8, device="AMD").realize()
  scales = Tensor(np.ascontiguousarray(ds4.scales.reshape(-1)), dtype=dtypes.float32, device="AMD").realize()
  sums = Tensor(np.ascontiguousarray(ds4.sums.reshape(-1)), dtype=dtypes.float32, device="AMD").realize()
  out = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD")
  lazy = out.custom_kernel(words, values, scales, sums, fxn=_kernel(false_sites))[0]
  lazy.realize()
  program = to_program(_sink(false_sites), Device["AMD"].renderer)
  runtime = get_runtime("AMD", program)
  buffers = {0: lazy.uop.buffer._buf, 1: words.uop.buffer._buf, 2: values.uop.buffer._buf,
             3: scales.uop.buffer._buf, 4: sums.uop.buffer._buf}
  args = tuple(buffers[index] for index in program.arg.globals)
  binary = program.src[4].arg; disassembly, tool = disassemble_amdgpu(binary)
  isa = analyze_final_isa(disassembly)
  source = program.src[3].arg; sink = _sink(false_sites)
  return {"words": words, "values": values, "scales": scales, "sums": sums, "out": lazy, "program": program,
          "runtime": runtime, "args": args, "static": {"generated_id": generated_id(false_sites), "candidate_id": None,
            "sink_uops": len(sink.toposort()), "source_bytes": len(source.encode()), "source_lines": len(source.splitlines()),
            "rendered_statements": source.count(";"), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary).hexdigest(), "binary_bytes": len(binary),
            "isa_sha256": hashlib.sha256(disassembly.encode()).hexdigest(), "isa_instruction_count": len(isa["instructions"]),
            "disassembly_tool": tool, "program_key": program.key.hex()}}


def run_invocation_v1(*, rounds:int=30, warmups:int=5, seed:int=20260711,
                      system_snapshot_id:str | None=None) -> dict[str, Any]:
  if rounds < 30: raise ValueError("invocation-v1 requires rounds >= 30")
  if warmups < 1: raise ValueError("invocation-v1 requires warmups >= 1")
  q4 = make_finite_q4k_bytes(16, 256, seed)
  activation = make_q8_activation_inputs(16, 256, seed + 1, ACTIVATION_LAYOUT_MMQ_DS4)
  ds4 = activation.ds4_activation
  if ds4 is None: raise RuntimeError("DS4 construction failed")
  state = {sites: _prepare_point(sites, q4, ds4) for sites in POINTS}
  overhead_samples = [_empty_clock() for _ in range(2000)]; overhead = int(statistics.median(overhead_samples))
  samples = {sites: {phase: [] for phase in PHASES} | {"device_kernel_time": []} for sites in POINTS}

  # Warm the exact isolated operations before collecting cache-hit samples.
  for sites, item in state.items():
    for _ in range(warmups):
      lazy = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(
        item["words"], item["values"], item["scales"], item["sums"], fxn=_kernel(sites))[0]
      linear, _ = lazy.linear_with_vars(); compile_linear(linear)
      item["runtime"](*item["args"], global_size=item["program"].arg.global_size,
                      local_size=item["program"].arg.local_size, wait=True)
      item["out"].numpy()

  order = [sites for sites in POINTS for _ in range(rounds)]; random.Random(seed).shuffle(order)
  for sites in order:
    item = state[sites]
    _, elapsed = _clock(lambda: (np.asarray(q4, dtype=np.uint8), np.asarray(ds4.values, dtype=np.int8),
                                  np.asarray(ds4.scales, dtype=np.float32), np.asarray(ds4.sums, dtype=np.float32)))
    samples[sites]["numpy_views"].append(elapsed)
    _, elapsed = _clock(lambda: Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device="AMD").realize())
    samples[sites]["q4_construct_realize_transfer"].append(elapsed)
    def make_ds4() -> tuple[Tensor, Tensor, Tensor]:
      return (Tensor(np.ascontiguousarray(ds4.values.reshape(-1)), dtype=dtypes.int8, device="AMD").realize(),
              Tensor(np.ascontiguousarray(ds4.scales.reshape(-1)), dtype=dtypes.float32, device="AMD").realize(),
              Tensor(np.ascontiguousarray(ds4.sums.reshape(-1)), dtype=dtypes.float32, device="AMD").realize())
    _, elapsed = _clock(make_ds4); samples[sites]["ds4_construct_realize_transfer"].append(elapsed)
    def allocate_output() -> Tensor:
      output = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD"); output.uop.buffer.allocate(); return output
    _, elapsed = _clock(allocate_output); samples[sites]["output_allocation"].append(elapsed)
    lazy, elapsed = _clock(lambda: Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(
      item["words"], item["values"], item["scales"], item["sums"], fxn=_kernel(sites))[0])
    samples[sites]["uop_construction"].append(elapsed)
    scheduled, elapsed = _clock(lambda: lazy.linear_with_vars()); samples[sites]["schedule_creation"].append(elapsed)
    linear, _ = scheduled
    _, elapsed = _clock(lambda: compile_linear(linear)); samples[sites]["warmed_compile_cache_lookup"].append(elapsed)
    device_seconds:float = 0.0
    def launch() -> None:
      nonlocal device_seconds
      device_seconds = float(item["runtime"](*item["args"], global_size=item["program"].arg.global_size,
                                            local_size=item["program"].arg.local_size, wait=True))
    _, elapsed = _clock(launch)
    device_ns = int(device_seconds * 1e9); samples[sites]["device_kernel_time"].append(device_ns)
    samples[sites]["enqueue_sync_host_residual"].append(max(0, elapsed - device_ns))
    _, elapsed = _clock(lambda: _staged_ds4_lifecycle_for_spec(Q4KQ81MMQTileSpec(
      role="ffn_gate_up", m=16, n=16, k=256, m_tile=16, n_tile=16, activation_layout=Q8_1_MMQ_DS4_LAYOUT)))
    samples[sites]["lifecycle_construction"].append(elapsed)
    _, elapsed = _clock(lambda: item["out"].numpy().astype(np.float32)); samples[sites]["output_readback"].append(elapsed)

  rows = []
  for sites in POINTS:
    rows.append({"false_sites": sites, "identity": state[sites]["static"],
                 "phases": {phase: _summarize(samples[sites][phase], overhead) for phase in PHASES},
                 "device_kernel_time": _summarize(samples[sites]["device_kernel_time"], 0)})
  return {"schema": SCHEMA, "provenance_class": "generated_host_microbenchmark", **_identity(system_snapshot_id),
          "shape": SHAPE, "points": list(POINTS), "candidate_ids": [],
          "buffer_policy": {"orchestration_shape": "16x16x256", "runtime_inputs_outputs": "preallocated resident AMD buffers",
                            "transfer_phases": "isolated temporary tensors excluded from runtime phase"},
          "protocol": {"rounds": rounds, "warmups": warmups, "seed": seed, "randomized_interleaved_order": order,
                       "clock": "time.perf_counter_ns", "device_time_policy": "reported separately and excluded from every host fit",
                       "enqueue_sync_host_residual": "max(0, host wait wall time - runtime returned device duration)"},
          "instrumentation_overhead": _summarize(overhead_samples, 0), "rows": rows,
          "host_fits": {phase: _fit(rows, phase) for phase in PHASES}, "production_dispatch_changed": False}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output", type=Path)
  parser.add_argument("--rounds", type=int, default=30); parser.add_argument("--warmups", type=int, default=5)
  parser.add_argument("--seed", type=int, default=20260711); parser.add_argument("--system-snapshot-id")
  args = parser.parse_args(); result = run_invocation_v1(rounds=args.rounds, warmups=args.warmups, seed=args.seed,
                                                         system_snapshot_id=args.system_snapshot_id)
  args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
