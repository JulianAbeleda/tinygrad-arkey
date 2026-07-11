#!/usr/bin/env python3
"""Generated, binary-bound gfx1100 calibration cases for the MMQ cycle model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Callable

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, UOp

from extra.qk.mmq_compile_evidence import analyze_final_isa, disassemble_amdgpu, parse_amdgpu_metadata

SCHEMA = "tinygrad.mmq_calibration.v1"


@dataclass(frozen=True)
class CalibrationCase:
  case_id: str
  family: str
  workgroups: int
  chain_length: int = 0
  independent_streams: int = 1
  stride: int = 1
  local_threads: int = 32

  def validate(self) -> None:
    if self.family not in ("launch", "dependent_valu", "independent_valu", "dependent_valu_int", "dependent_salu",
                           "mixed_salu_valu", "lds_barrier", "lds_wait", "store_only", "resource_pressure"):
      raise ValueError(f"unknown family {self.family!r}")
    if self.workgroups <= 0 or self.chain_length < 0 or self.independent_streams <= 0 or self.stride <= 0 or self.local_threads <= 0:
      raise ValueError("invalid calibration dimensions")
    if self.family == "dependent_valu" and self.chain_length <= 0: raise ValueError("dependent chains require chain_length")
    if self.family == "independent_valu" and (self.chain_length <= 0 or self.independent_streams < 2):
      raise ValueError("independent chains require length and at least two streams")
    if self.family == "resource_pressure" and self.independent_streams < 2: raise ValueError("resource pressure requires multiple streams")


def launch_case(workgroups:int) -> CalibrationCase:
  return CalibrationCase(f"launch.wg{workgroups}", "launch", workgroups)


def dependent_valu_case(workgroups:int, chain_length:int) -> CalibrationCase:
  return CalibrationCase(f"dependent_valu.wg{workgroups}.n{chain_length}", "dependent_valu", workgroups, chain_length)


def independent_valu_case(workgroups:int, chain_length:int, streams:int=4) -> CalibrationCase:
  return CalibrationCase(f"independent_valu.wg{workgroups}.n{chain_length}.s{streams}", "independent_valu", workgroups, chain_length, streams)


def lds_barrier_case(workgroups:int) -> CalibrationCase:
  return CalibrationCase(f"lds_barrier.wg{workgroups}.t64", "lds_barrier", workgroups, local_threads=64)


def lds_wait_case(workgroups:int) -> CalibrationCase:
  return CalibrationCase(f"lds_wait.wg{workgroups}.t64", "lds_wait", workgroups, local_threads=64)


def store_only_case(workgroups:int) -> CalibrationCase:
  return CalibrationCase(f"store_only.wg{workgroups}", "store_only", workgroups)


def resource_pressure_case(workgroups:int, streams:int) -> CalibrationCase:
  return CalibrationCase(f"resource_pressure.wg{workgroups}.s{streams}", "resource_pressure", workgroups, 8, streams)


def global_load_case(workgroups:int, stride:int) -> CalibrationCase:
  return CalibrationCase(f"global_load.wg{workgroups}.stride{stride}", "launch", workgroups, stride=stride)


def issue_case(family:str, workgroups:int=96, chain_length:int=64) -> CalibrationCase:
  if family not in ("dependent_valu_int", "dependent_salu", "mixed_salu_valu"): raise ValueError("unknown issue family")
  return CalibrationCase(f"{family}.wg{workgroups}.n{chain_length}", family, workgroups, chain_length)


def _kernel(case:CalibrationCase) -> Callable[..., UOp]:
  case.validate()
  def kernel(out:UOp, inp:UOp) -> UOp:
    gid, lane = UOp.special(case.workgroups, "gidx0"), UOp.special(case.local_threads, "lidx0")
    idx = gid * case.local_threads + lane
    if case.family == "launch": value = inp[idx * case.stride]
    elif case.family == "store_only": value = gid.cast(dtypes.float32)
    elif case.family in ("lds_barrier", "lds_wait"):
      local = UOp.placeholder((case.local_threads,), dtypes.float32, 100, addrspace=AddrSpace.LOCAL)
      stage = local[lane].store(inp[idx])
      order = UOp.barrier(UOp.group(stage)) if case.family == "lds_barrier" else UOp.group(stage)
      value = local.after(order)[lane]
    elif case.family == "dependent_valu":
      value = inp[idx]
      for step in range(case.chain_length): value = value * UOp.const(dtypes.float32, 1.000001) + UOp.const(dtypes.float32, step * 1e-7)
    elif case.family == "dependent_valu_int":
      ivalue = inp[idx].cast(dtypes.int32)
      for step in range(case.chain_length): ivalue = (ivalue * UOp.const(dtypes.int32, 3 + step % 4)) ^ UOp.const(dtypes.int32, step + 1)
      value = ivalue.cast(dtypes.float32)
    elif case.family == "dependent_salu":
      svalue = gid
      for step in range(case.chain_length): svalue = (svalue * UOp.const(dtypes.int32, 3 + step % 4)) ^ UOp.const(dtypes.int32, step + 1)
      value = svalue.cast(dtypes.float32)
    elif case.family == "mixed_salu_valu":
      svalue, value = gid, inp[idx]
      for step in range(case.chain_length):
        svalue = (svalue * UOp.const(dtypes.int32, 3 + step % 4)) ^ UOp.const(dtypes.int32, step + 1)
        value = value * UOp.const(dtypes.float32, 1.000001) + UOp.const(dtypes.float32, step * 1e-7)
      value = value + svalue.cast(dtypes.float32)
    else:
      values = [inp[(idx + stream) % (case.workgroups * case.local_threads)] for stream in range(case.independent_streams)]
      for step in range(case.chain_length):
        values = [value * UOp.const(dtypes.float32, 1.000001 + stream * 1e-7) + UOp.const(dtypes.float32, step * 1e-7)
                  for stream, value in enumerate(values)]
      value = values[0]
      for other in values[1:]: value = value + other
    return out[idx].store(value).sink(arg=KernelInfo(name="mmq_cal_" + case.case_id.replace(".", "_"), opts_to_apply=()))
  return kernel


def _sink(case:CalibrationCase) -> UOp:
  size = case.workgroups * case.local_threads
  return _kernel(case)(UOp.placeholder((size,), dtypes.float32, 0), UOp.placeholder((size * case.stride,), dtypes.float32, 1))


def run_calibration_case(case:CalibrationCase, *, warmups:int=5, rounds:int=30, device:str="AMD",
                         system_snapshot_id:str | None=None, artifact_output:Path | None=None) -> dict[str, Any]:
  case.validate()
  if warmups < 1 or rounds < 3: raise ValueError("calibration requires warmups >= 1 and rounds >= 3")
  size = case.workgroups * case.local_threads
  inp = Tensor.empty(size * case.stride, dtype=dtypes.float32, device=device).realize()
  out = Tensor.empty(size, device=device).custom_kernel(inp, fxn=_kernel(case))[0].realize()
  program = to_program(_sink(case), Device[device].renderer)
  from tinygrad.engine.realize import runtime_cache
  runtime = runtime_cache.get((program.key, device))
  if runtime is None: raise RuntimeError("calibration runtime is not loaded")
  binary = getattr(runtime, "lib", None)
  if binary != program.src[4].arg: raise RuntimeError("loaded calibration binary mismatch")
  global_size, local_size = program.arg.global_size, program.arg.local_size
  buffers = {0: out.uop.buffer._buf}
  if 1 in program.arg.globals: buffers[1] = inp.uop.buffer._buf
  args = tuple(buffers[index] for index in program.arg.globals)
  for _ in range(warmups): runtime(*args, global_size=global_size, local_size=local_size, wait=True)
  samples_ms = [float(runtime(*args, global_size=global_size, local_size=local_size, wait=True)) * 1e3 for _ in range(rounds)]
  metadata = parse_amdgpu_metadata(binary)
  disassembly, tool = disassemble_amdgpu(binary)
  isa = analyze_final_isa(disassembly, wavefront_size=metadata["wavefront_size"])
  source = program.src[3].arg
  hashes = {"sink_sha256": hashlib.sha256(repr(_sink(case)).encode()).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "isa_sha256": hashlib.sha256(disassembly.encode()).hexdigest()}
  result = {"schema": SCHEMA, "provenance_class": "generated_microbenchmark", "case": case.__dict__,
          "target": metadata["target"], "function_name": program.arg.function_name,
          "system_snapshot_id": system_snapshot_id, "system_binding_status": "bound" if system_snapshot_id else "unknown",
          "program_key": program.key.hex(), "hashes": hashes, "binary_bytes": len(binary), "resources": metadata,
          "isa": isa,
          "launch": {"global_size": list(global_size), "local_size": list(local_size)},
          "protocol": {"warmups": warmups, "rounds": rounds, "clock": {"status": "unknown"}},
          "memory_contract": {"logical_load_bytes_per_lane": 4, "stride_elements": case.stride,
                              "physical_transactions": {"status": "unknown", "join": "GL2/MC_WRREQ calibration proxy"}},
          "samples_ms": samples_ms, "median_ms": statistics.median(samples_ms),
          "min_ms": min(samples_ms), "max_ms": max(samples_ms), "disassembly_tool": tool,
          "production_dispatch_changed": False}
  if artifact_output is not None:
    artifact_output = Path(artifact_output); artifact_output.mkdir(parents=True, exist_ok=True)
    artifacts = {f"{case.case_id}.source.hip": source.encode(), f"{case.case_id}.hsaco": binary,
                 f"{case.case_id}.isa.txt": disassembly.encode()}
    result["artifacts"] = {}
    for name, data in artifacts.items():
      (artifact_output / name).write_bytes(data)
      result["artifacts"][name] = hashlib.sha256(data).hexdigest()
  return result


def default_calibration_matrix() -> tuple[CalibrationCase, ...]:
  return tuple([launch_case(wg) for wg in (1, 32, 64, 96, 128, 192)] +
               [dependent_valu_case(96, n) for n in (16, 64, 256)] +
               [independent_valu_case(96, n, 4) for n in (16, 64, 256)] +
               [lds_barrier_case(wg) for wg in (1, 96)] +
               [lds_wait_case(96), store_only_case(96)] +
               [resource_pressure_case(96, streams) for streams in (4, 8, 16, 32)] +
               [global_load_case(96, stride) for stride in (1, 2, 4, 8, 16, 32)] +
               [issue_case(family) for family in ("dependent_valu_int", "dependent_salu", "mixed_salu_valu")])


def run_calibration_matrix(output:Path, *, warmups:int=5, rounds:int=30, system_snapshot_id:str | None=None) -> dict[str, Any]:
  output = Path(output)
  if output.exists(): raise FileExistsError(output)
  output.mkdir(parents=True)
  results = []
  for case in default_calibration_matrix():
    result = run_calibration_case(case, warmups=warmups, rounds=rounds, system_snapshot_id=system_snapshot_id,
                                  artifact_output=output)
    (output / f"{case.case_id}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    results.append({"case_id": case.case_id, "binary_sha256": result["hashes"]["binary_sha256"], "median_ms": result["median_ms"]})
  ids = [row["case_id"] for row in results]
  coverage = {
    "launch_latency": [x for x in ids if x.startswith("launch.")],
    "grid_tail_cu_boundaries": [x for x in ids if x.startswith("launch.wg")],
    "valu_float_latency_issue": [x for x in ids if x.startswith(("dependent_valu.", "independent_valu."))],
    "valu_int_latency": [x for x in ids if x.startswith("dependent_valu_int.")],
    "salu_latency_issue": [x for x in ids if x.startswith("dependent_salu.")],
    "salu_valu_compatibility": [x for x in ids if x.startswith("mixed_salu_valu.")],
    "vmem_load_stride_transactions": [x for x in ids if x.startswith("global_load.")],
    "vmem_store": [x for x in ids if x.startswith("store_only.")],
    "lds_issue_latency": [x for x in ids if x.startswith(("lds_barrier.", "lds_wait."))],
    "barrier_wait_differential": [x for x in ids if x.startswith(("lds_barrier.", "lds_wait."))],
    "resource_pressure_occupancy": [x for x in ids if x.startswith("resource_pressure.")],
  }
  manifest = {"schema": SCHEMA, "provenance_class": "generated_microbenchmark", "system_snapshot_id": system_snapshot_id,
              "protocol": {"warmups": warmups, "rounds": rounds}, "coverage": coverage, "results": results,
              "production_dispatch_changed": False}
  (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  return manifest
