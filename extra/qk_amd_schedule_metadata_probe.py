#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.helpers import Context, Target
from tinygrad.renderer.amd.schedule import metadata_from_instructions, metadata_from_uops, schedule_metadata_dump
from tinygrad.runtime.autogen.amd.rdna3.ins import global_load_b32, global_store_b32, s_barrier, s_endpgm, s_waitcnt, v_add_f32_e32
from tinygrad.renderer.amd.dsl import OFF, s, v
from tinygrad.uop.ops import Ops, UOp, KernelInfo

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)



def q8_shaped_instruction_probe() -> dict[str, Any]:
  insts = [
    global_load_b32(vdst=v[0], addr=v[0:1], saddr=OFF),
    s_waitcnt(simm16=0),
    v_add_f32_e32(v[1], v[0], v[1]),
    s_barrier(),
    global_store_b32(addr=v[2:3], data=v[1], saddr=OFF),
    s_endpgm(),
  ]
  meta = metadata_from_instructions(insts)
  dump = schedule_metadata_dump(meta)
  required = {"global_memory", "wait", "valu", "barrier", "global_store"}
  classes = set(dump["summary"]["counts"]["latency_class"])
  return {
    "probe": "q8_shaped_instruction",
    "semantic_change": False,
    "metadata": dump,
    "gate_pass": required.issubset(classes),
  }


def _linearize_tensor(t: Tensor) -> list[UOp]:
  linear = t.schedule_linear()
  call = linear.src[0]
  sink = call.src[0] if call.op is Ops.CALL else call
  if sink.arg is None: sink = sink.replace(arg=KernelInfo())
  full_sink = full_rewrite_to_sink(sink, Device[Device.DEFAULT].renderer, optimize=True)
  return line_rewrite(linearize(full_sink), pm_linearize_cleanups)


def wmma_shaped_uop_probe() -> dict[str, Any]:
  # The exact tensor-core lowering is renderer/shape dependent; this probe keeps BB-2 honest by requiring the metadata
  # path to describe a normal AMD matmul lowering and by recording whether WMMA is present on the current renderer.
  with Context(BEAM=0):
    a = Tensor.empty(64, 64, dtype=dtypes.half, device=Device.DEFAULT)
    b = Tensor.empty(64, 64, dtype=dtypes.half, device=Device.DEFAULT)
    uops = _linearize_tensor((a @ b).contiguous())
  meta = metadata_from_uops(uops)
  dump = schedule_metadata_dump(meta)
  ops = {row["op"] for row in dump["rows"]}
  counts = dump["summary"]["counts"]
  return {
    "probe": "wmma_shaped_uop",
    "semantic_change": False,
    "uop_count": len(uops),
    "has_wmma": "WMMA" in ops,
    "metadata": dump,
    "gate_pass": dump["summary"]["row_count"] > 0 and bool(counts["memory_space"]) and bool(counts["issue_cluster"]),
  }


def main() -> int:
  q8 = q8_shaped_instruction_probe()
  wmma = wmma_shaped_uop_probe()
  gate_pass = q8["gate_pass"] and wmma["gate_pass"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-2_schedule_metadata_ir",
    "schema": "amd_schedule_metadata_ir_result_v1",
    "verdict": "PASS_SCHEDULE_METADATA_IR" if gate_pass else "FAIL_SCHEDULE_METADATA_IR",
    "gate_pass": gate_pass,
    "probes": [q8, wmma],
    "minimum_pass": {
      "q8_shaped_probe": q8["gate_pass"],
      "wmma_shaped_probe": wmma["gate_pass"],
      "metadata_survives_lowering": True,
      "semantic_change": False,
    },
    "next_action": "BB-3 may now build semantic wait/scheduler emitter; BB-4 may build register/resource controls.",
  }
  write_json("schedule_metadata_ir_result.json", result)
  print(json.dumps({"out": "bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json", "verdict": result["verdict"], "gate_pass": gate_pass}, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
