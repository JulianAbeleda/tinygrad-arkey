#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.runtime.autogen.amd.rdna3.ins import s_clause, s_delay_alu

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, expected, make_q4_words, make_q8
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, inst_name, insts_from_program
from extra.qk_decode_native_renderer_dnr3c2_dataflow_emitter_probe import build_b128_preload_fullrow_reduce


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c3_compound_shape_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def build_marker_count_candidate(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_b128_preload_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  insts: list[Any] = []
  clause_before = {18, 58, 60}
  delay_count = 0
  for i, inst in enumerate(old):
    if i in clause_before: insts.append(s_clause(simm16=0))
    insts.append(inst)
    name = inst_name(inst).lower()
    if delay_count < 30 and (name.startswith("v_") or name.startswith("ds_")):
      insts.append(s_delay_alu(simm16=0))
      delay_count += 1
  sink = base.src[0]
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr3c3_marker_count"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                               UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def build_naive_ds7_candidate(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_b128_preload_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  # Keep the five wave-local ds_bpermutes, the LDS store, and only the first cross-wave ds_load.
  # This reaches the oracle static ds count, but intentionally lacks the branch/exec semantics needed
  # to make one loaded wave partial equal the full four-wave row reduction.
  insts = list(old[:168]) + [old[174]] + list(old[178:])
  sink = base.src[0]
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr3c3_naive_ds7"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                               UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def run_candidate(name: str, fxn: Callable[[UOp, UOp, UOp, UOp, UOp], UOp], rows_check: int) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_tensor = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  insts = insts_from_program(fxn(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop))
  result: dict[str, Any] = {"name": name, "grouped": grouped(insts), "instruction_count": len(insts)}
  try:
    gate_out, up_out, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8_tensor, fxn=fxn)[:2]
    run_linear(gate_out.schedule_linear())
    got_gate = gate_out.numpy().astype(np.float32)[:rows_check]
    got_up = up_out.numpy().astype(np.float32)[:rows_check]
    exp_gate = expected(gate_host, q8_host, rows_check)
    exp_up = expected(up_host, q8_host, rows_check)
    gate_abs, up_abs = np.abs(got_gate - exp_gate), np.abs(got_up - exp_up)
    result.update({
      "launch": "PASS",
      "correctness": {
        "rows_check": rows_check,
        "gate_max_abs": float(gate_abs.max()),
        "gate_mean_abs": float(gate_abs.mean()),
        "up_max_abs": float(up_abs.max()),
        "up_mean_abs": float(up_abs.mean()),
      },
    })
  except Exception as e:
    result.update({"launch": "FAIL", "error": repr(e)})
  return result


def correct(result: dict[str, Any]) -> bool:
  c = result.get("correctness", {})
  return result.get("launch") == "PASS" and c.get("gate_max_abs", 1.0) <= 1e-3 and c.get("up_max_abs", 1.0) <= 1e-3


def main() -> int:
  dnr3c2 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c2_dataflow_emitter_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  og = oracle["instruction_contract"]["oracle_grouped"]
  marker = run_candidate("marker_count_candidate", build_marker_count_candidate, rows_check=128)
  naive_ds7 = run_candidate("naive_ds7_candidate", build_naive_ds7_candidate, rows_check=32)
  marker_group, ds7_group = marker["grouped"], naive_ds7["grouped"]
  gates = {
    "dnr3c2_load_shape_passed": dnr3c2.get("gate_pass") is True,
    "marker_candidate_launches": marker.get("launch") == "PASS",
    "marker_candidate_correct": correct(marker),
    "marker_counts_match_oracle": marker_group.get("s_clause") == 3 and marker_group.get("s_delay_alu") == 30,
    "load_budget_still_closed": marker_group.get("global_load", 999) <= og.get("global_load", 0),
    "naive_ds7_reaches_static_ds_budget": ds7_group.get("ds") <= og.get("ds", 0),
    "naive_ds7_correct": correct(naive_ds7),
    "branch_control_policy_present": marker_group.get("branch", 0) >= og.get("branch", 0),
    "performance_claim": False,
  }
  verdict = (
    "BLOCKED_DNR3C3_COMPOUND_SHAPE_NEEDS_SEMANTIC_BRANCH_REDUCTION_MODEL"
    if gates["marker_candidate_correct"] and gates["marker_counts_match_oracle"] and
       gates["naive_ds7_reaches_static_ds_budget"] and not gates["naive_ds7_correct"] and
       not gates["branch_control_policy_present"]
    else "BLOCKED_DNR3C3_COMPOUND_SHAPE_EVIDENCE_INCOMPLETE"
  )
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C3_DECODE_COMPOUND_SHAPE",
    "schema": "decode_native_renderer_dnr3c3_compound_shape_probe_v1",
    "verdict": verdict,
    "gate_pass": False,
    "default_behavior_changed": False,
    "performance_claim": False,
    "marker_candidate": marker,
    "naive_ds7_candidate": naive_ds7,
    "oracle_grouped": og,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C4 semantic branch/exec reduction model",
      "reason": (
        "The marker-count and coalesced-load pieces are launch-correct, but static DS reduction to the oracle budget "
        "breaks row-sum correctness without a real branch/exec lane-role model."
      ),
      "minimum_unblock": [
        "model which lanes/waves own cross-wave partial loading, final accumulation, and global store",
        "emit branch or exec-mask control flow that preserves exactly one full row sum per output",
        "replace the four unconditional cross-wave ds_loads with the oracle-shaped controlled reduction",
        "revalidate synthetic gate/up correctness with load shape and markers still enabled",
        "then time the compound candidate against the q8 oracle",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c2_dataflow_emitter_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    ],
    "next_action": "Build DNR-3C4 as a semantic branch/exec reduction model; do not delete DS ops or add dead branches to match counts.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": gates,
    "marker_grouped": marker_group,
    "marker_correctness": marker.get("correctness", {}),
    "naive_ds7_grouped": ds7_group,
    "naive_ds7_correctness": naive_ds7.get("correctness", {}),
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
