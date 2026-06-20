#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp, KernelInfo
from tinygrad.renderer.amd.schedule import apply_instruction_schedule, metadata_from_instructions, schedule_metadata_dump
from extra.q8_ffn_asm_fullrow_reduce import (
  HIDDEN, Q8_BYTES, build_fullrow_reduce, expected, make_q4_words, make_q8,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3b_compound_emitter_result.json"


def insts_from_program(program: UOp) -> list[Any]:
  return [u.arg for u in program.src[2].src]


def inst_name(inst: Any) -> str:
  return str(inst).split("(", 1)[0]


def count_prefix(insts: list[Any], prefix: str) -> int:
  return sum(1 for inst in insts if inst_name(inst).lower().startswith(prefix.lower()))


def grouped(insts: list[Any]) -> dict[str, int]:
  names = [inst_name(i).lower() for i in insts]
  def c(*prefixes: str) -> int: return sum(1 for n in names if any(n.startswith(p) for p in prefixes))
  return {
    "dot4": c("v_dot4"),
    "fma": c("v_fma", "v_mad"),
    "convert": c("v_cvt"),
    "valu": c("v_"),
    "salu": c("s_"),
    "ds": c("ds_"),
    "barrier": c("s_barrier"),
    "global_load": c("global_load", "flat_load"),
    "global_store": c("global_store", "flat_store"),
    "shuffle": c("ds_bpermute"),
    "branch": c("s_cbranch"),
    "waitcnt": c("s_waitcnt"),
    "s_clause": c("s_clause"),
    "s_delay_alu": c("s_delay_alu"),
  }


def scheduled_fullrow_reduce(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_fullrow_reduce(gate, up, gate_words, up_words, q8)
  before = insts_from_program(base)
  scheduled, _actions = apply_instruction_schedule(before)
  sink = base.src[0]
  # Keep launch/resource semantics identical to DNR-2; only replace the instruction stream.
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr3b_generic_scheduled"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                               UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in scheduled))))


def run_candidate(rows_check: int) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()

  base_program = build_fullrow_reduce(gate.uop, up.uop, gate_words.uop, up_words.uop, q8.uop)
  before = insts_from_program(base_program)
  after, actions = apply_instruction_schedule(before)
  metadata = schedule_metadata_dump(metadata_from_instructions(after))

  result: dict[str, Any] = {
    "before_grouped": grouped(before),
    "after_grouped": grouped(after),
    "action_count": len(actions),
    "actions": [a.to_dict() for a in actions],
    "metadata_summary": metadata["summary"],
    "instruction_count_before": len(before),
    "instruction_count_after": len(after),
  }

  try:
    gate_out, up_out, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=scheduled_fullrow_reduce)[:2]
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


def main() -> int:
  dnr3a = json.loads((ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3a_scheduler_resource_plan_result.json").read_text())
  probe = run_candidate(rows_check=64)
  after = probe["after_grouped"]
  correctness = probe.get("correctness", {})
  correct = probe.get("launch") == "PASS" and correctness.get("gate_max_abs", 1.0) <= 1e-3 and correctness.get("up_max_abs", 1.0) <= 1e-3
  oracle_shaped = (
    after.get("s_clause") == 3 and after.get("s_delay_alu") == 30 and
    after.get("global_load") <= 11 and after.get("branch") >= 5 and after.get("ds") <= 7
  )
  gates = {
    "dnr3a_structural_plan_passed": dnr3a.get("gate_pass") is True,
    "generic_emitter_changed_stream": probe["instruction_count_after"] != probe["instruction_count_before"],
    "generic_emitter_launches": probe.get("launch") == "PASS",
    "generic_emitter_correct": correct,
    "oracle_shape_reached": oracle_shaped,
    "performance_claim": False,
  }
  if not gates["generic_emitter_launches"]:
    verdict = "BLOCKED_DNR3B_GENERIC_EMITTER_DOES_NOT_LAUNCH"
  elif not gates["generic_emitter_correct"]:
    verdict = "BLOCKED_DNR3B_GENERIC_EMITTER_BREAKS_CORRECTNESS"
  elif not gates["oracle_shape_reached"]:
    verdict = "BLOCKED_DNR3B_GENERIC_EMITTER_CORRECT_BUT_NOT_ORACLE_SHAPED"
  else:
    verdict = "PASS_DNR3B_COMPOUND_EMITTER_STRUCTURAL_READY_FOR_TIMING"

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3B_DECODE_COMPOUND_SCHEDULER_RESOURCE_EMITTER",
    "schema": "decode_native_renderer_dnr3b_compound_emitter_probe_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "probe": probe,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C oracle-shaped compound emitter",
      "reason": (
        "The generic schedule-action emitter can be applied to the correct DNR-2 stream, but it is not a "
        "decode-MMVQ oracle-shaped scheduler/resource emitter."
      ),
      "minimum_unblock": [
        "coalesced Q4_K/q8 global-load rewrite to reduce grouped global loads toward 11",
        "decode-specific marker policy targeting s_clause=3 and s_delay_alu=30, not generic marker spam",
        "branch/exec policy derived from lane roles",
        "LDS/reduction policy reducing ds ops toward oracle without breaking output",
        "register/live-range ledger tied to emitted instruction stream",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3a_scheduler_resource_plan_result.json",
      "extra/q8_ffn_asm_fullrow_reduce.py",
    ],
    "next_action": "Build DNR-3C as a decode-specific oracle-shaped emitter; the existing generic scheduler is insufficient.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": gates,
    "before_grouped": probe["before_grouped"],
    "after_grouped": probe["after_grouped"],
    "correctness": correctness,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if probe.get("launch") == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
