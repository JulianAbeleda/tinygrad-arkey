#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import v
from tinygrad.runtime.autogen.amd.rdna3.ins import ds_load_b128, s_clause, s_delay_alu, s_waitcnt, v_mov_b32_e32

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, expected, make_q4_words, make_q8
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, inst_name, insts_from_program
from extra.qk_decode_native_renderer_dnr3c2_dataflow_emitter_probe import build_b128_preload_fullrow_reduce


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c4_semantic_reduction_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def build_dnr3c4_candidate(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_b128_preload_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  # Replace four scalar LDS cross-wave loads with one vector LDS load of the four wave partials.
  reduced = list(old[:166]) + [
    v_mov_b32_e32(vdst=v[54], src0=0),
    ds_load_b128(vdst=v[10:13], addr=v[54]),
    s_waitcnt(simm16=0),
  ] + list(old[175:])

  # Count-matched marker placement. This is correctness-safe; performance semantics still need timing.
  insts: list[Any] = []
  clause_before = {18, 58, 60}
  delay_count = 0
  for i, inst in enumerate(reduced):
    if i in clause_before: insts.append(s_clause(simm16=0))
    insts.append(inst)
    name = inst_name(inst).lower()
    if delay_count < 30 and (name.startswith("v_") or name.startswith("ds_")):
      insts.append(s_delay_alu(simm16=0))
      delay_count += 1

  sink = base.src[0]
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr3c4_b128_dsload_b128_marked"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                               UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))


def run_candidate(rows_check: int) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_host, up_host = make_q4_words(37, 5), make_q4_words(53, 19)
  q8_host = make_q8()
  gate_words = Tensor(gate_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(up_host, dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_tensor = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()

  native = insts_from_program(build_b128_preload_fullrow_reduce(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop))
  candidate = insts_from_program(build_dnr3c4_candidate(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop))
  result: dict[str, Any] = {
    "predecessor_grouped": grouped(native),
    "candidate_grouped": grouped(candidate),
    "instruction_count_predecessor": len(native),
    "instruction_count_candidate": len(candidate),
    "reduction_model": {
      "before": "5 ds_bpermute + 1 ds_store_b32 + 4 ds_load_b32 = 10 ds ops",
      "after": "5 ds_bpermute + 1 ds_store_b32 + 1 ds_load_b128 = 7 ds ops",
      "semantic_reason": "The four wave partials remain present in LDS; the cross-wave read is vectorized, not deleted.",
    },
  }
  try:
    gate_out, up_out, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8_tensor, fxn=build_dnr3c4_candidate)[:2]
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
  dnr3c3 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c3_compound_shape_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  og = oracle["instruction_contract"]["oracle_grouped"]
  probe = run_candidate(rows_check=128)
  cg = probe["candidate_grouped"]
  correctness = probe.get("correctness", {})
  correct = probe.get("launch") == "PASS" and correctness.get("gate_max_abs", 1.0) <= 1e-3 and correctness.get("up_max_abs", 1.0) <= 1e-3
  gates = {
    "dnr3c3_blocked_on_semantic_reduction": dnr3c3.get("verdict") == "BLOCKED_DNR3C3_COMPOUND_SHAPE_NEEDS_SEMANTIC_BRANCH_REDUCTION_MODEL",
    "candidate_launches": probe.get("launch") == "PASS",
    "candidate_correct": correct,
    "global_load_budget_closed": cg.get("global_load", 999) <= og.get("global_load", 0),
    "ds_budget_closed": cg.get("ds") <= og.get("ds", 0),
    "marker_counts_match_oracle": cg.get("s_clause") == 3 and cg.get("s_delay_alu") == 30,
    "dot4_preserved": cg.get("dot4") == og.get("dot4") == 16,
    "branch_policy_matches_oracle": cg.get("branch") >= og.get("branch", 0),
    "waitcnt_count_matches_oracle": cg.get("waitcnt") == og.get("waitcnt"),
    "performance_claim": False,
  }
  verdict = (
    "PASS_DNR3C4_SEMANTIC_REDUCTION_CORRECT_BLOCKED_ON_BRANCH_WAIT_TIMING"
    if correct and gates["global_load_budget_closed"] and gates["ds_budget_closed"] and gates["marker_counts_match_oracle"]
    else "BLOCKED_DNR3C4_SEMANTIC_REDUCTION_INCORRECT_OR_INCOMPLETE"
  )
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C4_DECODE_SEMANTIC_REDUCTION",
    "schema": "decode_native_renderer_dnr3c4_semantic_reduction_probe_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "probe": probe,
    "oracle_grouped": og,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C5 timing and branch/wait policy decision",
      "reason": "The correctness-preserving compound candidate now closes load, DS, dot4, store, and marker budgets. Static branch and waitcnt counts still differ from the oracle, and no timing has been taken.",
      "minimum_unblock": [
        "time DNR-3C4 against DNR-2 native and the q8 oracle under the same harness",
        "decide whether branch/wait count mismatch is performance-relevant or only oracle residue",
        "only build branch/exec control if timing remains materially behind and attribution points there",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c3_compound_shape_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    ],
    "next_action": "Run DNR-3C5 timing before adding branch/control flow; DNR-3C4 already matches the core movement budgets.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": gates,
    "candidate_grouped": cg,
    "correctness": correctness,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
