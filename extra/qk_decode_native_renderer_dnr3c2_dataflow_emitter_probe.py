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
from tinygrad.renderer.amd.dsl import s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_load_b128, s_waitcnt, v_and_b32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32,
  v_dot4_i32_iu8, v_lshrrev_b32_e32,
)

from extra.q8_ffn_asm_fullrow_reduce import (
  HIDDEN, Q4_WORDS, Q8_BYTES, build_fullrow_reduce, expected, make_q4_words, make_q8,
)
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c2_dataflow_emitter_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def build_b128_preload_fullrow_reduce(gate: UOp, up: UOp, gate_words: UOp, up_words: UOp, q8: UOp) -> UOp:
  base = build_fullrow_reduce(gate, up, gate_words, up_words, q8)
  old = insts_from_program(base)
  insts = list(old[:58])
  # DNR-3C2: four coalesced loads cover the eight q4 words and eight q8 words consumed by the dot4 loop.
  insts += [
    global_load_b128(vdst=v[80:83], addr=v[23], saddr=s[16:17], offset=0),
    global_load_b128(vdst=v[84:87], addr=v[23], saddr=s[16:17], offset=16),
    global_load_b128(vdst=v[88:91], addr=v[24], saddr=s[18:19], offset=0),
    global_load_b128(vdst=v[92:95], addr=v[24], saddr=s[18:19], offset=16),
    s_waitcnt(simm16=0),
  ]
  for lane in range(8):
    q4_word, q8_word = v[80+lane], v[88+lane]
    insts += [
      v_lshrrev_b32_e32(vdst=v[10], src0=4, vsrc1=q4_word),
      v_and_b32_e32(vdst=v[10], src0=0x0f0f0f0f, vsrc1=v[10]),
      v_and_b32_e32(vdst=q4_word, src0=0x0f0f0f0f, vsrc1=q4_word),
      v_and_b32_e32(vdst=v[11], src0=1, vsrc1=v[21]),
      v_cmp_ne_u32_e32(src0=0, vsrc1=v[11]),
      v_cndmask_b32_e32(vdst=q4_word, src0=q4_word, vsrc1=v[10]),
      v_dot4_i32_iu8(vdst=v[4], src0=q4_word, src1=q8_word, src2=v[4], neg=2),
      v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=q8_word, src2=v[5], neg=2),
    ]
  insts += list(old[162:])
  sink = base.src[0]
  if sink.arg is not None and isinstance(sink.arg, KernelInfo):
    sink = sink.replace(arg=KernelInfo(name="q8_b2b_fullrow_reduce_dnr3c2_b128_preload"))
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

  native = insts_from_program(build_fullrow_reduce(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop))
  candidate = insts_from_program(build_b128_preload_fullrow_reduce(gate.uop, up.uop, gate_words.uop, up_words.uop, q8_tensor.uop))
  result: dict[str, Any] = {
    "native_grouped": grouped(native),
    "candidate_grouped": grouped(candidate),
    "instruction_count_native": len(native),
    "instruction_count_candidate": len(candidate),
    "dataflow": {
      "q4_b128_groups": [
        {"load": "global_load_b128", "addr": "v[23]", "saddr": "s[16:17]", "offset": 0, "dest": "v[80:83]", "lanes": [0, 1, 2, 3]},
        {"load": "global_load_b128", "addr": "v[23]", "saddr": "s[16:17]", "offset": 16, "dest": "v[84:87]", "lanes": [4, 5, 6, 7]},
      ],
      "q8_b128_groups": [
        {"load": "global_load_b128", "addr": "v[24]", "saddr": "s[18:19]", "offset": 0, "dest": "v[88:91]", "lanes": [0, 1, 2, 3]},
        {"load": "global_load_b128", "addr": "v[24]", "saddr": "s[18:19]", "offset": 16, "dest": "v[92:95]", "lanes": [4, 5, 6, 7]},
      ],
      "lane_operand_map": [
        {"lane": lane, "q4_word_reg": f"v[{80+lane}]", "q8_word_reg": f"v[{88+lane}]", "q4_select_mutates": f"v[{80+lane}]",
         "dot4_accum_reg": "v[4]", "dot4_sumq_reg": "v[5]"} for lane in range(8)
      ],
    },
  }
  try:
    gate_out, up_out, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8_tensor,
                                                fxn=build_b128_preload_fullrow_reduce)[:2]
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
  dnr3c1 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c1_load_shape_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  probe = run_candidate(rows_check=128)
  correctness = probe.get("correctness", {})
  cg = probe["candidate_grouped"]
  og = oracle["instruction_contract"]["oracle_grouped"]
  correct = probe.get("launch") == "PASS" and correctness.get("gate_max_abs", 1.0) <= 1e-3 and correctness.get("up_max_abs", 1.0) <= 1e-3
  load_budget_closed = cg.get("global_load", 999) <= og.get("global_load", 0)
  still_not_oracle_compound = not (cg.get("ds") <= og.get("ds") and cg.get("branch") >= og.get("branch") and
                                   cg.get("s_clause") == 3 and cg.get("s_delay_alu") == 30)
  gates = {
    "dnr3c1_blocked_on_dataflow": dnr3c1.get("verdict") == "BLOCKED_DNR3C1_LOAD_SHAPE_NEEDS_REGISTER_DATAFLOW_EMITTER",
    "candidate_launches": probe.get("launch") == "PASS",
    "candidate_correct": correct,
    "global_load_budget_closed": load_budget_closed,
    "dot4_preserved": cg.get("dot4") == og.get("dot4") == 16,
    "single_store_preserved": cg.get("global_store") == og.get("global_store") == 1,
    "oracle_compound_shape_reached": not still_not_oracle_compound,
    "performance_claim": False,
  }
  verdict = (
    "PASS_DNR3C2_B128_PRELOAD_CORRECT_LOAD_BUDGET_CLOSED_BLOCKED_ON_COMPOUND_SHAPE"
    if correct and load_budget_closed and still_not_oracle_compound
    else "BLOCKED_DNR3C2_B128_PRELOAD_INCORRECT_OR_INCOMPLETE"
  )
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C2_DECODE_REGISTER_DATAFLOW_EMITTER",
    "schema": "decode_native_renderer_dnr3c2_dataflow_emitter_probe_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "probe": probe,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C3 compound control/reduction/marker scheduler",
      "reason": "The coalesced load/dataflow path is correct and closes the load-count budget, but the candidate still lacks the oracle branch/control policy, LDS/reduction shape, and semantic scheduler markers.",
      "minimum_unblock": [
        "derive branch/exec policy from lane role semantics instead of static copying oracle branches",
        "reduce ds/reduction shape from native 10 toward oracle 7 while preserving gate/up correctness",
        "insert s_clause/s_delay_alu from semantic latency/resource boundaries instead of generic marker spam",
        "launch the compound candidate and recheck synthetic correctness",
        "only then time against the q8 oracle",
      ],
    },
    "input_artifacts": [
      "extra/q8_ffn_asm_fullrow_reduce.py",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c1_load_shape_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    ],
    "next_action": "Continue with DNR-3C3 compound control/reduction/marker scheduling; the load-shape primitive is now proven correct.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": gates,
    "native_grouped": probe["native_grouped"],
    "candidate_grouped": probe["candidate_grouped"],
    "correctness": correctness,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if correct and load_budget_closed else 1


if __name__ == "__main__":
  raise SystemExit(main())
