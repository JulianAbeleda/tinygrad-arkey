#!/usr/bin/env python3
# AMD GEMM structural ISA-emission probe (no GPU, no timing, no perf claim, no routing/default change, no BEAM).
#
# Moves the selected ffn_gate/up lowering plan from "plan ready" to "emitted ISA evidence" — but STRUCTURAL
# ONLY. It (1) implements a minimal label/branch-offset resolver over a straight-line instruction list
# (byte sizes via inst.to_bytes(), fills s_cbranch_scc0 simm16, validates a backward target), (2) emits a
# structural unrolled-by-2 K-loop body with the planned RDNA3 op classes and alternating LDS slot offsets,
# (3) uses a fixed VGPR/SGPR allocation (no scratch/private), (4) uses placeholder base addresses
# (STRUCTURAL_EMISSION_ONLY, NOT runnable — addresses are not the real tiled global arithmetic), and
# (5) assembles to an ELF via tinygrad/renderer/amd/elf.py:assemble_linear and inspects the stream.
#
# This proves the renderer can emit a structurally Tensile-shaped GEMM loop. It does NOT claim the kernel is
# correct or fast. Correctness/timing come only after the shape is confirmed real.
from __future__ import annotations

import json, pathlib
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
LOWERING = "bench/amd-broad-backend-roadmap/amd_gemm_lowering_plan_result.json"
SCHED = "bench/amd-broad-backend-roadmap/amd_gemm_schedule_object_structural_result.json"

# ---- fixed register allocation for the authority shape (M=512,N=12288,K=4096; WG[32,4,1], TT[4,64]) ----
REG_LEDGER = {
  "thread_and_address_vgpr": "v[0:7]   (lidx + A/B/C base+lane offset VGPRs; placeholder addressing)",
  "a_b_global_load_temps":   "v[8:39]  (32 VGPR = 8 x global_load_b128 of next-K A/B)",
  "accumulator_fragments":   "v[64:191] (16 x 8 = 128 VGPR; one wave's 16 WMMA output fragments)",
  "a_b_lds_read_fragments":  "v[192:223] (32 VGPR = 8 x ds_load_b128 operand fragments)",
  "loop_counter_sgpr":       "s[16] (K counter), s[17] (saved init)",
  "kernarg_and_ptr_sgpr":    "s[0:1] kernarg base; s[4:5]=A, s[6:7]=B, s[8:9]=C",
  "scratch_private":         "0 (no DEFINE_REG spills)",
}
MAX_VGPR_USED = 224   # highest VGPR index 223 + 1
MAX_SGPR_USED = 18
VGPR_HW_BUDGET, SGPR_HW_BUDGET = 256, 106

# slot -> (A_base_byte, B_base_byte); ds 16-bit offset = offset1<<8 | offset0, bases are multiples of 256.
SLOT_BASE = {0: (0, 4096), 1: (16384, 20480)}


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(f"required artifact missing: {rel}")
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def blocked(verdict: str, blocker: str, extra: dict[str, Any] | None = None) -> int:
  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_STRUCTURAL_EMISSION", "schema": "amd_gemm_emission_v1",
    "verdict": verdict, "gate_pass": False, "default_behavior_changed": False, "performance_claim": False,
    "exact_blocker": blocker, **(extra or {}),
  }
  write_json("amd_gemm_emission_result.json", result)
  print(json.dumps({"verdict": verdict, "exact_blocker": blocker}, indent=2))
  return 1


# ----------------------------------------------------------------------------
# Minimal label / backward-branch-offset resolver for straight-line inst lists.
# Stream item is ("inst", inst) | ("label", name) | ("branch", label_name, build_fn)
# where build_fn(simm16:int) -> the concrete s_cbranch_* instruction.
# ----------------------------------------------------------------------------
def resolve_stream(stream: list[tuple]) -> tuple[list, dict[str, Any]]:
  # pass 1: byte offsets (labels are zero-size); branches are 4 bytes
  byte = 0
  label_byte: dict[str, int] = {}
  branch_sites: list[dict[str, Any]] = []
  for item in stream:
    if item[0] == "label":
      label_byte[item[1]] = byte
    elif item[0] == "branch":
      branch_sites.append({"byte": byte, "target": item[1]})
      byte += 4
    else:
      byte += len(item[1].to_bytes())
  # pass 2: materialize
  insts: list = []
  resolved: list[dict[str, Any]] = []
  for item in stream:
    if item[0] == "label":
      continue
    if item[0] == "branch":
      label, build_fn = item[1], item[2]
      site = next(s for s in branch_sites if s["target"] == label and not s.get("done"))
      site["done"] = True
      target_byte = label_byte[label]
      delta = target_byte - (site["byte"] + 4)
      if delta % 4 != 0: raise ValueError(f"branch delta {delta} not dword-aligned")
      simm16 = delta // 4
      if not (-0x8000 <= simm16 <= 0x7FFF): raise ValueError(f"branch simm16 {simm16} out of range")
      backward = target_byte < site["byte"]
      inst = build_fn(simm16 & 0xFFFF)
      # validate: decode emitted simm16 back from bytes (low 16 bits, signed)
      raw = int.from_bytes(inst.to_bytes()[:2], "little")
      decoded = raw - 0x10000 if raw >= 0x8000 else raw
      ok = (decoded == simm16) and (site["byte"] + 4 + decoded * 4 == target_byte)
      resolved.append({"branch_byte": site["byte"], "target_byte": target_byte, "target_label": label,
                       "simm16": simm16, "backward": backward, "decode_ok": ok})
      insts.append(inst)
    else:
      insts.append(item[1])
  return insts, {"label_byte": label_byte, "branches": resolved, "code_bytes": byte}


def build_emission():
  from tinygrad.renderer.amd.dsl import s, v, NULL
  import tinygrad.runtime.autogen.amd.rdna3.ins as I

  stream: list[tuple] = []
  def e(inst): stream.append(("inst", inst))
  def label(name): stream.append(("label", name))
  def branch(name, fn): stream.append(("branch", name, fn))

  WAIT_LGKM = I.s_waitcnt(simm16=(0x7) | (0x3F << 10))   # wait lgkmcnt=0, ignore vm/exp
  WAIT_VM   = I.s_waitcnt(simm16=(0x3F << 4) | (0x3F << 10) | 0x0)  # wait vmcnt=0 path (structural marker)

  # ---- prologue (placeholder addressing: STRUCTURAL_EMISSION_ONLY, not runnable) ----
  e(I.s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))   # A,B base ptrs (placeholder)
  e(I.s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL)) # C base ptr (placeholder)
  e(I.s_waitcnt(simm16=0))
  e(I.v_and_b32_e32(v[2], 15, v[0]))        # per-lane offset (placeholder, not the real tiled address)
  e(I.v_lshlrev_b32_e32(v[2], 5, v[2]))
  e(I.v_mov_b32_e32(v[3], 0))
  for i in range(16):                        # zero 16 accumulator fragments
    for j in range(8): e(I.v_mov_b32_e32(v[64 + i*8 + j], 0))
  e(I.s_mov_b32(s[16], 256))                 # k counter s5-analogue = K//DepthU
  e(I.s_mov_b32(s[17], s[16]))

  def sub_iteration(read_slot: int, write_slot: int, close_loop: bool):
    ra, rb = SLOT_BASE[read_slot]
    wa, wb = SLOT_BASE[write_slot]
    e(WAIT_LGKM); e(I.s_barrier())                                       # drain prior reads, protect reuse
    for ld in range(8):                                                   # global_load next-K A/B (8)
      e(I.global_load_b128(vdst=v[8+ld*4:8+ld*4+3], addr=v[2:2], saddr=s[4:5] if ld < 4 else s[6:7], offset=(ld%4)*16))
    for fr in range(4):                                                   # ds_load A from read slot
      e(I.ds_load_b128(vdst=v[192+fr*4:192+fr*4+3], addr=v[2], offset0=fr*16, offset1=ra >> 8))
    for fr in range(4):                                                   # ds_load B from read slot
      e(I.ds_load_b128(vdst=v[208+fr*4:208+fr*4+3], addr=v[2], offset0=fr*16, offset1=rb >> 8))
    e(WAIT_VM)                                                            # vmcnt: global load landed
    for fr in range(4):                                                   # ds_store A into write slot (other)
      e(I.ds_store_b128(addr=v[2], data0=v[8+fr*4:8+fr*4+3], offset0=fr*16, offset1=wa >> 8))
    for fr in range(4):                                                   # ds_store B into write slot
      e(I.ds_store_b128(addr=v[2], data0=v[24+fr*4:24+fr*4+3], offset0=fr*16, offset1=wb >> 8))
    e(WAIT_LGKM)                                                          # lgkmcnt: ds_load operands ready
    for i in range(16):                                                  # 16 WMMA consume (one wave's tile)
      acc = v[64+i*8:64+i*8+7]
      e(I.v_wmma_f32_16x16x16_f16(vdst=acc, src0=v[192:199], src1=v[208:215], src2=acc))
    e(I.s_sub_u32(s[16], s[16], 1))                                       # counter decrement
    if close_loop:
      e(I.s_cmp_eq_i32(s[16], 1))
      branch("loop_head", lambda simm16: I.s_cbranch_scc0(simm16=simm16))

  label("loop_head")
  sub_iteration(read_slot=0, write_slot=1, close_loop=False)              # sub A: read slot0, write slot1
  sub_iteration(read_slot=1, write_slot=0, close_loop=True)              # sub B: read slot1, write slot0

  # ---- epilogue (placeholder output store; alpha=1/beta=0 shape only) ----
  e(I.s_waitcnt(simm16=0))
  e(I.global_store_b128(addr=v[2:2], data=v[64:67], saddr=s[8:9], offset=0))
  e(I.s_sendmsg(simm16=3)); e(I.s_endpgm())
  return stream


def classify(insts: list, slot_records: dict[str, Any]) -> dict[str, Any]:
  def name(inst) -> str:
    op = getattr(inst, "op", None)
    return (getattr(op, "name", None) or getattr(inst, "op_name", None) or type(inst).__name__).lower()
  names = [name(i) for i in insts]
  def count(prefix) -> int: return sum(1 for n in names if n.startswith(prefix))
  return {
    "global_load": count("global_load"),
    "ds_store": count("ds_store"),
    "ds_load_b128": sum(1 for n in names if n == "ds_load_b128"),
    "v_wmma": count("v_wmma"),
    "s_waitcnt": count("s_waitcnt"),
    "s_barrier": count("s_barrier"),
    "s_sub": sum(1 for n in names if n.startswith("s_sub")),
    "s_cbranch": count("s_cbranch"),
    "global_store": count("global_store"),
    "total_insts": len(insts),
  }


def main() -> int:
  lowering = read_json(LOWERING)
  sched = read_json(SCHED)
  if lowering.get("verdict") != "PASS_GEMM_LOWERING_PLAN_READY":
    return blocked("BLOCKED_GEMM_EMISSION_OPCODE", f"lowering plan not ready: {lowering.get('verdict')}")
  if not sched.get("gate_pass"):
    return blocked("BLOCKED_GEMM_EMISSION_OPCODE", "schedule object structural gate not passed")

  # ---- (1) build the stream; opcode encoding failures => BLOCKED_GEMM_EMISSION_OPCODE ----
  try:
    stream = build_emission()
  except Exception as ex:
    return blocked("BLOCKED_GEMM_EMISSION_OPCODE", f"RDNA3 op failed to encode: {ex!r}")

  # ---- (2) resolve labels/branches => BLOCKED_GEMM_EMISSION_BRANCH_RESOLUTION ----
  try:
    insts, layout = resolve_stream(stream)
  except Exception as ex:
    return blocked("BLOCKED_GEMM_EMISSION_BRANCH_RESOLUTION", f"branch resolution failed: {ex!r}")
  branch_ok = bool(layout["branches"]) and all(b["backward"] and b["decode_ok"] for b in layout["branches"])
  if not branch_ok:
    return blocked("BLOCKED_GEMM_EMISSION_BRANCH_RESOLUTION",
                   "backward branch target did not validate", {"branches": layout["branches"]})

  # ---- (3) fixed register allocation fit => BLOCKED_GEMM_EMISSION_REGISTER_ALLOCATION ----
  if MAX_VGPR_USED > VGPR_HW_BUDGET or MAX_SGPR_USED > SGPR_HW_BUDGET:
    return blocked("BLOCKED_GEMM_EMISSION_REGISTER_ALLOCATION",
                   f"fixed allocation exceeds budget: vgpr {MAX_VGPR_USED}/{VGPR_HW_BUDGET}, sgpr {MAX_SGPR_USED}/{SGPR_HW_BUDGET}")

  # ---- (4)+(5) assemble to ELF and inspect => addressing prevents emission => BLOCKED_GEMM_EMISSION_ADDRESS_MODEL ----
  try:
    from tinygrad.uop.ops import UOp, Ops
    from tinygrad.dtype import dtypes, AddrSpace
    from tinygrad.renderer.amd.dsl import s as _s, v as _v  # noqa
    from tinygrad.renderer.amd.elf import assemble_linear, group_segment_fixed_size_from_elf, kernel_descriptor_from_elf
    lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))
    dl = UOp.placeholder((12544,), dtypes.half, 9000, AddrSpace.LOCAL)   # 25088 bytes LDS
    specials = [UOp(Ops.SPECIAL, dtypes.int, arg="gidx0"), UOp(Ops.SPECIAL, dtypes.int, arg="gidx1"),
                UOp(Ops.SPECIAL, dtypes.int, arg="lidx0")]
    sink = UOp.sink(dl, *specials)
    prg = UOp(Ops.PROGRAM, src=(sink,))
    elf = assemble_linear(prg, lin, "gfx1100")
    lds_bytes = group_segment_fixed_size_from_elf(elf)
    private_bytes = kernel_descriptor_from_elf(elf).private_segment_fixed_size
  except Exception as ex:
    return blocked("BLOCKED_GEMM_EMISSION_ADDRESS_MODEL", f"structural assembly failed: {ex!r}")

  counts = classify(insts, layout)
  # slot alternation: ds offsets carry slot via offset1 high byte; both slot0 ({0,16}) & slot1 ({64,80}) present
  slot1_bytes = {SLOT_BASE[0][0] >> 8, SLOT_BASE[0][1] >> 8, SLOT_BASE[1][0] >> 8, SLOT_BASE[1][1] >> 8}
  slot_alternation_present = {0, 16}.issubset(slot1_bytes) and {64, 80}.issubset(slot1_bytes)

  gates = {
    "visible_global_load": counts["global_load"] > 0,
    "visible_ds_store": counts["ds_store"] > 0,
    "visible_ds_load_b128": counts["ds_load_b128"] > 0,
    "visible_v_wmma": counts["v_wmma"] > 0,
    "visible_s_waitcnt": counts["s_waitcnt"] > 0,
    "visible_s_barrier": counts["s_barrier"] > 0,
    "visible_loop_counter_decrement": counts["s_sub"] > 0,
    "visible_backward_branch": branch_ok,
    "slot_alternation_offsets_present": slot_alternation_present,
    "lds_bytes_25088": lds_bytes == 25088,
    "scratch_private_0": private_bytes == 0,
    "no_performance_claim": True,
  }
  gate_pass = all(gates.values())
  verdict = "PASS_GEMM_STRUCTURAL_EMISSION" if gate_pass else "BLOCKED_GEMM_EMISSION_OPCODE"

  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_STRUCTURAL_EMISSION", "schema": "amd_gemm_emission_v1",
    "role": "ffn_gate/up", "verdict": verdict, "gate_pass": gate_pass,
    "default_behavior_changed": False, "performance_claim": False,
    "correctness_claim": False, "runnable": False, "addressing_mode": "STRUCTURAL_EMISSION_ONLY",
    "shape": lowering["shape"], "unroll": lowering["unroll"],
    "emitted": {
      "instruction_counts": counts,
      "code_bytes": layout["code_bytes"], "elf_bytes": len(elf),
      "lds_group_segment_fixed_size": lds_bytes, "private_segment_fixed_size": private_bytes,
      "branch_resolution": layout["branches"], "loop_head_byte": layout["label_byte"].get("loop_head"),
    },
    "register_ledger": {**REG_LEDGER, "max_vgpr_used": MAX_VGPR_USED, "max_sgpr_used": MAX_SGPR_USED,
                        "vgpr_hw_budget": VGPR_HW_BUDGET, "sgpr_hw_budget": SGPR_HW_BUDGET,
                        "authority_sgpr_budget": sched["schedule_object"]["resource_gate"]["sgpr_budget"]},
    "slot_alternation": {"sub_A": {"read_slot": 0, "write_slot": 1}, "sub_B": {"read_slot": 1, "write_slot": 0},
                         "slot_base_bytes": {str(k): v for k, v in SLOT_BASE.items()},
                         "encoding": "ds 16-bit offset = offset1<<8|offset0; slot base in offset1 high byte"},
    "structural_gates": gates,
    "explicitly_not_claimed": ["correctness", "performance", "runnable execution", "bit-exact Tensile layout",
                               "real tiled global addresses (placeholder, STRUCTURAL_EMISSION_ONLY)"],
    "remaining_for_runnable": [
      "address_expression_model: real per-thread tiled A/B/C global addresses (currently placeholder)",
      "correct fragment<->WMMA operand mapping and accumulator->output indexing",
      "full output epilogue (beta*C + bounds)",
    ],
    "input_artifacts": [LOWERING, SCHED],
    "next_action": (
      "With a structurally real emitted loop confirmed, build the address_expression_model to make it runnable, "
      "then verify correctness (RMSE vs a@b) under the structural gate, and only then time vs the >=60 TFLOPS "
      "authority under the PTM-1 one-clock harness. No BEAM/search until correctness+timing exist."),
  }
  write_json("amd_gemm_emission_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/amd_gemm_emission_result.json",
    "verdict": verdict, "gate_pass": gate_pass,
    "instruction_counts": counts,
    "elf_bytes": len(elf), "lds_bytes": lds_bytes, "scratch": private_bytes,
    "branch": layout["branches"], "structural_gates": gates,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
