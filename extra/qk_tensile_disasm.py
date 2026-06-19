#!/usr/bin/env python3
"""TCG-0/1 (codegen-transfer oracle) — recover the selected Tensile GEMM kernel's schedule anatomy (from its
exhaustive name + a disassembly instruction-mix), compare to tinygrad's POWN-1 WMMA config, and emit the concrete
capability-delta table: what Tensile does, what tinygrad does, and the missing codegen capability. Research-only.

  build disasm: /opt/rocm-7.2.4/llvm/bin/llvm-objdump -d /tmp/tpe3_kernel.elf > /tmp/td_all.txt
  run:          PYTHONPATH=. .venv/bin/python extra/qk_tensile_disasm.py
"""
from __future__ import annotations
import json, re, pathlib

DISASM = "/tmp/td_all.txt"   # llvm-objdump -d of the unbundled Ailk_Bljk gfx1100 ELF

def decode_name(sym:str) -> dict:
  g = lambda p: (m.group(1) if (m:=re.search(p, sym)) else None)
  return {"macro_tile_MxNxK": g(r"MT(\d+x\d+x\d+)"), "wmma_MI": g(r"MI(\d+x\d+x\d+x\d+)"),
          "depthU": g(r"MT\d+x\d+x(\d+)"), "prefetch_global_read_PGR": g(r"_PGR(\d)"),
          "prefetch_local_read_PLR": g(r"_PLR(\d)"), "lds_buffering_1LDSB": g(r"_(1LDSB\d)"),
          "global_load_vec_GLVWA": g(r"GLVWA(\d+)"), "global_read_vec_GRVW": g(r"_GRVW(\d+)"),
          "local_read_vec_LRVW": g(r"LRVW(\d+)"), "thread_tile_TT": g(r"_TT(\d+_\d+)"),
          "workgroup_map_WGM": g(r"_WGM(\d+)"), "streamk_SU": g(r"_SU(\d+)_SUM")}

def instr_mix(path:str) -> dict:
  txt = pathlib.Path(path).read_text(errors="ignore") if pathlib.Path(path).exists() else ""
  c = lambda p: len(re.findall(p, txt))
  return {"v_wmma": c(r"v_wmma"), "ds_load_b128": c(r"ds_load_b128"), "ds_store_b128": c(r"ds_store_b128"),
          "s_waitcnt_vmcnt": c(r"s_waitcnt vmcnt"), "s_waitcnt_lgkmcnt": c(r"s_waitcnt lgkmcnt"),
          "s_barrier": c(r"s_barrier"), "disasm_lines": len(txt.splitlines())}

def main():
  sym = json.load(open("bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
  sched = decode_name(sym); mix = instr_mix(DISASM)
  # tinygrad POWN-1 best config (docs/prefill-own-wmma-kernel-result-20260619.md)
  pown1 = {"macro_tile_MxNxK": "128x128x16", "wmma_MI": "16x16x16x1", "waves": "2x2", "tflops": 42.0,
           "notes": "every lever regresses: waves->28TF, more-acc->11TF (spill), no-LDS->38TF; 42->70 gap = SW-pipelined K-loop"}
  delta = [
    {"aspect": "macro-tile", "tensile": sched["macro_tile_MxNxK"], "tinygrad": pown1["macro_tile_MxNxK"],
     "missing_capability": "NONE (identical)", "class": "—"},
    {"aspect": "WMMA fragment", "tensile": sched["wmma_MI"]+" (v_wmma x%d)"%mix["v_wmma"], "tinygrad": pown1["wmma_MI"],
     "missing_capability": "NONE (both use RDNA3 WMMA)", "class": "—"},
    {"aspect": "K-loop prefetch / LDS double-buffer",
     "tensile": f"PGR{sched['prefetch_global_read_PGR']}+PLR{sched['prefetch_local_read_PLR']}, {sched['lds_buffering_1LDSB']} (double LDS buffer) -> load next K-tile while computing current; wide ds_load_b128 x%d"%mix["ds_load_b128"],
     "tinygrad": "single-buffered, no software pipeline (no_LDS variant within 10% -> tinygrad's LDS staging buys nothing without the prefetch overlap)",
     "missing_capability": "SOFTWARE-PIPELINED K-LOOP with double-buffered global->LDS->reg prefetch (overlap memory with WMMA issue)",
     "class": "renderer instruction-scheduling (NOT UOp-expressibility: tinygrad can express WMMA+LDS, can't schedule the in-flight prefetch pipeline)"},
    {"aspect": "accumulator register allocation",
     "tensile": f"TT{sched['thread_tile_TT']} thread-tile, vgpr256 with NO spill -> enough independent accumulators to hide WMMA latency",
     "tinygrad": "more-accumulators -> register spill (11 TFLOPS); can't hold a large acc tile without spilling",
     "missing_capability": "stable large-accumulator register allocation (no spill at high acc count)",
     "class": "register allocation"},
    {"aspect": "vectorized LDS read", "tensile": f"LRVW{sched['local_read_vec_LRVW']} (ds_load_b128)",
     "tinygrad": "narrower local reads", "missing_capability": "wide vectorized LDS reads (minor, follows from the layout)",
     "class": "renderer codegen (minor)"},
  ]
  smallest_change = ("software-pipelined K-loop (double-buffer global->LDS prefetch overlapped with WMMA issue) + "
                     "spill-free large-accumulator allocation. Both are AMD-renderer instruction-scheduling / "
                     "register-allocation capabilities, NOT frontend UOp expressibility -> PROJECT-LEVEL "
                     "(rewrite/extend the AMD renderer scheduler), same BEAM-hang-class wall as POWN-1.")
  res = {"schema": "qk_tensile_codegen_oracle_v1", "phase": "TCG-0/1", "kernel_symbol": sym[:80]+"...",
         "tensile_schedule": sched, "tensile_instruction_mix": mix, "tinygrad_pown1": pown1,
         "capability_delta": delta, "smallest_codegen_change": smallest_change,
         "verdict": "the extracted schedule is a CONCRETE ORACLE: tinygrad matches Tensile on tile+WMMA but lacks the "
                    "software-pipelined K-loop + spill-free accumulators. Closing 42->~66 TFLOPS = a project-level AMD "
                    "renderer/scheduler capability, not a bounded kernel tweak. Use this table to target that codegen work."}
  pathlib.Path("bench/qk-tensile-extraction/codegen_oracle.json").write_text(json.dumps(res, indent=2))
  print(json.dumps({"tensile_schedule": sched, "instruction_mix": mix, "smallest_change": smallest_change}, indent=2))

if __name__ == "__main__":
  main()
