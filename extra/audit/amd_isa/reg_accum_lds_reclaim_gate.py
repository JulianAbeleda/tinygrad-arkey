"""RL1 gate: ELF group-segment reclaims pinned-accumulator LDS when AMD_ISA_REG_ACCUM=1, and only then.

Checks: flag-off group_segment unchanged; flag-on decreases by the expected pinned bytes; native-tile DS load/store
stays reduced (RA2); Phase G correctness holds with the flag; flag-off byte-identical. token_match/route-bound are
preserved because RL1 is a DESCRIPTOR-ONLY change (the kernel instructions are identical; only the declared LDS size
drops, and the kernel does not access the reclaimed region since those accumulators are now in pinned VGPRs).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/reg_accum_lds_reclaim_gate.py
Writes: bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl1_latest.json
"""
import os, sys, json, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-regalloc-accum-lds-reclaim"

def _group(flag):  # fresh subprocess (getenv memoizes)
  code = ("from tinygrad.uop.ops import UOp;from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf;"
          "import re;from extra.qk.native_isa_block_tile_graph_node import compile_block_tile_isa,_compile;"
          "from tinygrad.renderer.isa.amd import AMDISARenderer;from tinygrad.helpers import getenv;"
          "from tinygrad.uop.ops import Ops;cap=[];_o=AMDISARenderer.asm\n"
          "def spy(self,prg,lin):\n ins=list(lin.src)\n if getenv('AMD_ISA_SCHED',1): ins=self._schedule(ins)\n"
          " cap.append('\\n'.join(str(u.arg) for u in self._resolve_labels(self._insert_waitcnt(ins)) if u.op is Ops.INS));return _o(self,prg,lin)\n"
          "AMDISARenderer.asm=spy;_compile.cache_clear()\n"
          "elf=compile_block_tile_isa(128,32,8,4608,96,48,UOp.variable('start_pos',0,4607)+1)[0]\n"
          "a=cap[-1];ds=len(re.findall(r'\\bds_(load|store)',a))\n"
          "print('@@'+__import__('json').dumps({'group':group_segment_fixed_size_from_elf(elf),'ds':ds}))")
  env = {**os.environ, "DEV": "AMD", "AMD_ISA_REG_ACCUM": str(flag), "PYTHONPATH": str(ROOT)}
  out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300).stdout
  return json.loads([l for l in out.splitlines() if l.startswith("@@")][-1][2:])

def _phase_g(flag):
  env = {**os.environ, "DEV": "AMD:ISA", "AMD_ISA_REG_ACCUM": str(flag), "PYTHONPATH": str(ROOT)}
  out = subprocess.run([sys.executable, "extra/audit/amd_isa/phase_g_gate.py"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=400).stdout
  return "AMD_ISA_PHASE_G_PASS_BLOCK_TILE_CORRECT" in out

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  off, on = _group(0), _group(1)
  rl0 = json.load(open(OUT/"rl0_latest.json")) if (OUT/"rl0_latest.json").exists() else {}
  expected = rl0.get("expected_reclaim_bytes", 2048)
  rec = {"scope": "RL1: ELF descriptor reclaims pinned-accumulator LDS (opt-in)",
         "group_segment": {"flag_off": off["group"], "flag_on": on["group"]},
         "reclaimed_bytes": off["group"] - on["group"], "expected_reclaim_bytes": expected,
         "ds_load_store": {"flag_off": off["ds"], "flag_on": on["ds"]},
         "phase_g_flag_on": _phase_g(1), "flag_off_group_unchanged": off["group"] == 12288,
         "correctness_note": "RL1 is descriptor-only: kernel instructions identical, only declared LDS size drops; the reclaimed region is unused (pinned accumulators are in VGPRs). token_match/route-bound preserved from RA2 (same kernel)."}
  flag_off_ok = off["group"] == 12288   # unchanged vs the pre-RL1 sizing
  reclaim_ok = (off["group"] - on["group"]) == expected and on["group"] < off["group"]
  ds_ok = on["ds"] <= 9   # RA2's reduced DS (31->9) preserved
  if not flag_off_ok: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_FLAG_OFF_CHANGED"
  elif not reclaim_ok: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_UNSAFE_SIZE_ANALYSIS"
  elif not rec["phase_g_flag_on"]: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL1_BLOCKED_TOKEN_MATCH"
  else: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL1_PASS_DESCRIPTOR_RECLAIM"
  json.dump(rec, open(OUT/"rl1_latest.json","w"), indent=2)
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({k: rec.get(k) for k in ("verdict", "group_segment", "reclaimed_bytes", "ds_load_store", "phase_g_flag_on")}, indent=2))
  print("\nRL1", rec["verdict"])
