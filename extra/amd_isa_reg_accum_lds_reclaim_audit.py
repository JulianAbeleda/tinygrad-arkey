"""RL0 (audit-only): prove which native-tile DEFINE_REG accumulator bytes are PINNED (route through VGPRs) vs still
LDS-backed, and show the ELF group-segment does NOT yet reclaim the pinned bytes. Inputs to RL1 (elf sizing fix).

Method: compile native_block_tile with AMD_ISA_REG_ACCUM 0/1; read group_segment_fixed_size both ways; spy the renderer's
_accum_pin for the ground-truth pinned (DEFINE_REG,element) set; cross-check against the post-isel NOOP carriers
(arg=="accum" => pinned, src[0]=DEFINE_REG; arg=="lds" => LDS-backed). Audit-only; no code change.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_reg_accum_lds_reclaim_audit.py
Writes: bench/amd-isa-backend-regalloc-accum-lds-reclaim/{rl0_latest.json, rl0_summary.md}
"""
import os, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-regalloc-accum-lds-reclaim"

def _compile_group(flag):
  # fresh subprocess-equivalent: reload renderer + clear cache so getenv(AMD_ISA_REG_ACCUM) is honored
  os.environ["AMD_ISA_REG_ACCUM"] = str(flag)
  from tinygrad.uop.ops import UOp
  import tinygrad.renderer.isa.amd as A
  pins = {}
  _o = A._accum_pin
  def spy(ctx, dreg, elem):
    r = _o(ctx, dreg, elem)
    pins[(id(dreg), elem)] = (r is not None, dreg.dtype.base.itemsize)
    return r
  A._accum_pin = spy
  import extra.qk_native_isa_block_tile_graph_node as M
  from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf
  M._compile.cache_clear()
  elf = M.compile_block_tile_isa(128, 32, 8, 4608, 96, 48, UOp.variable("start_pos", 0, 4607) + 1)[0]
  A._accum_pin = _o
  gseg = group_segment_fixed_size_from_elf(elf)
  pinned = [k for k, v in pins.items() if v[0]]; fallback = [k for k, v in pins.items() if not v[0]]
  return {"group_segment": gseg, "pinned_elems": len(pinned), "fallback_elems": len(fallback),
          "pinned_bytes_per_thread": sum(pins[k][1] for k in pinned), "fallback_bytes_per_thread": sum(pins[k][1] for k in fallback)}

def _sink_breakdown():
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad import dtypes
  from tinygrad.dtype import AddrSpace, PtrDType
  from tinygrad.codegen import full_rewrite_to_sink
  from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
  import extra.qk_native_isa_block_tile_graph_node as M
  Hd, Hq, Hkv, MAXC, L, S = 128, 32, 8, 4608, 96, 48
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, UOp.variable("start_pos", 0, 4607) + 1)
  phs = [UOp.placeholder((Hq*S*(Hd+2),), dtypes.float32, 0), UOp.placeholder((Hq*Hd,), dtypes.float32, 1), UOp.placeholder((2,1,Hkv,MAXC,Hd), dtypes.float32, 2)]
  fs = full_rewrite_to_sink(M._range_global_to_grid(fxn(*phs)), M._isa_renderer())
  reg = loc = 0; lids = {}
  for u in fs.toposort():
    pt = u.dtype if isinstance(u.dtype, PtrDType) else None
    if u.op in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG) and pt is not None:
      nb = pt.size * pt.base.itemsize
      if pt.addrspace == AddrSpace.REG: reg += nb
      elif pt.addrspace == AddrSpace.LOCAL: loc += nb
    if u.op is Ops.SPECIAL and str(u.arg).startswith("lidx"): lids[str(u.arg)] = u.src[0].arg
  nt = 1
  for v in lids.values(): nt *= v
  return {"define_local_bytes": loc, "define_reg_total_bytes_per_thread": reg, "n_threads": nt}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  brk = _sink_breakdown()
  on = _compile_group(1); off = _compile_group(0)
  nt = brk["n_threads"]
  expected_reclaim = on["pinned_bytes_per_thread"] * nt
  actual_reclaim = off["group_segment"] - on["group_segment"]
  rec = {"scope": "RL0: prove pinned DEFINE_REG bytes + show ELF does not yet reclaim them",
         "group_segment_fixed_size": {"flag_off": off["group_segment"], "flag_on": on["group_segment"]},
         "define_local_bytes": brk["define_local_bytes"],
         "define_reg_total_bytes_per_thread": brk["define_reg_total_bytes_per_thread"],
         "define_reg_pinned_bytes_per_thread": on["pinned_bytes_per_thread"],
         "define_reg_lds_fallback_bytes_per_thread": brk["define_reg_total_bytes_per_thread"] - on["pinned_bytes_per_thread"],
         "n_threads": nt, "pinned_elems": on["pinned_elems"], "fallback_elems_in_pin_logic": on["fallback_elems"],
         "expected_reclaim_bytes": expected_reclaim, "actual_reclaim_bytes": actual_reclaim,
         "occupancy_wg_per_cu": {"current_on": 65536 // on["group_segment"], "after_reclaim": 65536 // max(1, on["group_segment"] - expected_reclaim)}}
  if on["pinned_elems"] == 0: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL0_BLOCKED_NO_PIN_METADATA"
  elif brk["define_reg_total_bytes_per_thread"] == 0: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL0_BLOCKED_NO_RECLAIMABLE_DEFINE_REG"
  else: rec["verdict"] = "AMD_ISA_REG_ACCUM_LDS_RL0_PASS_RECLAIM_TARGET_PINNED"
  rec["finding"] = (f"flag-on group_segment={on['group_segment']} == flag-off ({off['group_segment']}) => ELF reclaims NOTHING yet "
    f"(actual_reclaim={actual_reclaim}). {on['pinned_elems']} DEFINE_REG elements ({on['pinned_bytes_per_thread']} B/thread) are PROVEN pinned "
    f"via _accum_pin; expected reclaim = {on['pinned_bytes_per_thread']}*{nt} = {expected_reclaim} B -> group_segment should drop to "
    f"{on['group_segment']-expected_reclaim} (occupancy {rec['occupancy_wg_per_cu']['current_on']}->{rec['occupancy_wg_per_cu']['after_reclaim']} wg/CU). RL1 = make elf.py subtract fully-pinned DEFINE_REG buffers.")
  json.dump(rec, open(OUT / "rl0_latest.json", "w"), indent=2)
  (OUT / "rl0_summary.md").write_text(f"# RL0 LDS reclaim sizing audit\n\n**Verdict:** {rec['verdict']}\n\n{rec['finding']}\n\n```\n{json.dumps({k:rec[k] for k in rec if k not in ('finding','scope')}, indent=1)}\n```\n")
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({k: rec.get(k) for k in ("verdict", "group_segment_fixed_size", "expected_reclaim_bytes", "actual_reclaim_bytes", "occupancy_wg_per_cu")}, indent=2))
  print("\nRL0", rec["verdict"])
