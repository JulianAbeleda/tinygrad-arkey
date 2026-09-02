#!/usr/bin/env python3
"""Native Flash fragment readback using the promoted-program ABI."""
import argparse, json, traceback
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, PackedFragmentLoopSpec, CooperativeTileLoadSpec
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.renderer.isa.amd_attention_abi import lower_cooperative_tile_load, expand_loop_fragment

N, HEAD_BLOCKS, LANES = 2048, 8, 32
FRAGS = LANES * HEAD_BLOCKS * 16
MODES = ("q-global", "q-shared", "k-global", "k-shared", "v-global", "v-shared")

def emitter(fragment_mode):
  if fragment_mode not in MODES: raise ValueError(fragment_mode)
  shared_mode, role = fragment_mode.endswith("shared"), fragment_mode[0].upper()
  def emit(out_ph, inp_ph):
    sem = UOp.cooperative_tile_load(inp_ph, UOp.const(dtypes.weakint, 0), CooperativeTileLoadSpec(tile_base=UOp.const(dtypes.weakint, 0)))
    shared = lower_cooperative_tile_load(sem)
    lane = UOp.special(128, "lidx0"); consumer_lane = lane.alu(Ops.AND, UOp.const(dtypes.weakint, 31)); vals = []
    for block in range(HEAD_BLOCKS):
      spec = PackedFragmentLoopSpec(native_abi=("nv_sm120_packed_fragment_hd128_loop_v1" if shared_mode else "amd_gfx1100_packed_fragment_hd128_loop_v1"), role=role, head_block=block, storage="shared" if shared_mode else "global", shared_phase_abi="single_buffer_barrier_v1" if shared_mode else None)
      row = consumer_lane.alu(Ops.AND, UOp.const(dtypes.weakint, 15))
      if shared_mode:
        frag = UOp(Ops.PACKED_FRAGMENT_LOAD, dtypes.half.vec(16), (shared, consumer_lane, row, UOp.const(dtypes.weakint, 0)), arg=spec)
        lowered = expand_loop_fragment(frag)
      else:
        lowered = UOp(Ops.STACK, dtypes.half.vec(16), tuple(inp_ph.index(row*128 + block*16 + i).load() for i in range(16)))
      for i in range(16):
        out_idx = consumer_lane*UOp.const(dtypes.weakint, HEAD_BLOCKS*16) + UOp.const(dtypes.weakint, block*16+i)
        vals.append(out_ph.index(out_idx, ptr=True).store(lowered.gep(i), lane < 32))
    return UOp.sink(*vals, arg=KernelInfo(name="nv_flash_fragment_readback"))
  return emit

def run():
  x = (np.arange(N, dtype=np.float32) * 0.25 - 17).astype(np.float16); inp = Tensor(x, device="NV").realize(); got = {}
  for mode in MODES:
    p = KernelProgram("nv_flash_fragment_readback", "nv_flash_fragment_readback_v1", KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED, emitter(mode), output_spec=OutputSpec((FRAGS,), dtypes.half))
    got[mode] = execute_promoted_program(None, inp, program=p).realize().numpy().copy()
  expected = np.asarray([x[(lane % 16)*128 + block*16 + i] for lane in range(32) for block in range(8) for i in range(16)], dtype=np.float16)
  result = {"schema":"tinygrad.nv_flash_fragment_readback/v2", "shape":{"tokens":16,"Hd":128}}
  for mode in MODES:
    d = np.abs(got[mode].astype(np.float32)-expected.astype(np.float32)); bad = np.flatnonzero(d > 0)
    result[mode] = {"exact":bool(np.array_equal(got[mode], expected)), "max_abs":float(d.max()), "first_mismatch":None if len(bad)==0 else {"flat":int(bad[0]), "lane":int(bad[0]//128), "head_block":int((bad[0]%128)//16), "elem":int(bad[0]%16)}}
  result["k_shared_vs_global_max_abs"] = float(np.max(np.abs(got["k-shared"].astype(np.float32)-got["k-global"].astype(np.float32))))
  result["v_shared_vs_global_max_abs"] = float(np.max(np.abs(got["v-shared"].astype(np.float32)-got["v-global"].astype(np.float32))))
  result["status"] = "PASS" if all(result[m]["exact"] for m in MODES) else "MISMATCH"; return result

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--out"); args=ap.parse_args()
  try: result=run()
  except Exception as e: result={"status":"ERROR","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}
  text=json.dumps(result,indent=2,sort_keys=True); print(text)
  if args.out: open(args.out,"w").write(text+"\n")
