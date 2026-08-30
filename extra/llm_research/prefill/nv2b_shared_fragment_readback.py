"""NV2b K-fragment shared-tile versus direct-global readback oracle."""
import argparse, json, traceback
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, PackedFragmentLoopSpec
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.renderer.isa.amd_attention_abi import lower_cooperative_tile_load, expand_loop_fragment
from tinygrad.uop.ops import CooperativeTileLoadSpec
from tinygrad.dtype import AddrSpace

N = 2048
HEAD_BLOCKS = 8
CONSUMER_LANES = 32
FRAGS = CONSUMER_LANES * HEAD_BLOCKS * 16

def emitter(out_ph, inp_ph):
  sem = UOp.cooperative_tile_load(inp_ph, UOp.const(dtypes.weakint, 0),
    CooperativeTileLoadSpec(tile_base=UOp.const(dtypes.weakint, 0)))
  shared = lower_cooperative_tile_load(sem)
  lane = UOp.special(128, "lidx0")
  consumer_lane = lane.alu(Ops.AND, UOp.const(dtypes.weakint, 31))
  vals = []
  for role, owner in (("shared", shared),):
    for block in range(HEAD_BLOCKS):
      spec = PackedFragmentLoopSpec(native_abi=("nv_sm120_packed_fragment_hd128_loop_v1" if role == "shared" else
        "amd_gfx1100_packed_fragment_hd128_loop_v1"), role="V",
        head_block=block, storage="shared" if role == "shared" else "global",
        shared_phase_abi="single_buffer_barrier_v1" if role == "shared" else None)
      row = consumer_lane.alu(Ops.AND, UOp.const(dtypes.weakint, 15))
      frag = UOp(Ops.PACKED_FRAGMENT_LOAD, dtypes.half.vec(16),
        (owner, consumer_lane, row, UOp.const(dtypes.weakint, 0)), arg=spec)
      lowered = expand_loop_fragment(frag) if role == "shared" else UOp(Ops.STACK, dtypes.half.vec(16),
        tuple(inp_ph.index(row*128 + block*16 + i).load() for i in range(16)))
      for i in range(16):
        out_idx = consumer_lane*UOp.const(dtypes.weakint, HEAD_BLOCKS*16) + UOp.const(dtypes.weakint, block*16+i)
        vals.append(out_ph.index(out_idx, ptr=True).store(lowered.gep(i), lane < 32))
  return UOp.sink(*vals, arg=KernelInfo(name="nv2b_shared_fragment_readback"))

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
  result = {"schema":"nv2b_shared_fragment_readback/v1", "status":"FAILED"}
  try:
    p = KernelProgram("nv2b_shared_fragment_readback", "nv2b_shared_fragment_readback_v1",
      KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED, emitter,
      output_spec=OutputSpec((FRAGS,), dtypes.half))
    x = (np.arange(N, dtype=np.float32) * 0.25 - 17).astype(np.float16)
    inp = Tensor(x, device="NV").realize()
    got = execute_promoted_program(None, inp, program=p).realize().numpy()
    expected=np.asarray([x[(lane % 16)*128 + block*16 + i] for lane in range(32) for block in range(8) for i in range(16)], dtype=np.float16)
    result.update(status="PASS" if np.array_equal(got, expected) else "FAIL",
      exact=bool(np.array_equal(got, expected)), finite=bool(np.isfinite(got).all()),
      max_abs=float(np.max(np.abs(got-expected))),
      shared_head=got[:32].astype(float).tolist(), expected_head=expected[:32].astype(float).tolist(),
      census={"shared_define_local":1,"shared_elements":2048,"barriers":1,
              "shared_fragments":256,"direct_fragments":256,"fragment_values":FRAGS})
  except Exception as e:
    result.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
  with open(a.out, "w") as f: json.dump(result, f, indent=2, sort_keys=True); f.write("\n")
if __name__ == "__main__": main()
