"""Run the AMDISARenderer's assembled kernel through the remu RDNA3 functional emulator (github.com/Qazalin/remu)
and compare to the numpy reference. remu models real RDNA3 instruction semantics incl. v_wmma_f32_16x16x16_f16, so:
  remu bit-exact but GPU=NaN  -> the logical stream is correct => a HARDWARE hazard (timing/scoreboard)
  remu also wrong             -> a LOGIC bug in the emitted stream (then single-step with wave_* to localize)
No GPU needed. libremu.so is the prebuilt 0.1.2 release (has run_asm + WMMA)."""
import os, sys, ctypes
os.environ["ALLOW_DEVICE_USAGE"]="1"
import numpy as np
sys.path.insert(0, os.getcwd())
from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.codegen import to_program

LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")

def kernel_bytes(M,N,K):
  """Capture the FINAL resolved instruction stream (post schedule/waitcnt/label-resolution) as raw .text bytes."""
  ren=AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  cap={}
  orig=ren._resolve_labels
  def wrap(insts):
    r=orig(insts); cap['final']=list(r); return r
  ren._resolve_labels=wrap
  a=Tensor.empty(M,K,dtype="half"); b=Tensor.empty(K,N,dtype="half")
  ast=[u for u in (a@b).schedule_linear().toposort() if u.op is Ops.SINK][0]
  to_program(ast,ren)
  raw=b"".join(u.arg.to_bytes() for u in cap['final'])
  assert len(raw)%4==0, len(raw)
  return raw

def run_remu(M,N,K, seed=0):
  np.random.seed(seed)
  A=np.random.randn(M,K).astype(np.float16)
  B=np.random.randn(K,N).astype(np.float16)
  OUT=np.zeros((M,N),dtype=np.float16)
  text=kernel_bytes(M,N,K)
  # kernarg buffer: [out_ptr, A_ptr, B_ptr] at offsets 0x0/0x8/0x10 (tinygrad order [out, *ins])
  args=(ctypes.c_uint64*3)(OUT.ctypes.data, A.ctypes.data, B.ctypes.data)
  lib=ctypes.CDLL(LIBREMU)
  lib.run_asm.restype=ctypes.c_int
  lib.run_asm.argtypes=[ctypes.c_char_p, ctypes.c_uint32]+[ctypes.c_uint32]*6+[ctypes.POINTER(ctypes.c_uint64)]
  # single workgroup, wave32
  rc=lib.run_asm(ctypes.c_char_p(text), len(text), 1,1,1, 32,1,1, args)
  ref=(A.astype(np.float32)@B.astype(np.float32))
  got=OUT.astype(np.float32)
  nanfrac=float(np.isnan(got).mean())
  ok=np.isfinite(got)
  rmse=float(np.sqrt(((got[ok]-ref[ok])**2).mean())) if ok.any() else float('nan')
  print(f"remu {M}x{N}x{K}: rc={rc} nan_frac={nanfrac:.4f} rmse(non-nan)={rmse:.5f} PASS={rmse<5e-2 and nanfrac==0}")
  print(f"   got[0,:6]={got[0,:6]}")
  print(f"   ref[0,:6]={ref[0,:6]}")
  return rc

if __name__=="__main__":
  MNK=[(int(x) for x in a.split('x')) for a in sys.argv[1:]] if len(sys.argv)>1 else [(32,64,64),(64,32,64),(64,64,64)]
  for m,n,k in ([(32,64,64),(64,32,64),(64,64,64)] if len(sys.argv)==1 else [tuple(int(x) for x in s.split('x')) for s in sys.argv[1:]]):
    run_remu(m,n,k)
