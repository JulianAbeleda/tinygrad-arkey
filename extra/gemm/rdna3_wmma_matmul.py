# RDNA3 (gfx1100) WMMA GEMM — hand-assembly, zero LLVM, via assemble->ELF.
# Built incrementally: STAGE 1 = single 16x16x16 tile to nail the RDNA3 WMMA operand layout.
# RDNA3 wave32 v_wmma_f32_16x16x16_f16:
#   A (src0) = 8 VGPR/lane (16 fp16): lane l(0..15) holds A[l][0:16]; lanes 16..31 replicate.
#   B (src1) = 8 VGPR/lane (16 fp16): lane l holds B[0:16][l] (a column). We pass B TRANSPOSED in
#     memory (Bt[n][k]=B[k][n]) so a column is contiguous -> Bt[l][0:16].
#   C/D (src2/vdst) = 8 VGPR/lane (8 fp32): D[i] of lane l = C[row=i*2+(l>>4&1)][col=l&15].
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv, colored
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.renderer.amd.dsl import s, v, VCC_LO, NULL, src, ttmp
from tinygrad.runtime.autogen.amd.rdna3.ins import *

FA, FB, ACC = 20, 32, 44   # VGPR bases: A frag(8), B frag(8), accumulator(8)

def build_tile_kernel():
  I = []
  def e(i): I.append(i); return i
  # kernargs: A_ptr->s[4:5], B_ptr(transposed)->s[6:7], C_ptr->s[8:9]
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  # per-lane byte offset = (tid&15)*32  (16 fp16 row/col); v[3]=0 high
  e(v_and_b32_e32(v[2], 15, v[0]))
  e(v_lshlrev_b32_e32(v[2], 5, v[2]))
  e(v_mov_b32_e32(v[3], 0))
  for i in range(8): e(v_mov_b32_e32(v[ACC+i], 0))   # zero accumulator
  # load A[lane][0:16] = 2x b128 contiguous
  e(global_load_b128(vdst=v[FA:FA+3],   addr=v[2:2], saddr=s[4:5], offset=0))
  e(global_load_b128(vdst=v[FA+4:FA+7], addr=v[2:2], saddr=s[4:5], offset=16))
  # load Bt[lane][0:16] = column of B, contiguous because transposed
  e(global_load_b128(vdst=v[FB:FB+3],   addr=v[2:2], saddr=s[6:7], offset=0))
  e(global_load_b128(vdst=v[FB+4:FB+7], addr=v[2:2], saddr=s[6:7], offset=16))
  e(s_waitcnt(simm16=0))
  e(v_wmma_f32_16x16x16_f16(vdst=v[ACC:ACC+7], src0=v[FA:FA+7], src1=v[FB:FB+7], src2=v[ACC:ACC+7]))
  # store C: col=tid&15, parity=(tid>>4)&1, row=i*2+parity ; byte off=(row*16+col)*2
  e(v_and_b32_e32(v[10], 15, v[0]))            # col
  e(v_lshrrev_b32_e32(v[11], 4, v[0])); e(v_and_b32_e32(v[11], 1, v[11]))  # parity
  e(v_lshlrev_b32_e32(v[12], 4, v[11]))        # parity*16
  e(v_add_nc_u32_e32(v[12], v[12], v[10]))     # (row0*16 + col), row0=parity
  e(v_lshlrev_b32_e32(v[12], 1, v[12]))        # *2 bytes
  e(v_mov_b32_e32(v[13], 0))
  for i in range(8):
    e(v_cvt_f16_f32_e32(v[14], v[ACC+i]))
    e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
    if i < 7: e(v_add_nc_u32_e32(v[12], 64, v[12]))   # +2 rows = +32 elems = +64 bytes
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  return I

def test_tile():
  dev = Device[Device.DEFAULT]
  print("arch:", getattr(dev.renderer, 'arch', '?'))
  insts = build_tile_kernel()
  rng = np.random.default_rng(0)
  a_np = rng.standard_normal((16,16)).astype(np.float16)
  b_np = rng.standard_normal((16,16)).astype(np.float16)
  bt_np = np.ascontiguousarray(b_np.T)   # store B transposed so columns are contiguous
  a = Tensor(a_np); bt = Tensor(bt_np); c = Tensor.empty(16,16, dtype=dtypes.half)
  Tensor.realize(a, bt, c)
  def asm_kernel(A, B, C):
    lidxs = [UOp.special(32, "lidx0")]
    sink = UOp.sink(A.base, B.base, C.base, UOp.special(1,"gidx0"), *lidxs,
                    arg=KernelInfo(name=colored("rdna3_tile","cyan"), estimates=Estimates(ops=16*16*16*2, mem=16*16*2*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
               UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
  c = Tensor.custom_kernel(a, bt, c, fxn=asm_kernel)[2]
  linear = c.schedule_linear()
  with Context(DEBUG=2): run_linear(linear)
  c_np = c.float().numpy()
  ref = a_np.astype(np.float32) @ b_np.astype(np.float32)
  err = np.sqrt(np.mean((c_np - ref)**2)) / (np.sqrt(np.mean(ref**2)) + 1e-9)
  print(f"relative RMSE {err:.6f}")
  if err < 0.05: print("TILE CORRECT")
  else:
    print("WRONG. asm[0,:6]:", c_np[0,:6]); print("ref  [0,:6]:", ref[0,:6])
    print("asm[:6,0]:", c_np[:6,0]); print("ref [:6,0]:", ref[:6,0])

def build_gemm(M, N, K, TM, TN):
  # workgroup (1 wave32) computes a (TM*16)x(TN*16) output tile. grid=(N//(TN*16), M//(TM*16),1).
  # s[2]=gx (col-block), s[3]=gy (row-block). A: MxK row-major. Bt: NxK row-major (B transposed).
  assert M%(TM*16)==0 and N%(TN*16)==0 and K%16==0
  FAb = 10; FBb = FAb+TM*8; VA = FBb+TN*8; ACCb = VA+(TM+TN)   # sequential, no overlap
  assert ACCb+TM*TN*8 <= 256, f"VGPR overflow: {ACCb+TM*TN*8}"
  I=[]; B=[]; lbl={}
  def e(i): I.append(i); return i
  def label(n): lbl[n]=sum(i.size() for i in I)
  def br(t): B.append((len(I)-1,t))
  sh = {4:6, 2:5, 1:4}
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  e(v_and_b32_e32(v[1], 15, v[0]))
  e(s_lshl_b32(s[10], s[3], sh[TM]))               # gy*TM*16
  e(s_lshl_b32(s[11], s[2], sh[TN]))               # gx*TN*16
  e(v_add_nc_u32_e32(v[2], s[10], v[1]))
  e(v_add_nc_u32_e32(v[3], s[11], v[1]))
  for tm in range(TM):
    e(v_add_nc_u32_e32(v[VA+tm], tm*16, v[2]) if tm else v_mov_b32_e32(v[VA+tm], v[2]))
    e(v_mul_lo_u32(v[VA+tm], v[VA+tm], K*2))
  for tn in range(TN):
    e(v_add_nc_u32_e32(v[VA+TM+tn], tn*16, v[3]) if tn else v_mov_b32_e32(v[VA+TM+tn], v[3]))
    e(v_mul_lo_u32(v[VA+TM+tn], v[VA+TM+tn], K*2))
  for i in range(TM*TN*8): e(v_mov_b32_e32(v[ACCb+i], 0))
  e(s_mov_b32(s[16], 0))
  label('LOOP')
  for tm in range(TM):
    e(global_load_b128(vdst=v[FAb+tm*8:FAb+tm*8+3],   addr=v[VA+tm:VA+tm], saddr=s[4:5], offset=0))
    e(global_load_b128(vdst=v[FAb+tm*8+4:FAb+tm*8+7], addr=v[VA+tm:VA+tm], saddr=s[4:5], offset=16))
  for tn in range(TN):
    e(global_load_b128(vdst=v[FBb+tn*8:FBb+tn*8+3],   addr=v[VA+TM+tn:VA+TM+tn], saddr=s[6:7], offset=0))
    e(global_load_b128(vdst=v[FBb+tn*8+4:FBb+tn*8+7], addr=v[VA+TM+tn:VA+TM+tn], saddr=s[6:7], offset=16))
  e(s_waitcnt(simm16=0))
  for tm in range(TM):
    for tn in range(TN):
      ac=ACCb+(tm*TN+tn)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FAb+tm*8:FAb+tm*8+7], src1=v[FBb+tn*8:FBb+tn*8+7], src2=v[ac:ac+7]))
  for r in range(TM+TN): e(v_add_nc_u32_e32(v[VA+r], 32, v[VA+r]))
  e(s_add_i32(s[16], s[16], 1)); e(s_cmp_lt_i32(s[16], K//16)); e(s_cbranch_scc1(simm16=0)); br('LOOP')
  e(v_and_b32_e32(v[4], 15, v[0]))
  e(v_lshrrev_b32_e32(v[5], 4, v[0])); e(v_and_b32_e32(v[5], 1, v[5]))
  for tm in range(TM):
    for tn in range(TN):
      ac=ACCb+(tm*TN+tn)*8
      e(v_add_nc_u32_e32(v[7], s[10], v[5]))
      e(v_add_nc_u32_e32(v[7], tm*16, v[7]))
      e(v_add_nc_u32_e32(v[8], s[11], v[4]))
      e(v_add_nc_u32_e32(v[8], tn*16, v[8]))
      e(v_mul_lo_u32(v[7], v[7], N)); e(v_add_nc_u32_e32(v[7], v[7], v[8]))
      e(v_lshlrev_b32_e32(v[7], 1, v[7]))
      for i in range(8):
        e(v_cvt_f16_f32_e32(v[6], v[ac+i]))
        e(global_store_b16(addr=v[7:7], data=v[6], saddr=s[8:9], offset=0))
        if i<7: e(v_add_nc_u32_e32(v[7], N*4, v[7]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  for idx,t in B:
    off=(lbl[t]-sum(i.size() for i in I[:idx+1]))//4
    assert -32768<=off<=32767; I[idx].simm16=off
  return I

def test_gemm():
  M=N=K=getenv("N",2048); TM,TN=getenv("TM",4),getenv("TN",4)
  insts=build_gemm(M,N,K,TM,TN)
  rng=np.random.default_rng(1)
  a_np=(rng.standard_normal((M,K))*0.1).astype(np.float16)
  bt_np=(rng.standard_normal((N,K))*0.1).astype(np.float16)
  a=Tensor(a_np); bt=Tensor(bt_np); c=Tensor.empty(M,N,dtype=dtypes.half); Tensor.realize(a,bt,c)
  grid=(N//(TN*16), M//(TM*16), 1)
  def asm_kernel(A,Bt,C):
    g=[UOp.special(grid[0],"gidx0"), UOp.special(grid[1],"gidx1")]
    sink=UOp.sink(A.base,Bt.base,C.base,*g,UOp.special(32,"lidx0"),
                  arg=KernelInfo(name=colored("rdna3_gemm","cyan"),estimates=Estimates(ops=M*N*K*2,mem=(M*K+N*K+M*N)*2)))
    return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.DEVICE,arg=Device.DEFAULT),UOp(Ops.LINEAR,src=tuple([UOp(Ops.INS,arg=x) for x in insts]))))
  c=Tensor.custom_kernel(a,bt,c,fxn=asm_kernel)[2]; linear=c.schedule_linear()
  ets=[]
  with Context(DEBUG=2):
    for _ in range(getenv("CNT",8)):
      st=GlobalCounters.time_sum_s; run_linear(linear); ets.append(GlobalCounters.time_sum_s-st)
  print(f"REAL TFLOPS {M*N*K*2/min(ets)*1e-12:.2f}  (TM={TM} TN={TN} N={N})")
  c_np=c.float().numpy(); ref=a_np.astype(np.float32)@bt_np.astype(np.float32).T
  err=np.sqrt(np.mean((c_np-ref)**2))/(np.sqrt(np.mean(ref**2))+1e-9)
  print(f"relative RMSE {err:.6f}  {'CORRECT' if err<0.05 else 'WRONG'}")

if __name__ == "__main__":
  if getenv("GEMM",1): test_gemm()
  else: test_tile()
