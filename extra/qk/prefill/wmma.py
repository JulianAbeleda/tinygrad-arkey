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

def waitcnt_lgkm(n):
  # DS/LDS wait: lgkmcnt=bits[9:4] (per extra/qk/prefill/wmma.py). vmcnt/expcnt maxed (don't wait on them).
  return s_waitcnt(simm16=(0x7) | ((n & 0x3F) << 4) | (0x3F << 10))

def build_lds_tile():
  # P0 (A3): single 16x16x16 tile round-tripped through LDS. Proves RDNA3 LDS plumbing end-to-end:
  # DEFINE_LOCAL alloc + ds_store + s_barrier + lgkmcnt waits + ds_load -> WMMA. Each lane writes/reads its
  # own A-row and Bt-col (no real cross-lane share yet — just the plumbing).
  FA, FB, ACC = 20, 32, 44
  LDS_A, LDS_B = 0, 16*16*2   # B region after A (512 bytes)
  I=[]
  def e(i): I.append(i); return i
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  e(v_and_b32_e32(v[2], 15, v[0]))               # lane&15
  e(v_lshlrev_b32_e32(v[2], 5, v[2]))            # *32 = per-lane byte offset (row of 16 fp16)
  e(v_mov_b32_e32(v[3], 0))
  for i in range(8): e(v_mov_b32_e32(v[ACC+i], 0))
  # global load A row + Bt col (saddr + v[2:3])
  e(global_load_b128(vdst=v[FA:FA+3],   addr=v[2:2], saddr=s[4:5], offset=0))
  e(global_load_b128(vdst=v[FA+4:FA+7], addr=v[2:2], saddr=s[4:5], offset=16))
  e(global_load_b128(vdst=v[FB:FB+3],   addr=v[2:2], saddr=s[6:7], offset=0))
  e(global_load_b128(vdst=v[FB+4:FB+7], addr=v[2:2], saddr=s[6:7], offset=16))
  e(waitcnt_vm(0))                               # global loads landed
  # LDS addresses: A at lane*32 (+LDS_A), B at lane*32 (+LDS_B)
  e(v_mov_b32_e32(v[16], v[2]))
  e(v_add_nc_u32_e32(v[17], LDS_B, v[2]))
  e(ds_store_b128(addr=v[16], data0=v[FA:FA+3],   offset0=0,  offset1=0))
  e(ds_store_b128(addr=v[16], data0=v[FA+4:FA+7], offset0=16, offset1=0))
  e(ds_store_b128(addr=v[17], data0=v[FB:FB+3],   offset0=0,  offset1=0))
  e(ds_store_b128(addr=v[17], data0=v[FB+4:FB+7], offset0=16, offset1=0))
  e(waitcnt_lgkm(0))                             # ds_store done
  e(s_barrier())
  e(ds_load_b128(vdst=v[FA:FA+3],   addr=v[16], offset0=0,  offset1=0))
  e(ds_load_b128(vdst=v[FA+4:FA+7], addr=v[16], offset0=16, offset1=0))
  e(ds_load_b128(vdst=v[FB:FB+3],   addr=v[17], offset0=0,  offset1=0))
  e(ds_load_b128(vdst=v[FB+4:FB+7], addr=v[17], offset0=16, offset1=0))
  e(waitcnt_lgkm(0))                             # ds_load landed
  e(v_wmma_f32_16x16x16_f16(vdst=v[ACC:ACC+7], src0=v[FA:FA+7], src1=v[FB:FB+7], src2=v[ACC:ACC+7]))
  # store C (same layout as build_tile_kernel)
  e(v_and_b32_e32(v[10], 15, v[0]))
  e(v_lshrrev_b32_e32(v[11], 4, v[0])); e(v_and_b32_e32(v[11], 1, v[11]))
  e(v_lshlrev_b32_e32(v[12], 4, v[11]))
  e(v_add_nc_u32_e32(v[12], v[12], v[10]))
  e(v_lshlrev_b32_e32(v[12], 1, v[12]))
  e(v_mov_b32_e32(v[13], 0))
  for i in range(8):
    e(v_cvt_f16_f32_e32(v[14], v[ACC+i]))
    e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
    if i < 7: e(v_add_nc_u32_e32(v[12], 64, v[12]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  return I

def test_lds_tile():
  insts=build_lds_tile()
  rng=np.random.default_rng(0)
  a_np=rng.standard_normal((16,16)).astype(np.float16); b_np=rng.standard_normal((16,16)).astype(np.float16)
  bt_np=np.ascontiguousarray(b_np.T)
  a=Tensor(a_np); bt=Tensor(bt_np); c=Tensor.empty(16,16,dtype=dtypes.half); Tensor.realize(a,bt,c)
  def asm_kernel(A,B,C):
    lds=UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=2048, addrspace=AddrSpace.LOCAL), (), 'lds')
    sink=UOp.sink(A.base,B.base,C.base,lds,UOp.special(1,"gidx0"),UOp.special(32,"lidx0"),
                  arg=KernelInfo(name=colored("rdna3_lds_tile","cyan"),estimates=Estimates(ops=16*16*16*2,mem=16*16*2*3)))
    return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.DEVICE,arg=Device.DEFAULT),UOp(Ops.LINEAR,src=tuple([UOp(Ops.INS,arg=x) for x in insts]))))
  c=Tensor.custom_kernel(a,bt,c,fxn=asm_kernel)[2]; linear=c.schedule_linear()
  with Context(DEBUG=2): run_linear(linear)
  c_np=c.float().numpy(); ref=a_np.astype(np.float32)@b_np.astype(np.float32)
  err=np.sqrt(np.mean((c_np-ref)**2))/(np.sqrt(np.mean(ref**2))+1e-9)
  print(f"relative RMSE {err:.6f}  {'LDS TILE CORRECT' if err<0.05 else 'WRONG'}")
  if err>=0.05:
    print("asm[0,:6]:", c_np[0,:6]); print("ref [0,:6]:", ref[0,:6])

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

def waitcnt_vm(n):
  # s_waitcnt simm16 (matches the proven in-repo encoder, extra/qk/prefill/wmma.py):
  #   expcnt=bits[2:0], lgkmcnt=bits[9:4], vmcnt=bits[15:10].
  # Wait until <=n outstanding VMEM loads; leave expcnt/lgkmcnt maxed (don't wait on them).
  if getenv("FULLWAIT",0): return s_waitcnt(simm16=0)
  return s_waitcnt(simm16=(0x7) | ((0x3F) << 4) | ((n & 0x3F) << 10))

def build_gemm_pipe(M, N, K, TM, TN):
  # Double-buffered software-pipelined GEMM (A2). Unroll-by-2: F0 holds even-k frags, F1 holds odd-k.
  # Prefetch next-k loads while WMMAs on the current buffer run; targeted s_waitcnt(vmcnt) instead of full barrier.
  # 1 wave32/workgroup computes a (TM*16)x(TN*16) tile. A: MxK row-major. Bt: NxK row-major (B transposed).
  assert M%(TM*16)==0 and N%(TN*16)==0 and K%32==0, "K must be multiple of 32 (unroll-by-2)"
  NK = K//16; assert NK>=4, "need >=4 k-tiles"
  LOOPS = NK//2 - 1
  LPB = TM*2 + TN*2                              # b128 loads per buffer (each frag = 2x b128)
  F0A=10; F0B=F0A+TM*8; F1A=F0B+TN*8; F1B=F1A+TM*8; VA=F1B+TN*8; ACCb=VA+(TM+TN)
  assert ACCb+TM*TN*8 <= 256, f"VGPR overflow: {ACCb+TM*TN*8}"
  I=[]; Br=[]; lbl={}
  def e(i): I.append(i); return i
  def label(n): lbl[n]=sum(i.size() for i in I)
  def br(t): Br.append((len(I)-1,t))
  sh = {4:6, 2:5, 1:4}
  def issue_loads(Ab, Bb):                       # load current-k frags into buffers, advance addrs by one k-tile
    for tm in range(TM):
      e(global_load_b128(vdst=v[Ab+tm*8:Ab+tm*8+3],   addr=v[VA+tm:VA+tm], saddr=s[4:5], offset=0))
      e(global_load_b128(vdst=v[Ab+tm*8+4:Ab+tm*8+7], addr=v[VA+tm:VA+tm], saddr=s[4:5], offset=16))
    for tn in range(TN):
      e(global_load_b128(vdst=v[Bb+tn*8:Bb+tn*8+3],   addr=v[VA+TM+tn:VA+TM+tn], saddr=s[6:7], offset=0))
      e(global_load_b128(vdst=v[Bb+tn*8+4:Bb+tn*8+7], addr=v[VA+TM+tn:VA+TM+tn], saddr=s[6:7], offset=16))
    for r in range(TM+TN): e(v_add_nc_u32_e32(v[VA+r], 32, v[VA+r]))
  def do_wmmas(Ab, Bb):
    for tm in range(TM):
      for tn in range(TN):
        ac=ACCb+(tm*TN+tn)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[Ab+tm*8:Ab+tm*8+7], src1=v[Bb+tn*8:Bb+tn*8+7], src2=v[ac:ac+7]))
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  e(v_and_b32_e32(v[1], 15, v[0]))
  e(s_lshl_b32(s[10], s[3], sh[TM]))
  e(s_lshl_b32(s[11], s[2], sh[TN]))
  e(v_add_nc_u32_e32(v[2], s[10], v[1]))
  e(v_add_nc_u32_e32(v[3], s[11], v[1]))
  for tm in range(TM):
    e(v_add_nc_u32_e32(v[VA+tm], tm*16, v[2]) if tm else v_mov_b32_e32(v[VA+tm], v[2]))
    e(v_mul_lo_u32(v[VA+tm], v[VA+tm], K*2))
  for tn in range(TN):
    e(v_add_nc_u32_e32(v[VA+TM+tn], tn*16, v[3]) if tn else v_mov_b32_e32(v[VA+TM+tn], v[3]))
    e(v_mul_lo_u32(v[VA+TM+tn], v[VA+TM+tn], K*2))
  for i in range(TM*TN*8): e(v_mov_b32_e32(v[ACCb+i], 0))
  issue_loads(F0A, F0B)                          # k=0 -> F0
  e(s_mov_b32(s[16], 0))
  label('LOOP')                                  # invariant: F0 holds k=2j (in flight/done)
  issue_loads(F1A, F1B)                          # k=2j+1 -> F1
  e(waitcnt_vm(LPB)); do_wmmas(F0A, F0B)         # F0 ready (only F1's LPB outstanding)
  issue_loads(F0A, F0B)                          # prefetch k=2j+2 -> F0
  e(waitcnt_vm(LPB)); do_wmmas(F1A, F1B)         # F1 ready
  e(s_add_i32(s[16], s[16], 1)); e(s_cmp_lt_i32(s[16], LOOPS)); e(s_cbranch_scc1(simm16=0)); br('LOOP')
  issue_loads(F1A, F1B)                          # tail: k=NK-1 -> F1 (F0 already holds k=NK-2)
  e(waitcnt_vm(LPB)); do_wmmas(F0A, F0B)
  e(s_waitcnt(simm16=0)); do_wmmas(F1A, F1B)
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
  for idx,t in Br:
    off=(lbl[t]-sum(i.size() for i in I[:idx+1]))//4
    assert -32768<=off<=32767; I[idx].simm16=off
  return I

def _run_insts(insts, a, bt, c, M, N, K, TM, TN, name):
  grid=(N//(TN*16), M//(TM*16), 1)
  def asm_kernel(A,Bt,C):
    g=[UOp.special(grid[0],"gidx0"), UOp.special(grid[1],"gidx1")]
    sink=UOp.sink(A.base,Bt.base,C.base,*g,UOp.special(32,"lidx0"),
                  arg=KernelInfo(name=colored(name,"cyan"),estimates=Estimates(ops=M*N*K*2,mem=(M*K+N*K+M*N)*2)))
    return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.DEVICE,arg=Device.DEFAULT),UOp(Ops.LINEAR,src=tuple([UOp(Ops.INS,arg=x) for x in insts]))))
  out=Tensor.custom_kernel(a,bt,c,fxn=asm_kernel)[2]
  return out.schedule_linear(), out

def build_gemm_lds(M, N, K):
  # P1 (A3): LDS-staged, multi-wave. Workgroup=128 threads=4 wave32 as 2x2; each wave computes a 64x64 sub-tile
  # (4x4 WMMA tiles). Cooperative global->LDS load of a 128x16 A-slice + 128x16 Bt-slice each K-iter, full
  # s_barrier, single-buffer LDS. BM=BN=128, BK=16. A: MxK row-major; Bt: NxK row-major (B transposed).
  BM=BN=128; BK=16; WM=WN=4; THREADS=128
  assert M%BM==0 and N%BN==0 and K%BK==0
  NK=K//BK
  FA=10; FB=FA+WM*8; ACCb=FB+WN*8                 # 10, 42, 74; ACC=16*8=128 -> 74..201
  TA=ACCb+WM*WN*8                                  # cooperative-load temps: TA(A row 8), TA+8(Bt row 8)
  assert TA+16 <= 256, f"VGPR overflow {TA+16}"
  LDS_A=0; LDS_B=BM*BK*2                           # 4096; total 8192 bytes
  I=[]; Br=[]; lbl={}
  def e(i): I.append(i); return i
  def label(n): lbl[n]=sum(i.size() for i in I)
  def br(t): Br.append((len(I)-1,t))
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  # lanepart=(tid&15)*32 ; wave=tid>>5 ; wave_m=wave>>1 ; wave_n=wave&1.
  # wave_m/wave_n must survive the K-loop (read in epilogue) -> hold in v[218]/v[219], ABOVE all frag/acc/temp ranges.
  WMR, WNR = 218, 219
  e(v_and_b32_e32(v[1], 15, v[0])); e(v_lshlrev_b32_e32(v[1], 5, v[1]))     # v[1]=lanepart
  e(v_lshrrev_b32_e32(v[8], 5, v[0]))                                       # v[8]=wave (scratch)
  e(v_lshrrev_b32_e32(v[WMR], 1, v[8])); e(v_and_b32_e32(v[WMR], 1, v[WMR]))# wave_m
  e(v_and_b32_e32(v[WNR], 1, v[8]))                                         # wave_n
  # frag LDS read bases: vAfrag=wave_m*2048+lanepart ; vBfrag=LDS_B+wave_n*2048+lanepart
  e(v_lshlrev_b32_e32(v[6], 11, v[WMR])); e(v_add_nc_u32_e32(v[6], v[6], v[1]))
  e(v_lshlrev_b32_e32(v[7], 11, v[WNR])); e(v_add_nc_u32_e32(v[7], v[7], v[1])); e(v_add_nc_u32_e32(v[7], LDS_B, v[7]))
  # coop-load LDS store addr = tid*32 ; global byte base = (gy*BM + tid)*K*2  (gy=s[3], gx=s[2])
  e(v_lshlrev_b32_e32(v[4], 5, v[0]))                                       # vA_lds=tid*32
  e(v_add_nc_u32_e32(v[5], LDS_B, v[4]))                                    # vB_lds=tid*32+LDS_B
  if getenv("ZEROGID",0): e(s_mov_b32(s[10], 0)); e(s_mov_b32(s[11], 0))
  else: e(s_lshl_b32(s[10], s[3], 7)); e(s_lshl_b32(s[11], s[2], 7))        # gy*BM, gx*BN
  e(v_add_nc_u32_e32(v[2], s[10], v[0])); e(v_mul_lo_u32(v[2], v[2], K*2))  # vA_glob
  e(v_add_nc_u32_e32(v[3], s[11], v[0])); e(v_mul_lo_u32(v[3], v[3], K*2))  # vB_glob
  for i in range(WM*WN*8): e(v_mov_b32_e32(v[ACCb+i], 0))
  e(s_mov_b32(s[16], 0))
  label('LOOP')
  # cooperative global -> regs -> LDS
  e(global_load_b128(vdst=v[TA:TA+3],    addr=v[2:2], saddr=s[4:5], offset=0))
  e(global_load_b128(vdst=v[TA+4:TA+7],  addr=v[2:2], saddr=s[4:5], offset=16))
  e(global_load_b128(vdst=v[TA+8:TA+11], addr=v[3:3], saddr=s[6:7], offset=0))
  e(global_load_b128(vdst=v[TA+12:TA+15],addr=v[3:3], saddr=s[6:7], offset=16))
  e(waitcnt_vm(0))
  e(ds_store_b128(addr=v[4], data0=v[TA:TA+3],     offset0=0,  offset1=0))
  e(ds_store_b128(addr=v[4], data0=v[TA+4:TA+7],   offset0=16, offset1=0))
  e(ds_store_b128(addr=v[5], data0=v[TA+8:TA+11],  offset0=0,  offset1=0))
  e(ds_store_b128(addr=v[5], data0=v[TA+12:TA+15], offset0=16, offset1=0))
  e(waitcnt_lgkm(0)); e(s_barrier())
  # load fragments from LDS (offset for tile i = i*512 bytes; +16 for second b128 half)
  for mi in range(WM):
    o=mi*512
    e(ds_load_b128(vdst=v[FA+mi*8:FA+mi*8+3],   addr=v[6], offset0=o&0xFF,      offset1=o>>8))
    e(ds_load_b128(vdst=v[FA+mi*8+4:FA+mi*8+7], addr=v[6], offset0=(o+16)&0xFF, offset1=(o+16)>>8))
  for ni in range(WN):
    o=ni*512
    e(ds_load_b128(vdst=v[FB+ni*8:FB+ni*8+3],   addr=v[7], offset0=o&0xFF,      offset1=o>>8))
    e(ds_load_b128(vdst=v[FB+ni*8+4:FB+ni*8+7], addr=v[7], offset0=(o+16)&0xFF, offset1=(o+16)>>8))
  e(waitcnt_lgkm(0))
  for mi in range(WM):
    for ni in range(WN):
      ac=ACCb+(mi*WN+ni)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+mi*8:FA+mi*8+7], src1=v[FB+ni*8:FB+ni*8+7], src2=v[ac:ac+7]))
  e(s_barrier())                                  # before next iter overwrites LDS
  e(v_add_nc_u32_e32(v[2], BK*2, v[2])); e(v_add_nc_u32_e32(v[3], BK*2, v[3]))
  e(s_add_i32(s[16], s[16], 1)); e(s_cmp_lt_i32(s[16], NK)); e(s_cbranch_scc1(simm16=0)); br('LOOP')
  # epilogue: global row = gy*BM + wave_m*64 + mi*16 + (i*2+parity); col = gx*BN + wave_n*64 + ni*16 + (lane&15)
  e(v_and_b32_e32(v[8], 15, v[0]))                                          # lane&15 (col within tile)
  e(v_lshrrev_b32_e32(v[9], 4, v[0])); e(v_and_b32_e32(v[9], 1, v[9]))      # parity
  e(v_lshlrev_b32_e32(v[21], 6, v[WMR])); e(v_add_nc_u32_e32(v[21], s[10], v[21]))  # row base = gy*BM + wave_m*64
  e(v_lshlrev_b32_e32(v[22], 6, v[WNR])); e(v_add_nc_u32_e32(v[22], s[11], v[22]))  # col base = gx*BN + wave_n*64
  if getenv("NOSTORE",0):
    e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
    for idx,t in Br:
      off=(lbl[t]-sum(i.size() for i in I[:idx+1]))//4
      assert -32768<=off<=32767; I[idx].simm16=off
    return I
  for mi in range(WM):
    for ni in range(WN):
      ac=ACCb+(mi*WN+ni)*8
      e(v_add_nc_u32_e32(v[12], v[21], v[9]))            # rowbase + parity
      e(v_add_nc_u32_e32(v[12], mi*16, v[12]))           # + mi*16
      e(v_add_nc_u32_e32(v[13], v[22], v[8]))            # colbase + (lane&15)
      e(v_add_nc_u32_e32(v[13], ni*16, v[13]))           # + ni*16
      e(v_mul_lo_u32(v[12], v[12], N)); e(v_add_nc_u32_e32(v[12], v[12], v[13]))
      e(v_lshlrev_b32_e32(v[12], 1, v[12]))
      for i in range(8):
        e(v_cvt_f16_f32_e32(v[14], v[ac+i]))
        e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
        if i<7: e(v_add_nc_u32_e32(v[12], N*4, v[12]))   # +2 rows
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  for idx,t in Br:
    off=(lbl[t]-sum(i.size() for i in I[:idx+1]))//4
    assert -32768<=off<=32767; I[idx].simm16=off
  return I

def build_gemm_lds2(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA=0, PLRAB=0, LEANADDR=0, DSHALF=0):
  # P2/P3 (A3): parametric LDS-staged multi-wave GEMM. WAVES_M x WAVES_N wave32; each wave does WM x WN WMMA
  # tiles. BK = K-block depth (KT=BK/16 substeps). PAD = LDS row-pad bytes (bank-conflict avoidance). DBUF =
  # double-buffer LDS via unroll-by-2 (prefetch next block while computing current; removes the inner barrier).
  # PLRA = intra-block A-prefetch local read (KT==2 single-buffer only): prefetch substep1's A fragments into
  # the DEAD coop-load temp regs (CTA/CTB, register-lifetime overlap a la Tensile's pool) while substep0's
  # WMMAs run -> hides substep1's A ds_load latency behind compute. Partial PLR (A only; B' wouldn't fit 256).
  THREADS=WAVES_M*WAVES_N*32; BM=WAVES_M*WM*16; BN=WAVES_N*WN*16; KT=BK//16; CPR=BK//8; RSTRIDE=THREADS//CPR
  assert M%BM==0 and N%BN==0 and K%BK==0 and THREADS%CPR==0 and BM%RSTRIDE==0 and BN%RSTRIDE==0
  loadsA=BM//RSTRIDE; loadsB=BN//RSTRIDE; NBLK=K//BK
  SA=BK*2+PAD; SB=BK*2+PAD; LDS_A=SA*BM; BUFSZ=LDS_A+SB*BN; NBUF=2 if DBUF else 1
  FA=10; FB=FA+WM*8; ACCb=FB+WN*8; CTA=ACCb+WM*WN*8; CTB=CTA+loadsA*4; SCR=CTB+loadsB*4
  FB2=SCR+2                                          # 2nd fragment buffer for full A+B PLR (PLRAB), past everything
  assert SCR+2<=256, f"VGPR overflow {SCR+2}"
  if PLRAB: assert FB2+WM*8+WN*8<=256, f"PLRAB VGPR overflow {FB2+WM*8+WN*8} (needs smaller tile than {WM}x{WN})"
  assert BUFSZ*NBUF<=65536, f"LDS overflow {BUFSZ*NBUF}"
  I=[]; Br=[]; lbl={}
  def e(i): I.append(i); return i
  def label(n): lbl[n]=sum(i.size() for i in I)
  def br(t): Br.append((len(I)-1,t))
  def dsoff(o): return dict(offset0=o&0xFF, offset1=(o>>8)&0xFF)
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  # wave_m=(tid>>5)>>log2(WAVES_N) ... waves laid out row-major: wave=tid>>5; wave_m=wave//WAVES_N; wave_n=wave%WAVES_N
  e(v_lshrrev_b32_e32(v[8], 5, v[0]))                                       # wave
  if WAVES_N==1: e(v_mov_b32_e32(v[19], v[8])); e(v_mov_b32_e32(v[20], 0))
  elif WAVES_N==2: e(v_lshrrev_b32_e32(v[19],1,v[8])); e(v_and_b32_e32(v[20],1,v[8]))
  elif WAVES_N==4: e(v_lshrrev_b32_e32(v[19],2,v[8])); e(v_and_b32_e32(v[20],3,v[8]))
  else: raise AssertionError("WAVES_N in {1,2,4}")
  # frag read bases: vAfrag=(wave_m*WM*16 + (tid&15))*SA ; vBfrag=LDS_A + (wave_n*WN*16 + (tid&15))*SB
  e(v_and_b32_e32(v[1], 15, v[0]))                                          # tid&15
  e(v_lshlrev_b32_e32(v[6], 4, v[19]))                                      # wave_m*16  (* WM below via mul)
  e(v_mul_lo_u32(v[6], v[6], WM)); e(v_add_nc_u32_e32(v[6], v[6], v[1])); e(v_mul_lo_u32(v[6], v[6], SA))
  e(v_lshlrev_b32_e32(v[7], 4, v[20])); e(v_mul_lo_u32(v[7], v[7], WN)); e(v_add_nc_u32_e32(v[7], v[7], v[1]))
  e(v_mul_lo_u32(v[7], v[7], SB))                                          # raw (LDS_A added in compute offset)
  # coop-load: thread tid -> chunk=tid%CPR, row0=tid//CPR; loads rows row0 + j*RSTRIDE (j<loadsX), chunk fixed.
  lg2=CPR.bit_length()-1
  e(v_and_b32_e32(v[10], CPR-1, v[0])); e(v_lshrrev_b32_e32(v[11], lg2, v[0]))   # chunk, row0  (temp, pre-loop)
  e(s_lshl_b32(s[10], s[3], BM.bit_length()-1)); e(s_lshl_b32(s[11], s[2], BN.bit_length()-1))  # gy*BM, gx*BN
  # vA_glob = (gy*BM + row0)*K*2 + chunk*16 ; vA_lds = row0*SA + chunk*16
  e(v_add_nc_u32_e32(v[2], s[10], v[11])); e(v_mul_lo_u32(v[2], v[2], K*2)); e(v_lshlrev_b32_e32(v[12],4,v[10])); e(v_add_nc_u32_e32(v[2], v[2], v[12]))
  e(v_mul_lo_u32(v[4], v[11], SA)); e(v_add_nc_u32_e32(v[4], v[4], v[12]))
  e(v_add_nc_u32_e32(v[3], s[11], v[11])); e(v_mul_lo_u32(v[3], v[3], K*2)); e(v_add_nc_u32_e32(v[3], v[3], v[12]))
  e(v_mul_lo_u32(v[5], v[11], SB)); e(v_add_nc_u32_e32(v[5], v[5], v[12]))
  for i in range(WM*WN*8): e(v_mov_b32_e32(v[ACCb+i], 0))
  # ---- LEANADDR (Lever A): move per-iter coop-load address arith from VALU -> SALU. Precompute the INVARIANT
  # per-lane row byte-offsets (one vgpr per cooperative-load row), and advance the K-position by incrementing
  # SCALAR buffer base pointers (s_add) instead of recomputing vector addresses (v_add) every K-block. ----
  VRA=SCR+1; VRB=VRA+loadsA; SKA=18; SKB=20         # invariant row vgprs (A then B); scalar k-base sgpr pairs
  if LEANADDR:
    assert VRB+loadsB<=256, f"LEANADDR VGPR overflow {VRB+loadsB}"
    for j in range(loadsA): e(v_mov_b32_e32(v[VRA+j], v[2]) if j==0 else v_add_nc_u32_e32(v[VRA+j], j*RSTRIDE*K*2, v[2]))
    for j in range(loadsB): e(v_mov_b32_e32(v[VRB+j], v[3]) if j==0 else v_add_nc_u32_e32(v[VRB+j], j*RSTRIDE*K*2, v[3]))
    e(s_mov_b32(s[SKA], s[4])); e(s_mov_b32(s[SKA+1], s[5])); e(s_mov_b32(s[SKB], s[6])); e(s_mov_b32(s[SKB+1], s[7]))
  def coop_load_lean(buf):                          # addr=invariant row vgpr, saddr=advancing scalar k-base
    for j in range(loadsA): e(global_load_b128(vdst=v[CTA+j*4:CTA+j*4+3], addr=v[VRA+j:VRA+j], saddr=s[SKA:SKA+1], offset=0))
    for j in range(loadsB): e(global_load_b128(vdst=v[CTB+j*4:CTB+j*4+3], addr=v[VRB+j:VRB+j], saddr=s[SKB:SKB+1], offset=0))
  def adv_kbase():                                  # SALU K-advance (replaces the v2/v3 VALU advance)
    e(s_add_u32(s[SKA], s[SKA], BK*2)); e(s_addc_u32(s[SKA+1], s[SKA+1], 0))
    e(s_add_u32(s[SKB], s[SKB], BK*2)); e(s_addc_u32(s[SKB+1], s[SKB+1], 0))
  def coop_load(buf):                              # global -> CT regs (vmcnt drains at caller)
    if LEANADDR: return coop_load_lean(buf)
    for j in range(loadsA):
      if j==0: ar=2
      else: e(v_add_nc_u32_e32(v[SCR], j*RSTRIDE*K*2, v[2])); ar=SCR
      e(global_load_b128(vdst=v[CTA+j*4:CTA+j*4+3], addr=v[ar:ar], saddr=s[4:5], offset=0))
    for j in range(loadsB):
      if j==0: br_=3
      else: e(v_add_nc_u32_e32(v[SCR], j*RSTRIDE*K*2, v[3])); br_=SCR
      e(global_load_b128(vdst=v[CTB+j*4:CTB+j*4+3], addr=v[br_:br_], saddr=s[6:7], offset=0))
  def coop_store(buf):                             # CT regs -> LDS[buf]
    bo=buf*BUFSZ
    for j in range(loadsA): e(ds_store_b128(addr=v[4], data0=v[CTA+j*4:CTA+j*4+3], **dsoff(bo+j*RSTRIDE*SA)))
    for j in range(loadsB): e(ds_store_b128(addr=v[5], data0=v[CTB+j*4:CTB+j*4+3], **dsoff(bo+LDS_A+j*RSTRIDE*SB)))
  def compute(buf):                                # WMMAs from LDS[buf]
    bo=buf*BUFSZ
    for kt in range(KT):
      for mi in range(WM):
        o=bo+mi*16*SA+kt*32
        e(ds_load_b128(vdst=v[FA+mi*8:FA+mi*8+3],   addr=v[6], **dsoff(o)))
        if not DSHALF: e(ds_load_b128(vdst=v[FA+mi*8+4:FA+mi*8+7], addr=v[6], **dsoff(o+16)))  # DSHALF: drop 2nd half (INCORRECT; ds_load-count throughput probe)
      for ni in range(WN):
        o=bo+LDS_A+ni*16*SB+kt*32
        e(ds_load_b128(vdst=v[FB+ni*8:FB+ni*8+3],   addr=v[7], **dsoff(o)))
        if not DSHALF: e(ds_load_b128(vdst=v[FB+ni*8+4:FB+ni*8+7], addr=v[7], **dsoff(o+16)))
      e(waitcnt_lgkm(0))
      for mi in range(WM):
        for ni in range(WN):
          ac=ACCb+(mi*WN+ni)*8
          e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+mi*8:FA+mi*8+7], src1=v[FB+ni*8:FB+ni*8+7], src2=v[ac:ac+7]))
  def compute_plra(buf):                           # A-prefetch PLR: substep1 A loaded during substep0 WMMAs
    assert KT==2 and (loadsA*4+loadsB*4)>=WM*8, "PLRA needs KT==2 and dead CTA/CTB room for WM*8 A-frags"
    bo=buf*BUFSZ; FAp=CTA                           # FAp reuses the dead coop-load temp regs (CTA..CTB)
    def la(dst,kt):                                 # load this wave's WM A-fragments for substep kt
      for mi in range(WM):
        o=bo+mi*16*SA+kt*32
        e(ds_load_b128(vdst=v[dst+mi*8:dst+mi*8+3],   addr=v[6], **dsoff(o)))
        e(ds_load_b128(vdst=v[dst+mi*8+4:dst+mi*8+7], addr=v[6], **dsoff(o+16)))
    def lb(kt):
      for ni in range(WN):
        o=bo+LDS_A+ni*16*SB+kt*32
        e(ds_load_b128(vdst=v[FB+ni*8:FB+ni*8+3],   addr=v[7], **dsoff(o)))
        e(ds_load_b128(vdst=v[FB+ni*8+4:FB+ni*8+7], addr=v[7], **dsoff(o+16)))
    def ww(As):
      for mi in range(WM):
        for ni in range(WN):
          ac=ACCb+(mi*WN+ni)*8
          e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[As+mi*8:As+mi*8+7], src1=v[FB+ni*8:FB+ni*8+7], src2=v[ac:ac+7]))
    la(FA,0); lb(0); e(waitcnt_lgkm(0))            # substep0 A,B ready
    la(FAp,1)                                       # PREFETCH substep1 A (no wait) -> overlaps substep0 WMMAs
    ww(FA)                                          # substep0 WMMAs (FA read before lb(1) overwrites FB; safe WAR)
    lb(1); e(waitcnt_lgkm(0))                       # substep1 B + wait (FAp prefetch already done, FB1 now ready)
    ww(FAp)                                         # substep1 WMMAs from the prefetched A
  def compute_plrab(buf):                          # FULL A+B PLR: both substep1 operands prefetched (needs 2nd buf FB2)
    assert KT==2, "PLRAB needs KT==2"
    bo=buf*BUFSZ; FAp=FB2; FBp=FB2+WM*8             # 2nd fragment buffer (A' then B')
    def la(dst,kt):
      for mi in range(WM):
        o=bo+mi*16*SA+kt*32
        e(ds_load_b128(vdst=v[dst+mi*8:dst+mi*8+3],   addr=v[6], **dsoff(o)))
        e(ds_load_b128(vdst=v[dst+mi*8+4:dst+mi*8+7], addr=v[6], **dsoff(o+16)))
    def lb(dst,kt):
      for ni in range(WN):
        o=bo+LDS_A+ni*16*SB+kt*32
        e(ds_load_b128(vdst=v[dst+ni*8:dst+ni*8+3],   addr=v[7], **dsoff(o)))
        e(ds_load_b128(vdst=v[dst+ni*8+4:dst+ni*8+7], addr=v[7], **dsoff(o+16)))
    def ww(As,Bs):
      for mi in range(WM):
        for ni in range(WN):
          ac=ACCb+(mi*WN+ni)*8
          e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[As+mi*8:As+mi*8+7], src1=v[Bs+ni*8:Bs+ni*8+7], src2=v[ac:ac+7]))
    la(FA,0); lb(FB,0); e(waitcnt_lgkm(0))         # substep0 A,B
    la(FAp,1); lb(FBp,1)                            # PREFETCH substep1 A AND B -> overlap substep0 WMMAs
    ww(FA,FB)                                       # substep0 WMMAs (separate buffers: no WAR on substep1 prefetch)
    e(waitcnt_lgkm(0)); ww(FAp,FBp)                # substep1 ready (loads hidden) -> WMMAs
  comp = compute_plrab if PLRAB else compute_plra if PLRA else compute
  if not DBUF:
    e(s_mov_b32(s[16], 0))
    label('LOOP')
    coop_load(0); e(waitcnt_vm(0)); coop_store(0); e(waitcnt_lgkm(0)); e(s_barrier())
    comp(0); e(s_barrier())
    if LEANADDR: adv_kbase()                        # SALU K-advance (no VALU v2/v3 advance)
    else: e(v_add_nc_u32_e32(v[2], BK*2, v[2])); e(v_add_nc_u32_e32(v[3], BK*2, v[3]))
    e(s_add_i32(s[16], s[16], 1)); e(s_cmp_lt_i32(s[16], NBLK)); e(s_cbranch_scc1(simm16=0)); br('LOOP')
  else:
    # double-buffer, unroll-by-2: prefetch next block into the OTHER buffer while computing current.
    coop_load(0); e(waitcnt_vm(0)); coop_store(0); e(waitcnt_lgkm(0)); e(s_barrier())
    e(v_add_nc_u32_e32(v[2], BK*2, v[2])); e(v_add_nc_u32_e32(v[3], BK*2, v[3]))
    e(s_mov_b32(s[16], 0)); NL=(NBLK//2)-1
    label('LOOP')
    coop_load(1); compute(0); e(waitcnt_vm(0)); coop_store(1); e(waitcnt_lgkm(0)); e(s_barrier())
    e(v_add_nc_u32_e32(v[2], BK*2, v[2])); e(v_add_nc_u32_e32(v[3], BK*2, v[3]))
    coop_load(0); compute(1); e(waitcnt_vm(0)); coop_store(0); e(waitcnt_lgkm(0)); e(s_barrier())
    e(v_add_nc_u32_e32(v[2], BK*2, v[2])); e(v_add_nc_u32_e32(v[3], BK*2, v[3]))
    e(s_add_i32(s[16], s[16], 1)); e(s_cmp_lt_i32(s[16], NL)); e(s_cbranch_scc1(simm16=0)); br('LOOP')
    coop_load(1); compute(0); e(waitcnt_vm(0)); coop_store(1); e(waitcnt_lgkm(0)); e(s_barrier())
    compute(1)
  # epilogue (recompute wave_m/wave_n from tid — v[19]/v[20] were clobbered by the K-loop frag loads)
  e(v_and_b32_e32(v[8], 15, v[0])); e(v_lshrrev_b32_e32(v[9], 4, v[0])); e(v_and_b32_e32(v[9], 1, v[9]))
  e(v_lshrrev_b32_e32(v[10], 5, v[0]))                                      # wave
  if WAVES_N==1: e(v_mov_b32_e32(v[11], v[10])); e(v_mov_b32_e32(v[15], 0))
  elif WAVES_N==2: e(v_lshrrev_b32_e32(v[11],1,v[10])); e(v_and_b32_e32(v[15],1,v[10]))
  else: e(v_lshrrev_b32_e32(v[11],2,v[10])); e(v_and_b32_e32(v[15],3,v[10]))
  e(v_lshlrev_b32_e32(v[21], 4, v[11])); e(v_mul_lo_u32(v[21], v[21], WM)); e(v_add_nc_u32_e32(v[21], s[10], v[21]))  # gy*BM + wave_m*WM*16
  e(v_lshlrev_b32_e32(v[22], 4, v[15])); e(v_mul_lo_u32(v[22], v[22], WN)); e(v_add_nc_u32_e32(v[22], s[11], v[22]))  # gx*BN + wave_n*WN*16
  for mi in range(WM):
    for ni in range(WN):
      ac=ACCb+(mi*WN+ni)*8
      e(v_add_nc_u32_e32(v[12], v[21], v[9])); e(v_add_nc_u32_e32(v[12], mi*16, v[12]))
      e(v_add_nc_u32_e32(v[13], v[22], v[8])); e(v_add_nc_u32_e32(v[13], ni*16, v[13]))
      e(v_mul_lo_u32(v[12], v[12], N)); e(v_add_nc_u32_e32(v[12], v[12], v[13])); e(v_lshlrev_b32_e32(v[12], 1, v[12]))
      for i in range(8):
        e(v_cvt_f16_f32_e32(v[14], v[ac+i]))
        e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
        if i<7: e(v_add_nc_u32_e32(v[12], N*4, v[12]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  for idx,t in Br:
    off=(lbl[t]-sum(i.size() for i in I[:idx+1]))//4
    assert -32768<=off<=32767; I[idx].simm16=off
  return I

def _run_insts_lds(insts, a, bt, c, M, N, K, name, lds_bytes, BM=128, BN=128, THREADS=128):
  grid=(N//BN, M//BM, 1)
  def asm_kernel(A,Bt,C):
    lds=UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), 'lds')
    g=[UOp.special(grid[0],"gidx0"), UOp.special(grid[1],"gidx1")]
    sink=UOp.sink(A.base,Bt.base,C.base,lds,*g,UOp.special(THREADS,"lidx0"),
                  arg=KernelInfo(name=colored(name,"cyan"),estimates=Estimates(ops=M*N*K*2,mem=(M*K+N*K+M*N)*2)))
    return UOp(Ops.PROGRAM, src=(sink,UOp(Ops.DEVICE,arg=Device.DEFAULT),UOp(Ops.LINEAR,src=tuple([UOp(Ops.INS,arg=x) for x in insts]))))
  out=Tensor.custom_kernel(a,bt,c,fxn=asm_kernel)[2]
  return out.schedule_linear(), out

def test_lds_gemm():
  N=getenv("N",2048); M=getenv("M",N); K=getenv("K",N); CNT=getenv("CNT",20)
  insts=build_gemm_lds(M,N,K)
  rng=np.random.default_rng(1)
  a_np=(rng.standard_normal((M,K))*0.1).astype(np.float16); bt_np=(rng.standard_normal((N,K))*0.1).astype(np.float16)
  c=Tensor.empty(M,N,dtype=dtypes.half); Tensor.realize(c)
  linear,out=_run_insts_lds(insts,Tensor(a_np),Tensor(bt_np),c,M,N,K,"rdna3_lds_gemm",max(8192,65536//getenv("LIMIT_OCC",1)))
  ets=[]
  with Context(DEBUG=2):
    for _ in range(CNT):
      st=GlobalCounters.time_sum_s; run_linear(linear); ets.append(GlobalCounters.time_sum_s-st)
  print(f"REAL TFLOPS {M*N*K*2/min(ets)*1e-12:.2f}  (LDS multi-wave, M={M} K={K} N={N})")
  err=_rmse(out,a_np,bt_np)
  print(f"relative RMSE {err:.6f}  {'CORRECT' if err<0.05 else 'WRONG'}")

def test_lds_gemm2():
  N=getenv("N",2048); M=getenv("M",N); K=getenv("K",N); CNT=getenv("CNT",20)
  WAVES_M=getenv("WAVES_M",2); WAVES_N=getenv("WAVES_N",2); WM=getenv("WM",4); WN=getenv("WN",4)
  BK=getenv("BK",16); PAD=getenv("PAD",0); DBUF=getenv("DBUF",0)
  THREADS=WAVES_M*WAVES_N*32; BM=WAVES_M*WM*16; BN=WAVES_N*WN*16
  insts=build_gemm_lds2(M,N,K,WAVES_M,WAVES_N,WM,WN,BK,PAD,DBUF)
  rng=np.random.default_rng(1)
  a_np=(rng.standard_normal((M,K))*0.1).astype(np.float16); bt_np=(rng.standard_normal((N,K))*0.1).astype(np.float16)
  c=Tensor.empty(M,N,dtype=dtypes.half); Tensor.realize(c)
  ldsb=max((BK*2+PAD)*(BM+BN)*(2 if DBUF else 1), 65536//getenv("LIMIT_OCC",8))
  linear,out=_run_insts_lds(insts,Tensor(a_np),Tensor(bt_np),c,M,N,K,"rdna3_lds2",ldsb,BM,BN,THREADS)
  ets=[]
  with Context(DEBUG=2):
    for _ in range(CNT):
      st=GlobalCounters.time_sum_s; run_linear(linear); ets.append(GlobalCounters.time_sum_s-st)
  err=_rmse(out,a_np,bt_np)
  print(f"REAL TFLOPS {M*N*K*2/min(ets)*1e-12:6.2f}  W{WAVES_M}x{WAVES_N} T{WM}x{WN} BK{BK} PAD{PAD} DBUF{DBUF} (M={M} K={K} N={N})  RMSE {err:.6f} {'CORRECT' if err<0.05 else 'WRONG'}")

def test_gemm():
  N=getenv("N",2048); M=getenv("M",N); K=getenv("K",N); TM,TN=getenv("TM",4),getenv("TN",4)
  insts=build_gemm_pipe(M,N,K,TM,TN) if getenv("USEPIPE",0) else build_gemm(M,N,K,TM,TN)
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

def _rmse(c, a_np, bt_np):
  c_np=c.float().numpy(); ref=a_np.astype(np.float32)@bt_np.astype(np.float32).T
  return np.sqrt(np.mean((c_np-ref)**2))/(np.sqrt(np.mean(ref**2))+1e-9)

def test_pipe():
  # Fair same-process back-to-back: un-pipelined baseline vs double-buffered pipeline, identical clock.
  M=N=K=getenv("N",2048); TM,TN=getenv("TM",4),getenv("TN",2); CNT=getenv("CNT",30)
  rng=np.random.default_rng(1)
  a_np=(rng.standard_normal((M,K))*0.1).astype(np.float16); bt_np=(rng.standard_normal((N,K))*0.1).astype(np.float16)
  flop=M*N*K*2
  # interleave the two kernels round-robin so clock drift hits both equally
  base_i=build_gemm(M,N,K,TM,TN); pipe_i=build_gemm_pipe(M,N,K,TM,TN)
  cb=Tensor.empty(M,N,dtype=dtypes.half); cp=Tensor.empty(M,N,dtype=dtypes.half); Tensor.realize(cb,cp)
  lb,ob=_run_insts(base_i,Tensor(a_np),Tensor(bt_np),cb,M,N,K,TM,TN,"base")
  lp,op=_run_insts(pipe_i,Tensor(a_np),Tensor(bt_np),cp,M,N,K,TM,TN,"pipe")
  eb,ep=[],[]
  with Context(DEBUG=2):
    for _ in range(CNT):
      st=GlobalCounters.time_sum_s; run_linear(lb); eb.append(GlobalCounters.time_sum_s-st)
      st=GlobalCounters.time_sum_s; run_linear(lp); ep.append(GlobalCounters.time_sum_s-st)
  rb=_rmse(ob,a_np,bt_np); rp=_rmse(op,a_np,bt_np)
  print(f"baseline  TM={TM} TN={TN}: best {flop/min(eb)*1e-12:6.2f}  median {flop/sorted(eb)[len(eb)//2]*1e-12:6.2f} TFLOPS  RMSE {rb:.6f} {'OK' if rb<0.05 else 'WRONG'}")
  print(f"pipeline  TM={TM} TN={TN}: best {flop/min(ep)*1e-12:6.2f}  median {flop/sorted(ep)[len(ep)//2]*1e-12:6.2f} TFLOPS  RMSE {rp:.6f} {'OK' if rp<0.05 else 'WRONG'}")
  print(f"speedup (best) {min(eb)/min(ep):.3f}x")

if __name__ == "__main__":
  if getenv("LDSGEMM2",0): test_lds_gemm2()
  elif getenv("LDSGEMM",0): test_lds_gemm()
  elif getenv("LDSTILE",0): test_lds_tile()
  elif getenv("PIPE",0): test_pipe()
  elif getenv("GEMM",1): test_gemm()
  else: test_tile()
