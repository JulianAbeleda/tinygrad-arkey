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

def waitcnt_lgkm(n):
  # DS/LDS wait: lgkmcnt=bits[9:4] (per extra/qk/prefill/wmma.py). vmcnt/expcnt maxed (don't wait on them).
  return s_waitcnt(simm16=(0x7) | ((n & 0x3F) << 4) | (0x3F << 10))

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

def build_gemm_lds2_q4k(M, N, K, WAVES_M, WAVES_N, WM, WN):
  # Q4_K fused-dequant variant of build_gemm_lds2. A is fp16 [M,K]. B(=Bt) is PACKED Q4_K bytes
  # [N rows x (K//256)*144 bytes]; row = out-neuron, K = in_features. C is fp16 [M,N]. The weight is
  # decoded to fp16 AT COOP-STORE into the SAME fp16 LDS B-tile, so compute()/epilogue are byte-identical
  # to build_gemm_lds2 (only the B global-load + decode + B addressing change). BK is fixed to 32 = exactly
  # one Q4_K sub-group; the K-loop runs over 256-elem SUPER-BLOCKS with the 8 groups Python-unrolled so the
  # group index g (nibble/byte layout + get_scale_min_k4) is a compile-time constant. First correctness
  # variant: requires BN==THREADS (one B-row per thread), DBUF=0, no PLR/LEANADDR. Decode is done in f32.
  BK=32; PAD=0; KT=BK//16
  THREADS=WAVES_M*WAVES_N*32; BM=WAVES_M*WM*16; BN=WAVES_N*WN*16
  assert BN==THREADS, f"q4k v1 requires BN==THREADS, got BN={BN} THREADS={THREADS}"
  assert M%BM==0 and N%BN==0 and K%256==0, f"shape {M}x{N}x{K} not tileable BM={BM} BN={BN}"
  CPR=BK//8; RSTRIDE=THREADS//CPR; loadsA=BM//RSTRIDE
  SA=BK*2+PAD; SB=BK*2+PAD; LDS_A=SA*BM; BUFSZ=LDS_A+SB*BN
  BKPR=(K//256)*144; NSB=K//256
  FA=10; FB=FA+WM*8; ACCb=FB+WN*8; CTA=ACCb+WM*WN*8
  HDR=CTA+loadsA*4; QW=HDR+4; OUT=QW+8                     # B-decode regs: header(4), quant words(8), out(8 = 16 fp16 half)
  # Decode temps live LOW in the FA/FB fragment region (free during decode; compute0 reloads it after). Two hard rules
  # discovered on gfx1100 raw-INS: (1) VGPRs >=238 read back garbage (ELF descriptor doesn't allocate that high), so keep
  # temps low; (2) an FP/cvt result feeding a dependent VALU op is NOT hw-interlocked -> declare RAW with s_delay_alu(1)
  # (s_nop does not satisfy the scoreboard). Decode is done in f32 (fp16 scalar-arith ops proved unreliable here).
  Tdf=10; Tdmf=11; Tdsc=12; Tdmn=13; Rsc=14; Rmn=15; Th=16; Tc=17; Tw=18; Tm=19; Te=20; Ts=21; Tp=22; Ttmp=23; ASCR=24
  assert ASCR < FA+WM*8, f"decode temps overflow FA region {ASCR} vs {FA+WM*8}"
  assert ASCR+1<=256, f"VGPR overflow {ASCR+1}"
  assert BUFSZ<=65536, f"LDS overflow {BUFSZ}"
  I=[]; Br=[]; lbl={}
  def e(i): I.append(i); return i
  def label(n): lbl[n]=sum(i.size() for i in I)
  def br(t): Br.append((len(I)-1,t))
  def dsoff(o): return dict(offset0=o&0xFF, offset1=(o>>8)&0xFF)
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  e(v_lshrrev_b32_e32(v[8], 5, v[0]))                                       # wave
  if WAVES_N==1: e(v_mov_b32_e32(v[19], v[8])); e(v_mov_b32_e32(v[20], 0))
  elif WAVES_N==2: e(v_lshrrev_b32_e32(v[19],1,v[8])); e(v_and_b32_e32(v[20],1,v[8]))
  elif WAVES_N==4: e(v_lshrrev_b32_e32(v[19],2,v[8])); e(v_and_b32_e32(v[20],3,v[8]))
  else: raise AssertionError("WAVES_N in {1,2,4}")
  e(v_and_b32_e32(v[1], 15, v[0]))                                          # tid&15
  e(v_lshlrev_b32_e32(v[6], 4, v[19]))
  e(v_mul_lo_u32(v[6], v[6], WM)); e(v_add_nc_u32_e32(v[6], v[6], v[1])); e(v_mul_lo_u32(v[6], v[6], SA))  # A frag base
  e(v_lshlrev_b32_e32(v[7], 4, v[20])); e(v_mul_lo_u32(v[7], v[7], WN)); e(v_add_nc_u32_e32(v[7], v[7], v[1]))
  e(v_mul_lo_u32(v[7], v[7], SB))                                          # B frag base (LDS_A added in compute)
  lg2=CPR.bit_length()-1
  e(v_and_b32_e32(v[10], CPR-1, v[0])); e(v_lshrrev_b32_e32(v[11], lg2, v[0]))   # A chunk, A row0
  e(s_lshl_b32(s[10], s[3], BM.bit_length()-1)); e(s_lshl_b32(s[11], s[2], BN.bit_length()-1))  # gy*BM, gx*BN
  # A coop addresses (identical to fp16 builder): vA_glob=(gy*BM+row0)*K*2+chunk*16 ; vA_lds=row0*SA+chunk*16
  e(v_add_nc_u32_e32(v[2], s[10], v[11])); e(v_mul_lo_u32(v[2], v[2], K*2)); e(v_lshlrev_b32_e32(v[12],4,v[10])); e(v_add_nc_u32_e32(v[2], v[2], v[12]))
  e(v_mul_lo_u32(v[4], v[11], SA)); e(v_add_nc_u32_e32(v[4], v[4], v[12]))
  # B addressing (this variant): vB_glob = (gx*BN + tid)*BKPR (packed bytes) ; vB_lds = tid*SB
  e(v_add_nc_u32_e32(v[3], s[11], v[0])); e(v_mul_lo_u32(v[3], v[3], BKPR))
  e(v_mul_lo_u32(v[5], v[0], SB))
  for i in range(WM*WN*8): e(v_mov_b32_e32(v[ACCb+i], 0))
  def sbyte_rs(idx): return (HDR+1+idx//4, (idx%4)*8)                       # (reg, shift) of scale byte idx in [0,12)
  def emit_scale(g):                                                       # get_scale_min_k4 -> sc int in Rsc, mn int in Rmn
    if g<4:
      r,sh=sbyte_rs(g);     e(v_lshrrev_b32_e32(v[Rsc], sh, v[r])); e(v_and_b32_e32(v[Rsc], 63, v[Rsc]))
      r,sh=sbyte_rs(4+g);   e(v_lshrrev_b32_e32(v[Rmn], sh, v[r])); e(v_and_b32_e32(v[Rmn], 63, v[Rmn]))
    else:
      gg=g-4
      r,sh=sbyte_rs(8+gg);  e(v_lshrrev_b32_e32(v[Th], sh, v[r])); e(v_and_b32_e32(v[Th], 0xff, v[Th]))   # high byte
      r,sh=sbyte_rs(gg);    e(v_lshrrev_b32_e32(v[Rsc], sh, v[r])); e(v_and_b32_e32(v[Rsc], 0xff, v[Rsc]))
      e(v_lshrrev_b32_e32(v[Rsc], 6, v[Rsc])); e(v_lshlrev_b32_e32(v[Rsc], 4, v[Rsc]))
      e(v_and_b32_e32(v[Ttmp], 0xf, v[Th])); e(v_or_b32_e32(v[Rsc], v[Rsc], v[Ttmp]))
      r,sh=sbyte_rs(4+gg);  e(v_lshrrev_b32_e32(v[Rmn], sh, v[r])); e(v_and_b32_e32(v[Rmn], 0xff, v[Rmn]))
      e(v_lshrrev_b32_e32(v[Rmn], 6, v[Rmn])); e(v_lshlrev_b32_e32(v[Rmn], 4, v[Rmn]))
      e(v_lshrrev_b32_e32(v[Ttmp], 4, v[Th])); e(v_or_b32_e32(v[Rmn], v[Rmn], v[Ttmp]))
  def expand_f16(Hin, dst):                                               # normal fp16 in Hin[15:0] -> f32 in dst (integer ops; d/dmin are normal, exp!=0)
    e(v_and_b32_e32(v[Tm], 0x3ff, v[Hin])); e(v_lshlrev_b32_e32(v[Tm], 13, v[Tm]))            # mant<<13
    e(v_lshrrev_b32_e32(v[Te], 10, v[Hin])); e(v_and_b32_e32(v[Te], 0x1f, v[Te]))             # exp
    e(v_add_nc_u32_e32(v[Te], 112, v[Te])); e(v_lshlrev_b32_e32(v[Te], 23, v[Te]))            # (exp + (127-15))<<23
    e(v_or_b32_e32(v[dst], v[Tm], v[Te]))
    e(v_lshrrev_b32_e32(v[Ts], 15, v[Hin])); e(v_and_b32_e32(v[Ts], 1, v[Ts])); e(v_lshlrev_b32_e32(v[Ts], 31, v[Ts]))  # sign<<31
    e(v_or_b32_e32(v[dst], v[dst], v[Ts]))
  def decode_group(g):                                                     # QW[0:8] loaded -> OUT (32 fp16) -> LDS. f32 math.
    expand_f16(HDR, Tdf)                                                   # d (HDR low16) -> f32 (recompute per group; compute0 clobbers Tdf reg after each group, HDR persists)
    e(v_lshrrev_b32_e32(v[Ttmp], 16, v[HDR])); expand_f16(Ttmp, Tdmf)     # dmin (HDR high16) -> f32
    emit_scale(g)                                                          # Rsc, Rmn = int sc, mn (integer, interlocked)
    e(v_cvt_f32_i32_e32(v[Rsc], v[Rsc])); e(s_delay_alu(simm16=1)); e(v_mul_f32_e32(v[Tdsc], v[Tdf], v[Rsc]))     # d*sc
    e(v_cvt_f32_i32_e32(v[Rmn], v[Rmn])); e(s_delay_alu(simm16=1)); e(v_mul_f32_e32(v[Tdmn], v[Tdmf], v[Rmn]))     # dmin*mn
    for half in range(2):
      for ll in range(16):
        l=half*16+ll; sh=(l%4)*8+(g%2)*4
        e(v_lshrrev_b32_e32(v[Tc], sh, v[QW+l//4])); e(v_and_b32_e32(v[Tc], 0xf, v[Tc]))      # code int (integer)
        e(v_cvt_f32_i32_e32(v[Tc], v[Tc])); e(s_delay_alu(simm16=1))                          # code f32
        e(v_mul_f32_e32(v[Tw], v[Tdsc], v[Tc])); e(s_delay_alu(simm16=1))                     # d*sc*code
        e(v_sub_f32_e32(v[Tw], v[Tw], v[Tdmn])); e(s_delay_alu(simm16=1))                     # - dmin*mn
        e(v_cvt_f16_f32_e32(v[Tw], v[Tw])); e(s_delay_alu(simm16=1))                          # -> fp16, fence before pack
        if ll%2==0: e(v_and_b32_e32(v[OUT+ll//2], 0xffff, v[Tw]))                             # even -> low16 (clear high)
        else: e(v_lshlrev_b32_e32(v[Tp], 16, v[Tw])); e(v_or_b32_e32(v[OUT+ll//2], v[OUT+ll//2], v[Tp]))  # odd -> high16
      e(ds_store_b128(addr=v[5], data0=v[OUT:OUT+3],   **dsoff(LDS_A+half*32+0)))
      e(ds_store_b128(addr=v[5], data0=v[OUT+4:OUT+7], **dsoff(LDS_A+half*32+16)))
  def coop_load_A():
    for j in range(loadsA):
      if j==0: ar=2
      else: e(v_add_nc_u32_e32(v[ASCR], j*RSTRIDE*K*2, v[2])); ar=ASCR
      e(global_load_b128(vdst=v[CTA+j*4:CTA+j*4+3], addr=v[ar:ar], saddr=s[4:5], offset=0))
  def coop_store_A():
    for j in range(loadsA): e(ds_store_b128(addr=v[4], data0=v[CTA+j*4:CTA+j*4+3], **dsoff(j*RSTRIDE*SA)))
  def compute0():
    for kt in range(KT):
      for mi in range(WM):
        o=mi*16*SA+kt*32
        e(ds_load_b128(vdst=v[FA+mi*8:FA+mi*8+3],   addr=v[6], **dsoff(o)))
        e(ds_load_b128(vdst=v[FA+mi*8+4:FA+mi*8+7], addr=v[6], **dsoff(o+16)))
      for ni in range(WN):
        o=LDS_A+ni*16*SB+kt*32
        e(ds_load_b128(vdst=v[FB+ni*8:FB+ni*8+3],   addr=v[7], **dsoff(o)))
        e(ds_load_b128(vdst=v[FB+ni*8+4:FB+ni*8+7], addr=v[7], **dsoff(o+16)))
      e(waitcnt_lgkm(0))
      for mi in range(WM):
        for ni in range(WN):
          ac=ACCb+(mi*WN+ni)*8
          e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+mi*8:FA+mi*8+7], src1=v[FB+ni*8:FB+ni*8+7], src2=v[ac:ac+7]))
  e(s_mov_b32(s[16], 0))
  label('LOOP')
  e(global_load_b128(vdst=v[HDR:HDR+3], addr=v[3:3], saddr=s[6:7], offset=0)); e(waitcnt_vm(0))
  # d/dmin are decoded per-group inside decode_group (Tdf/Tdmf regs live in the FA region that compute0 clobbers each
  # group; HDR itself persists across the 8-group loop, so re-expand from it). Nothing to precompute here.
  for g in range(8):
    qb=16+(g//2)*32
    e(global_load_b128(vdst=v[QW:QW+3],   addr=v[3:3], saddr=s[6:7], offset=qb))
    e(global_load_b128(vdst=v[QW+4:QW+7], addr=v[3:3], saddr=s[6:7], offset=qb+16))
    coop_load_A(); e(waitcnt_vm(0))
    decode_group(g)                                                        # B -> LDS
    coop_store_A()                                                         # A -> LDS
    e(waitcnt_lgkm(0)); e(s_barrier())
    compute0(); e(s_barrier())
    e(v_add_nc_u32_e32(v[2], BK*2, v[2]))                                   # advance A K-position
  e(v_add_nc_u32_e32(v[3], 144, v[3]))                                      # advance B super-block
  e(s_add_i32(s[16], s[16], 1)); e(s_cmp_lt_i32(s[16], NSB)); e(s_cbranch_scc1(simm16=0)); br('LOOP')
  # epilogue (identical to build_gemm_lds2)
  e(v_and_b32_e32(v[8], 15, v[0])); e(v_lshrrev_b32_e32(v[9], 4, v[0])); e(v_and_b32_e32(v[9], 1, v[9]))
  e(v_lshrrev_b32_e32(v[10], 5, v[0]))
  if WAVES_N==1: e(v_mov_b32_e32(v[11], v[10])); e(v_mov_b32_e32(v[15], 0))
  elif WAVES_N==2: e(v_lshrrev_b32_e32(v[11],1,v[10])); e(v_and_b32_e32(v[15],1,v[10]))
  else: e(v_lshrrev_b32_e32(v[11],2,v[10])); e(v_and_b32_e32(v[15],3,v[10]))
  e(v_lshlrev_b32_e32(v[21], 4, v[11])); e(v_mul_lo_u32(v[21], v[21], WM)); e(v_add_nc_u32_e32(v[21], s[10], v[21]))
  e(v_lshlrev_b32_e32(v[22], 4, v[15])); e(v_mul_lo_u32(v[22], v[22], WN)); e(v_add_nc_u32_e32(v[22], s[11], v[22]))
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

