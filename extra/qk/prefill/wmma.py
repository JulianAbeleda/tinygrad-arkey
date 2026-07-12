# RDNA3 (gfx1100) WMMA GEMM — hand-assembly, zero LLVM, via assemble->ELF.
# Built incrementally: STAGE 1 = single 16x16x16 tile to nail the RDNA3 WMMA operand layout.
# RDNA3 wave32 v_wmma_f32_16x16x16_f16:
#   A (src0) = 8 VGPR/lane (16 fp16): lane l(0..15) holds A[l][0:16]; lanes 16..31 replicate.
#   B (src1) = 8 VGPR/lane (16 fp16): lane l holds B[0:16][l] (a column). We pass B TRANSPOSED in
#     memory (Bt[n][k]=B[k][n]) so a column is contiguous -> Bt[l][0:16].
#   C/D (src2/vdst) = 8 VGPR/lane (8 fp32): D[i] of lane l = C[row=i*2+(l>>4&1)][col=l&15].
from dataclasses import dataclass
import os
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv, colored
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.renderer.amd.dsl import s, v, VCC_LO, NULL, src, ttmp
from tinygrad.runtime.autogen.amd.rdna3.ins import *


@dataclass(frozen=True)
class RegisterLease:
  """One contiguous virtual register reservation."""
  name: str
  start: int
  count: int
  bank: str
  @property
  def end(self) -> int: return self.start + self.count


class AMDRegisterLeaseAllocator:
  """Single reservation interface for AMD ABI and WMMA register windows."""
  def __init__(self, *, vgpr_capacity: int = 256, sgpr_capacity: int = 106):
    self.vgpr_capacity, self.sgpr_capacity = vgpr_capacity, sgpr_capacity
    self._leases: list[RegisterLease] = []
  @property
  def leases(self) -> tuple[RegisterLease, ...]: return tuple(self._leases)
  @property
  def virtual_vgpr_pool(self) -> int: return max((x.end for x in self._leases if x.bank == "vgpr"), default=0)
  @property
  def virtual_sgpr_pool(self) -> int: return max((x.end for x in self._leases if x.bank == "sgpr"), default=0)
  def reserve(self, name: str, start: int, count: int, *, bank: str) -> RegisterLease:
    if bank not in ("vgpr", "sgpr") or not name or not isinstance(start, int) or not isinstance(count, int) or start < 0 or count <= 0:
      raise ValueError("invalid AMD register lease")
    capacity = self.vgpr_capacity if bank == "vgpr" else self.sgpr_capacity
    if start + count > capacity: raise ValueError(f"{bank} lease exceeds virtual pool")
    if any(x.bank == bank and start < x.end and x.start < start + count for x in self._leases):
      raise ValueError(f"{bank} lease overlaps an existing reservation")
    lease = RegisterLease(name, start, count, bank); self._leases.append(lease); return lease
  def allocate(self, name: str, count: int, *, bank: str, align: int = 1) -> RegisterLease:
    if not isinstance(align, int) or align <= 0: raise ValueError("lease alignment must be positive")
    capacity = self.vgpr_capacity if bank == "vgpr" else self.sgpr_capacity
    cursor = 0
    for lease in sorted((x for x in self._leases if x.bank == bank), key=lambda x: x.start):
      cursor = (cursor + align - 1) // align * align
      if cursor + count <= lease.start: return self.reserve(name, cursor, count, bank=bank)
      cursor = max(cursor, lease.end)
    cursor = (cursor + align - 1) // align * align
    if cursor + count > capacity: raise ValueError(f"{bank} virtual pool exhausted")
    return self.reserve(name, cursor, count, bank=bank)
  @classmethod
  def with_fixed_abi(cls) -> "AMDRegisterLeaseAllocator":
    out = cls()
    for name, start, count in (("abi", 0, 4), ("buffer_a", 4, 2), ("buffer_b", 6, 2),
                               ("output", 8, 2), ("workgroup_coords", 10, 2), ("loop_counter", 16, 1)):
      out.reserve(name, start, count, bank="sgpr")
    out.reserve("fixed_lane_and_address", 0, 10, bank="vgpr")
    return out


FA, FB, ACC = 20, 32, 44   # VGPR bases: A frag(8), B frag(8), accumulator(8)

@dataclass(frozen=True)
class LDS2RegLayout:
  FA: int
  FB: int
  ACCb: int
  CTA: int
  CTB: int
  SCR: int
  FB2: int

  def validate(self, WM, WN, loadsA, loadsB, PLRAB=0):
    if self.FA < 0 or self.FB < 0 or self.ACCb < 0 or self.CTA < 0 or self.CTB < 0 or self.SCR < 0 or self.FB2 < 0:
      raise AssertionError(f"negative LDS2 VGPR layout field: {self}")
    if self.SCR+2 > 256: raise AssertionError(f"VGPR overflow {self.SCR+2}")
    if PLRAB and self.FB2+WM*8+WN*8 > 256:
      raise AssertionError(f"PLRAB VGPR overflow {self.FB2+WM*8+WN*8} (needs smaller tile than {WM}x{WN})")
    if self.FB < self.FA + WM*8: raise AssertionError(f"LDS2 layout overlaps A/B fragments: {self}")
    if self.ACCb < self.FB + WN*8: raise AssertionError(f"LDS2 layout overlaps B fragments/accumulators: {self}")
    if self.CTA < self.ACCb + WM*WN*8: raise AssertionError(f"LDS2 layout overlaps accumulators/CTA: {self}")
    if self.CTB < self.CTA + loadsA*4: raise AssertionError(f"LDS2 layout overlaps CTA/CTB: {self}")
    if self.SCR < self.CTB + loadsB*4: raise AssertionError(f"LDS2 layout overlaps CTB/SCR: {self}")
    if self.FB2 < self.SCR + 2: raise AssertionError(f"LDS2 layout overlaps scratch/PLRAB buffer: {self}")
    return self

def default_lds2_reg_layout(WM, WN, loadsA, loadsB) -> LDS2RegLayout:
  alloc = AMDRegisterLeaseAllocator.with_fixed_abi()
  fa = alloc.allocate("wmma_fragment_a", WM*8, bank="vgpr")
  fb = alloc.allocate("wmma_fragment_b", WN*8, bank="vgpr")
  acc = alloc.allocate("wmma_accumulator", WM*WN*8, bank="vgpr")
  cta = alloc.allocate("lds_pack_a", loadsA*4, bank="vgpr")
  ctb = alloc.allocate("lds_pack_b", loadsB*4, bank="vgpr")
  scratch = alloc.allocate("address_scratch", 2, bank="vgpr")
  return LDS2RegLayout(FA=fa.start, FB=fb.start, ACCb=acc.start, CTA=cta.start, CTB=ctb.start,
                       SCR=scratch.start, FB2=scratch.end)

def env_lds2_reg_layout(WM, WN, loadsA, loadsB) -> LDS2RegLayout:
  layout = default_lds2_reg_layout(WM, WN, loadsA, loadsB)
  shift = int(os.environ.get("PREFILL_LDS2_REG_BLOCK_SHIFT", "0"))
  if shift:
    vals = {k: v + shift for k, v in layout.__dict__.items()}
    layout = LDS2RegLayout(**vals)
  return layout

@dataclass(frozen=True)
class LDS2MemoryLayout:
  SA: int
  SB: int
  LDS_A: int
  BUFSZ: int
  NBUF: int

  def validate(self):
    if self.SA <= 0 or self.SB <= 0 or self.LDS_A <= 0 or self.BUFSZ <= 0 or self.NBUF not in (1, 2):
      raise AssertionError(f"invalid LDS2 memory layout: {self}")
    if self.BUFSZ*self.NBUF > 65536: raise AssertionError(f"LDS overflow {self.BUFSZ*self.NBUF}")
    return self

def default_lds2_memory_layout(BM, BN, BK, PAD, DBUF) -> LDS2MemoryLayout:
  SA=BK*2+PAD; SB=BK*2+PAD; LDS_A=SA*BM; BUFSZ=LDS_A+SB*BN; NBUF=2 if DBUF else 1
  return LDS2MemoryLayout(SA=SA, SB=SB, LDS_A=LDS_A, BUFSZ=BUFSZ, NBUF=NBUF)

@dataclass(frozen=True)
class LDS2WaitPolicy:
  vm_after_coop_load: int = 0
  lgkm_after_coop_store: int = 0
  lgkm_after_frag_load: int = 0

  def validate(self):
    for name in ("vm_after_coop_load", "lgkm_after_coop_store", "lgkm_after_frag_load"):
      val = getattr(self, name)
      if not 0 <= val <= 63: raise AssertionError(f"invalid LDS2 wait policy {name}={val}")
    return self

  def wait_after_coop_load(self): return waitcnt_vm(self.vm_after_coop_load)
  def wait_after_coop_store(self): return waitcnt_lgkm(self.lgkm_after_coop_store)
  def wait_after_frag_load(self): return waitcnt_lgkm(self.lgkm_after_frag_load)

def default_lds2_wait_policy() -> LDS2WaitPolicy:
  return LDS2WaitPolicy()

def env_lds2_wait_policy() -> LDS2WaitPolicy:
  return LDS2WaitPolicy(
    vm_after_coop_load=int(os.environ.get("PREFILL_LDS2_WAIT_VM_COOP_LOAD", "0")),
    lgkm_after_coop_store=int(os.environ.get("PREFILL_LDS2_WAIT_LGKM_COOP_STORE", "0")),
    lgkm_after_frag_load=int(os.environ.get("PREFILL_LDS2_WAIT_LGKM_FRAG_LOAD", "0")),
  )

@dataclass(frozen=True)
class LDS2Cadence:
  double_buffer: bool

  def validate(self, DBUF):
    if self.double_buffer != bool(DBUF):
      raise AssertionError(f"LDS2 cadence double_buffer={self.double_buffer} disagrees with DBUF={DBUF}")
    return self

def default_lds2_cadence(DBUF) -> LDS2Cadence:
  return LDS2Cadence(double_buffer=bool(DBUF))

@dataclass(frozen=True)
class LDS2LifecycleStep:
  op: str
  slot: int | None = None

@dataclass(frozen=True)
class LDS2LifecycleTemplate:
  double_buffer: bool
  prologue: tuple[LDS2LifecycleStep, ...]
  body: tuple[LDS2LifecycleStep, ...]
  tail: tuple[LDS2LifecycleStep, ...]

  def validate(self, DBUF):
    if self.double_buffer != bool(DBUF):
      raise AssertionError(f"LDS2 lifecycle double_buffer={self.double_buffer} disagrees with DBUF={DBUF}")
    valid = {
      "init_counter", "label_loop", "coop_load", "wait_coop_load", "coop_store", "wait_coop_store",
      "barrier", "compute", "compute_plr", "adv_k", "branch_nblk", "branch_nl",
    }
    for phase in (self.prologue, self.body, self.tail):
      for step in phase:
        if step.op not in valid: raise AssertionError(f"invalid LDS2 lifecycle op={step.op!r}")
        if step.op in {"coop_load", "coop_store", "compute", "compute_plr"} and step.slot not in (0, 1):
          raise AssertionError(f"invalid LDS2 lifecycle slot for {step.op}: {step.slot}")
    return self

def _ls(op, slot=None) -> LDS2LifecycleStep:
  return LDS2LifecycleStep(op, slot)

def default_lds2_lifecycle_template(DBUF) -> LDS2LifecycleTemplate:
  if not DBUF:
    return LDS2LifecycleTemplate(
      double_buffer=False,
      prologue=(_ls("init_counter"), _ls("label_loop")),
      body=(
        _ls("coop_load", 0), _ls("wait_coop_load"), _ls("coop_store", 0), _ls("wait_coop_store"), _ls("barrier"),
        _ls("compute_plr", 0), _ls("barrier"), _ls("adv_k"), _ls("branch_nblk"),
      ),
      tail=(),
    )
  return LDS2LifecycleTemplate(
    double_buffer=True,
    prologue=(
      _ls("coop_load", 0), _ls("wait_coop_load"), _ls("coop_store", 0), _ls("wait_coop_store"), _ls("barrier"),
      _ls("adv_k"), _ls("init_counter"), _ls("label_loop"),
    ),
    body=(
      _ls("coop_load", 1), _ls("compute", 0), _ls("wait_coop_load"), _ls("coop_store", 1), _ls("wait_coop_store"), _ls("barrier"), _ls("adv_k"),
      _ls("coop_load", 0), _ls("compute", 1), _ls("wait_coop_load"), _ls("coop_store", 0), _ls("wait_coop_store"), _ls("barrier"), _ls("adv_k"),
      _ls("branch_nl"),
    ),
    tail=(
      _ls("coop_load", 1), _ls("compute", 0), _ls("wait_coop_load"), _ls("coop_store", 1), _ls("wait_coop_store"), _ls("barrier"),
      _ls("compute", 1),
    ),
  )

def env_lds2_lifecycle_template(DBUF) -> LDS2LifecycleTemplate:
  template = default_lds2_lifecycle_template(DBUF)
  if DBUF and os.environ.get("PREFILL_LDS2_LIFECYCLE_PROLOGUE_INIT_BEFORE_ADV_K", "0") != "0":
    template = LDS2LifecycleTemplate(
      double_buffer=True,
      prologue=template.prologue[:5] + (template.prologue[6], template.prologue[5], template.prologue[7]),
      body=template.body,
      tail=template.tail,
    )
  return template

def waitcnt_lgkm(n):
  # DS/LDS wait: lgkmcnt=bits[9:4] (per extra/qk/prefill/wmma.py). vmcnt/expcnt maxed (don't wait on them).
  return s_waitcnt(simm16=(0x7) | ((n & 0x3F) << 4) | (0x3F << 10))

def waitcnt_vm(n):
  # s_waitcnt simm16 (matches the proven in-repo encoder, extra/qk/prefill/wmma.py):
  #   expcnt=bits[2:0], lgkmcnt=bits[9:4], vmcnt=bits[15:10].
  # Wait until <=n outstanding VMEM loads; leave expcnt/lgkmcnt maxed (don't wait on them).
  if getenv("FULLWAIT",0): return s_waitcnt(simm16=0)
  return s_waitcnt(simm16=(0x7) | ((0x3F) << 4) | ((n & 0x3F) << 10))

class LDS2PrimitiveEmitter:
  def __init__(self, emit, label, branch, **kwargs):
    self.e, self.label, self.branch = emit, label, branch
    self.__dict__.update(kwargs)
    self.VRA=self.SCR+1; self.VRB=self.VRA+self.loadsA; self.SKA=18; self.SKB=20

  def dsoff(self, o): return dict(offset0=o&0xFF, offset1=(o>>8)&0xFF)

  def emit_kernel_prologue(self):
    self.e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
    self.e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
    self.e(s_waitcnt(simm16=0))

  def emit_tile_setup(self):
    self.e(v_lshrrev_b32_e32(v[8], 5, v[0]))
    if self.WAVES_N==1: self.e(v_mov_b32_e32(v[19], v[8])); self.e(v_mov_b32_e32(v[20], 0))
    elif self.WAVES_N==2: self.e(v_lshrrev_b32_e32(v[19],1,v[8])); self.e(v_and_b32_e32(v[20],1,v[8]))
    elif self.WAVES_N==4: self.e(v_lshrrev_b32_e32(v[19],2,v[8])); self.e(v_and_b32_e32(v[20],3,v[8]))
    else: raise AssertionError("WAVES_N in {1,2,4}")
    self.e(v_and_b32_e32(v[1], 15, v[0]))
    self.e(v_lshlrev_b32_e32(v[6], 4, v[19]))
    self.e(v_mul_lo_u32(v[6], v[6], self.WM)); self.e(v_add_nc_u32_e32(v[6], v[6], v[1])); self.e(v_mul_lo_u32(v[6], v[6], self.SA))
    self.e(v_lshlrev_b32_e32(v[7], 4, v[20])); self.e(v_mul_lo_u32(v[7], v[7], self.WN)); self.e(v_add_nc_u32_e32(v[7], v[7], v[1]))
    self.e(v_mul_lo_u32(v[7], v[7], self.SB))
    lg2=self.CPR.bit_length()-1
    self.e(v_and_b32_e32(v[10], self.CPR-1, v[0])); self.e(v_lshrrev_b32_e32(v[11], lg2, v[0]))
    self.e(s_lshl_b32(s[10], s[3], self.BM.bit_length()-1)); self.e(s_lshl_b32(s[11], s[2], self.BN.bit_length()-1))
    self.e(v_add_nc_u32_e32(v[2], s[10], v[11])); self.e(v_mul_lo_u32(v[2], v[2], self.K*2)); self.e(v_lshlrev_b32_e32(v[12],4,v[10])); self.e(v_add_nc_u32_e32(v[2], v[2], v[12]))
    self.e(v_mul_lo_u32(v[4], v[11], self.SA)); self.e(v_add_nc_u32_e32(v[4], v[4], v[12]))
    self.e(v_add_nc_u32_e32(v[3], s[11], v[11])); self.e(v_mul_lo_u32(v[3], v[3], self.K*2)); self.e(v_add_nc_u32_e32(v[3], v[3], v[12]))
    self.e(v_mul_lo_u32(v[5], v[11], self.SB)); self.e(v_add_nc_u32_e32(v[5], v[5], v[12]))

  def zero_accumulators(self):
    for i in range(self.WM*self.WN*8): self.e(v_mov_b32_e32(v[self.ACCb+i], 0))

  def setup_leanaddr(self):
    if not self.LEANADDR: return
    assert self.VRB+self.loadsB<=256, f"LEANADDR VGPR overflow {self.VRB+self.loadsB}"
    for j in range(self.loadsA):
      self.e(v_mov_b32_e32(v[self.VRA+j], v[2]) if j==0 else v_add_nc_u32_e32(v[self.VRA+j], j*self.RSTRIDE*self.K*2, v[2]))
    for j in range(self.loadsB):
      self.e(v_mov_b32_e32(v[self.VRB+j], v[3]) if j==0 else v_add_nc_u32_e32(v[self.VRB+j], j*self.RSTRIDE*self.K*2, v[3]))
    self.e(s_mov_b32(s[self.SKA], s[4])); self.e(s_mov_b32(s[self.SKA+1], s[5]))
    self.e(s_mov_b32(s[self.SKB], s[6])); self.e(s_mov_b32(s[self.SKB+1], s[7]))

  def coop_load_lean(self, buf):
    for j in range(self.loadsA):
      self.e(global_load_b128(vdst=v[self.CTA+j*4:self.CTA+j*4+3], addr=v[self.VRA+j:self.VRA+j], saddr=s[self.SKA:self.SKA+1], offset=0))
    for j in range(self.loadsB):
      self.e(global_load_b128(vdst=v[self.CTB+j*4:self.CTB+j*4+3], addr=v[self.VRB+j:self.VRB+j], saddr=s[self.SKB:self.SKB+1], offset=0))

  def adv_kbase(self):
    self.e(s_add_u32(s[self.SKA], s[self.SKA], self.BK*2)); self.e(s_addc_u32(s[self.SKA+1], s[self.SKA+1], 0))
    self.e(s_add_u32(s[self.SKB], s[self.SKB], self.BK*2)); self.e(s_addc_u32(s[self.SKB+1], s[self.SKB+1], 0))

  def coop_load(self, buf):
    if self.LEANADDR: return self.coop_load_lean(buf)
    for j in range(self.loadsA):
      if j==0: ar=2
      else: self.e(v_add_nc_u32_e32(v[self.SCR], j*self.RSTRIDE*self.K*2, v[2])); ar=self.SCR
      self.e(global_load_b128(vdst=v[self.CTA+j*4:self.CTA+j*4+3], addr=v[ar:ar], saddr=s[4:5], offset=0))
    for j in range(self.loadsB):
      if j==0: br_=3
      else: self.e(v_add_nc_u32_e32(v[self.SCR], j*self.RSTRIDE*self.K*2, v[3])); br_=self.SCR
      self.e(global_load_b128(vdst=v[self.CTB+j*4:self.CTB+j*4+3], addr=v[br_:br_], saddr=s[6:7], offset=0))

  def coop_store(self, buf):
    bo=buf*self.BUFSZ
    for j in range(self.loadsA):
      self.e(ds_store_b128(addr=v[4], data0=v[self.CTA+j*4:self.CTA+j*4+3], **self.dsoff(bo+j*self.RSTRIDE*self.SA)))
    for j in range(self.loadsB):
      self.e(ds_store_b128(addr=v[5], data0=v[self.CTB+j*4:self.CTB+j*4+3], **self.dsoff(bo+self.LDS_A+j*self.RSTRIDE*self.SB)))

  def compute(self, buf):
    bo=buf*self.BUFSZ
    for kt in range(self.KT):
      for mi in range(self.WM):
        o=bo+mi*16*self.SA+kt*32
        self.e(ds_load_b128(vdst=v[self.FA+mi*8:self.FA+mi*8+3],   addr=v[6], **self.dsoff(o)))
        if not self.DSHALF: self.e(ds_load_b128(vdst=v[self.FA+mi*8+4:self.FA+mi*8+7], addr=v[6], **self.dsoff(o+16)))
      for ni in range(self.WN):
        o=bo+self.LDS_A+ni*16*self.SB+kt*32
        self.e(ds_load_b128(vdst=v[self.FB+ni*8:self.FB+ni*8+3],   addr=v[7], **self.dsoff(o)))
        if not self.DSHALF: self.e(ds_load_b128(vdst=v[self.FB+ni*8+4:self.FB+ni*8+7], addr=v[7], **self.dsoff(o+16)))
      self.e(self.wait.wait_after_frag_load())
      for mi in range(self.WM):
        for ni in range(self.WN):
          ac=self.ACCb+(mi*self.WN+ni)*8
          self.e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[self.FA+mi*8:self.FA+mi*8+7], src1=v[self.FB+ni*8:self.FB+ni*8+7], src2=v[ac:ac+7]))

  def compute_plra(self, buf):
    assert self.KT==2 and (self.loadsA*4+self.loadsB*4)>=self.WM*8, "PLRA needs KT==2 and dead CTA/CTB room for WM*8 A-frags"
    bo=buf*self.BUFSZ; FAp=self.CTA
    def la(dst,kt):
      for mi in range(self.WM):
        o=bo+mi*16*self.SA+kt*32
        self.e(ds_load_b128(vdst=v[dst+mi*8:dst+mi*8+3],   addr=v[6], **self.dsoff(o)))
        self.e(ds_load_b128(vdst=v[dst+mi*8+4:dst+mi*8+7], addr=v[6], **self.dsoff(o+16)))
    def lb(kt):
      for ni in range(self.WN):
        o=bo+self.LDS_A+ni*16*self.SB+kt*32
        self.e(ds_load_b128(vdst=v[self.FB+ni*8:self.FB+ni*8+3],   addr=v[7], **self.dsoff(o)))
        self.e(ds_load_b128(vdst=v[self.FB+ni*8+4:self.FB+ni*8+7], addr=v[7], **self.dsoff(o+16)))
    def ww(As):
      for mi in range(self.WM):
        for ni in range(self.WN):
          ac=self.ACCb+(mi*self.WN+ni)*8
          self.e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[As+mi*8:As+mi*8+7], src1=v[self.FB+ni*8:self.FB+ni*8+7], src2=v[ac:ac+7]))
    la(self.FA,0); lb(0); self.e(self.wait.wait_after_frag_load())
    la(FAp,1); ww(self.FA); lb(1); self.e(self.wait.wait_after_frag_load()); ww(FAp)

  def compute_plrab(self, buf):
    assert self.KT==2, "PLRAB needs KT==2"
    bo=buf*self.BUFSZ; FAp=self.FB2; FBp=self.FB2+self.WM*8
    def la(dst,kt):
      for mi in range(self.WM):
        o=bo+mi*16*self.SA+kt*32
        self.e(ds_load_b128(vdst=v[dst+mi*8:dst+mi*8+3],   addr=v[6], **self.dsoff(o)))
        self.e(ds_load_b128(vdst=v[dst+mi*8+4:dst+mi*8+7], addr=v[6], **self.dsoff(o+16)))
    def lb(dst,kt):
      for ni in range(self.WN):
        o=bo+self.LDS_A+ni*16*self.SB+kt*32
        self.e(ds_load_b128(vdst=v[dst+ni*8:dst+ni*8+3],   addr=v[7], **self.dsoff(o)))
        self.e(ds_load_b128(vdst=v[dst+ni*8+4:dst+ni*8+7], addr=v[7], **self.dsoff(o+16)))
    def ww(As,Bs):
      for mi in range(self.WM):
        for ni in range(self.WN):
          ac=self.ACCb+(mi*self.WN+ni)*8
          self.e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[As+mi*8:As+mi*8+7], src1=v[Bs+ni*8:Bs+ni*8+7], src2=v[ac:ac+7]))
    la(self.FA,0); lb(self.FB,0); self.e(self.wait.wait_after_frag_load())
    la(FAp,1); lb(FBp,1); ww(self.FA,self.FB); self.e(self.wait.wait_after_frag_load()); ww(FAp,FBp)

  def compute_selected_plr(self, buf):
    return self.compute_plrab(buf) if self.PLRAB else self.compute_plra(buf) if self.PLRA else self.compute(buf)

  def advance_k(self):
    if self.LEANADDR: self.adv_kbase()
    else: self.e(v_add_nc_u32_e32(v[2], self.BK*2, v[2])); self.e(v_add_nc_u32_e32(v[3], self.BK*2, v[3]))

  def emit_lifecycle_step(self, step):
    if step.op == "init_counter": self.e(s_mov_b32(s[16], 0))
    elif step.op == "label_loop": self.label('LOOP')
    elif step.op == "coop_load": self.coop_load(step.slot)
    elif step.op == "wait_coop_load": self.e(self.wait.wait_after_coop_load())
    elif step.op == "coop_store": self.coop_store(step.slot)
    elif step.op == "wait_coop_store": self.e(self.wait.wait_after_coop_store())
    elif step.op == "barrier": self.e(s_barrier())
    elif step.op == "compute": self.compute(step.slot)
    elif step.op == "compute_plr": self.compute_selected_plr(step.slot)
    elif step.op == "adv_k": self.advance_k()
    elif step.op == "branch_nblk":
      self.e(s_add_i32(s[16], s[16], 1)); self.e(s_cmp_lt_i32(s[16], self.NBLK)); self.e(s_cbranch_scc1(simm16=0)); self.branch('LOOP')
    elif step.op == "branch_nl":
      self.e(s_add_i32(s[16], s[16], 1)); self.e(s_cmp_lt_i32(s[16], self.NL)); self.e(s_cbranch_scc1(simm16=0)); self.branch('LOOP')
    else: raise AssertionError(f"unhandled LDS2 lifecycle step {step}")

  def emit_lifecycle(self, lifecycle):
    for phase in (lifecycle.prologue, lifecycle.body, lifecycle.tail):
      for step in phase: self.emit_lifecycle_step(step)

  def emit_epilogue(self):
    self.e(v_and_b32_e32(v[8], 15, v[0])); self.e(v_lshrrev_b32_e32(v[9], 4, v[0])); self.e(v_and_b32_e32(v[9], 1, v[9]))
    self.e(v_lshrrev_b32_e32(v[10], 5, v[0]))
    if self.WAVES_N==1: self.e(v_mov_b32_e32(v[11], v[10])); self.e(v_mov_b32_e32(v[15], 0))
    elif self.WAVES_N==2: self.e(v_lshrrev_b32_e32(v[11],1,v[10])); self.e(v_and_b32_e32(v[15],1,v[10]))
    else: self.e(v_lshrrev_b32_e32(v[11],2,v[10])); self.e(v_and_b32_e32(v[15],3,v[10]))
    self.e(v_lshlrev_b32_e32(v[21], 4, v[11])); self.e(v_mul_lo_u32(v[21], v[21], self.WM)); self.e(v_add_nc_u32_e32(v[21], s[10], v[21]))
    self.e(v_lshlrev_b32_e32(v[22], 4, v[15])); self.e(v_mul_lo_u32(v[22], v[22], self.WN)); self.e(v_add_nc_u32_e32(v[22], s[11], v[22]))
    for mi in range(self.WM):
      for ni in range(self.WN):
        ac=self.ACCb+(mi*self.WN+ni)*8
        self.e(v_add_nc_u32_e32(v[12], v[21], v[9])); self.e(v_add_nc_u32_e32(v[12], mi*16, v[12]))
        self.e(v_add_nc_u32_e32(v[13], v[22], v[8])); self.e(v_add_nc_u32_e32(v[13], ni*16, v[13]))
        self.e(v_mul_lo_u32(v[12], v[12], self.N)); self.e(v_add_nc_u32_e32(v[12], v[12], v[13])); self.e(v_lshlrev_b32_e32(v[12], 1, v[12]))
        for i in range(8):
          self.e(v_cvt_f16_f32_e32(v[14], v[ac+i]))
          self.e(global_store_b16(addr=v[12:12], data=v[14], saddr=s[8:9], offset=0))
          if i<7: self.e(v_add_nc_u32_e32(v[12], self.N*4, v[12]))

  def emit_kernel_end(self):
    self.e(s_waitcnt(simm16=0)); self.e(s_sendmsg(simm16=3)); self.e(s_endpgm())

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

def lower_lds2_gemm_kernel(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA=0, PLRAB=0, LEANADDR=0, DSHALF=0, *, reg_layout=None, memory_layout=None, wait_policy=None, cadence=None, lifecycle_template=None):
  # P2/P3 (A3): parametric LDS-staged multi-wave GEMM. WAVES_M x WAVES_N wave32; each wave does WM x WN WMMA
  # tiles. BK = K-block depth (KT=BK/16 substeps). PAD = LDS row-pad bytes (bank-conflict avoidance). DBUF =
  # double-buffer LDS via unroll-by-2 (prefetch next block while computing current; removes the inner barrier).
  # PLRA = intra-block A-prefetch local read (KT==2 single-buffer only): prefetch substep1's A fragments into
  # the DEAD coop-load temp regs (CTA/CTB, register-lifetime overlap a la Tensile's pool) while substep0's
  # WMMAs run -> hides substep1's A ds_load latency behind compute. Partial PLR (A only; B' wouldn't fit 256).
  THREADS=WAVES_M*WAVES_N*32; BM=WAVES_M*WM*16; BN=WAVES_N*WN*16; KT=BK//16; CPR=BK//8; RSTRIDE=THREADS//CPR
  assert M%BM==0 and N%BN==0 and K%BK==0 and THREADS%CPR==0 and BM%RSTRIDE==0 and BN%RSTRIDE==0
  loadsA=BM//RSTRIDE; loadsB=BN//RSTRIDE; NBLK=K//BK
  lds_layout = (memory_layout or default_lds2_memory_layout(BM, BN, BK, PAD, DBUF)).validate()
  SA, SB, LDS_A, BUFSZ, NBUF = lds_layout.SA, lds_layout.SB, lds_layout.LDS_A, lds_layout.BUFSZ, lds_layout.NBUF
  layout = (reg_layout or env_lds2_reg_layout(WM, WN, loadsA, loadsB)).validate(WM, WN, loadsA, loadsB, PLRAB)
  FA, FB, ACCb, CTA, CTB, SCR, FB2 = layout.FA, layout.FB, layout.ACCb, layout.CTA, layout.CTB, layout.SCR, layout.FB2
  wait = (wait_policy or env_lds2_wait_policy()).validate()
  cadence = (cadence or default_lds2_cadence(DBUF)).validate(DBUF)
  lifecycle = (lifecycle_template or env_lds2_lifecycle_template(DBUF)).validate(DBUF)
  I=[]; Br=[]; lbl={}
  def e(i): I.append(i); return i
  def label(n): lbl[n]=sum(i.size() for i in I)
  def br(t): Br.append((len(I)-1,t))
  prim = LDS2PrimitiveEmitter(e, label, br, K=K, BK=BK, KT=KT, NBLK=NBLK, NL=(NBLK//2)-1,
    M=M, N=N, WAVES_N=WAVES_N, BM=BM, BN=BN, CPR=CPR, WM=WM, WN=WN, PLRA=PLRA, PLRAB=PLRAB, LEANADDR=LEANADDR, DSHALF=DSHALF, loadsA=loadsA, loadsB=loadsB,
    RSTRIDE=RSTRIDE, SA=SA, SB=SB, LDS_A=LDS_A, BUFSZ=BUFSZ, FA=FA, FB=FB, ACCb=ACCb, CTA=CTA, CTB=CTB,
    SCR=SCR, FB2=FB2, wait=wait)
  prim.emit_kernel_prologue()
  prim.emit_tile_setup()
  prim.zero_accumulators()
  prim.setup_leanaddr()
  prim.emit_lifecycle(lifecycle)
  prim.emit_epilogue()
  prim.emit_kernel_end()
  for idx,t in Br:
    off=(lbl[t]-sum(i.size() for i in I[:idx+1]))//4
    assert -32768<=off<=32767; I[idx].simm16=off
  return I

def build_gemm_lds2(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA=0, PLRAB=0, LEANADDR=0, DSHALF=0, *, reg_layout=None, memory_layout=None, wait_policy=None, cadence=None, lifecycle_template=None):
  return lower_lds2_gemm_kernel(M, N, K, WAVES_M, WAVES_N, WM, WN, BK, PAD, DBUF, PLRA, PLRAB, LEANADDR, DSHALF,
    reg_layout=reg_layout, memory_layout=memory_layout, wait_policy=wait_policy, cadence=cadence,
    lifecycle_template=lifecycle_template)

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
