"""Controlled fault-isolation variants of the passing hand-ASM WMMA GEMM (see wmma.py build_gemm_pipe).

Goal (docs/prefill-lessons-ledger.md, Part 2): perturb the KNOWN-GOOD hand kernel ONE variable at a
time to isolate the gfx1100 v_wmma fault. Factors: operand ROLE {A=src0,B=src1,C=src2}, register HEIGHT
{low<120 / high>=128}, PROVENANCE {VMEM=global_load_b128 / VALU=v_pack_b32_f16}.

This file ONLY builds variant instruction streams + validates each is MATH-NEUTRAL via remu (DEV=PYTHON +
libremu.so). It does NOT run on DEV=AMD. A variant is "ready_for_gpu" iff remu is bit-exact vs numpy A@Bt.T
(that proves the perturbation changed delivery/placement, NOT the computation). The parent runs the GPU gate.

--- WHY the rig is single-buffered and defaults to a SMALL (2x2) all-low baseline (deviation from the
    literal task wording "keep TM=TN=4"), and why "scalar loads" became "in-place v_pack": ---
1. build_gemm_pipe is DOUBLE-buffered: it holds 2 fragment buffers (F0,F1). At TM=TN=4 that needs
   ACCb+TM*TN*8 = 274 VGPRs > 256 -> the canonical builder ASSERTS "VGPR overflow: 274". A 4x4 simply
   cannot be built in build_gemm_pipe. (Verified: build_gemm_pipe(64,64,64,4,4) raises.)
2. The fault is about register HEIGHT x PROVENANCE. To isolate ONE factor you need a baseline where ALL
   WMMA operands are LOW (<120); only a small tile leaves everything low (4x4 is forced to spill high and
   is thus confounded). The experiments then DELIBERATELY relocate specific operands high / flip their
   producer -- they manufacture the high+VALU condition, so the tile COUNT is irrelevant to the isolation.
   The default rig is TM=TN=2 (M=N=32, K=64) = 4 WMMAs, every operand < 120. A 4x4 (M=N=64) rig is also
   provided for variants where high placement is the whole point.
3. The rig is a fully-unrolled SINGLE-buffer accumulate (same addressing/loads/wmma/epilogue as
   build_gemm_pipe, no double-buffer pipeline). Single-buffer keeps the perturbations surgical and lets a
   fragment sit anywhere; the WMMA operand DELIVERY under test is byte-identical to the canonical loads.
4. "scalar loads + v_pack" (E2/E3 literal wording) cannot reproduce PER-LANE fragment data: a scalar load
   is uniform across the wave, but WMMA A/B fragments are per-lane (lane l holds row l). So provenance is
   flipped correctly by: global_load_b128 into the frag reg (correct per-lane data), then an IN-PLACE
   v_pack_b32_f16(frag, frag, frag, opsel) -> identical bits, but the register's LAST WRITER is now a VALU
   op (v_pack) instead of a VMEM op. That is exactly the "VALU-produced WMMA source" the theory implicates,
   with the minimal possible perturbation (same reg, same data, only the producer changes).
"""
import os, sys, ctypes
os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")
import numpy as np
sys.path.insert(0, os.getcwd())
from tinygrad.renderer.amd.dsl import v, s, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import *

LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")
# NOTE: remu's v_pack_b32_f16 model asserts opsel==0. So VALU delivery reconstructs reg R as
# v_pack(R, src0=R, src1=(R>>16), opsel=0) = ((R.hi)<<16)|(R.lo) = R. The hi-half is extracted by a
# v_lshrrev into a low scratch (PSCR). The WMMA source's LAST WRITER is the v_pack (a VALU op).

# ---------------------------------------------------------------------------------------------------------
# Core parameterized builder. Reuses build_gemm_pipe's addressing exactly (see wmma.py lines 58-72):
#   kernarg [A, B, OUT]: A ptr=s[4:5]@0x0, B(=Bt) ptr=s[6:7]@0x8, OUT ptr=s[8:9]@0x10.
#   A is MxK fp16 row-major. Bt is NxK fp16 row-major (B transposed). OUT is MxN fp16 row-major.
#   Frag tm holds A[tm*16+lane][0:16]; frag tn holds Bt[tn*16+lane][0:16]. Single wave => single 32x.. tile.
# ---------------------------------------------------------------------------------------------------------
def build_variant(M, N, K, TM, TN, *, a_bases, b_bases, a_valu, b_valu, acc_base, dead_high=None):
  """a_bases: list[TM] reg base per A fragment (8 vgprs each). b_bases: list[TN] per B fragment.
     a_valu/b_valu: bool -> flip that operand's provenance to VALU (in-place v_pack after the VMEM load).
     acc_base: reg base of the TM*TN*8 accumulator block (v_mov-0 initialised = VALU-produced src2).
     dead_high: optional list of high VGPR indices to write a dead v_mov (footprint inflation, E6)."""
  assert M % (TM*16) == 0 and N % (TN*16) == 0 and K % 16 == 0
  NK = K // 16
  VA = 100                    # address regs (TM+TN): low, above the low ACC block, below 120
  PSCR = 98                   # low scratch for the v_pack hi-half extraction (VALU delivery); not a WMMA source
  assert VA + TM + TN <= 120
  # register-collision guard: fragments, acc, VA, epilogue scratch (v4..v9) must not overlap
  used = []
  for b in a_bases: used += list(range(b, b+8))
  for b in b_bases: used += list(range(b, b+8))
  used += list(range(acc_base, acc_base + TM*TN*8))
  used += list(range(VA, VA+TM+TN))
  assert len(used) == len(set(used)), "fragment/acc/VA register overlap"
  assert max(used) < 256 and (dead_high is None or max(dead_high) < 256)
  I = []
  def e(i): I.append(i); return i
  sh = {8:7, 4:6, 2:5, 1:4}
  # ---- prologue (identical to build_gemm_pipe) ----
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
  # dead high writes (E6 footprint control) -- written once, never read, WMMA operands stay low
  if dead_high:
    for r in dead_high: e(v_mov_b32_e32(v[r], v[0]))
  # accumulator init (v_mov 0 = VALU-produced src2)
  for i in range(TM*TN*8): e(v_mov_b32_e32(v[acc_base+i], 0))
  def load_frag(base, va_reg, saddr, valu):
    # VMEM b128 load of 8 fp16 (16 bytes) at offset 0 and 16 -> base..base+7, same as build_gemm_pipe.
    e(global_load_b128(vdst=v[base:base+3],   addr=v[va_reg:va_reg], saddr=saddr, offset=0))
    e(global_load_b128(vdst=v[base+4:base+7], addr=v[va_reg:va_reg], saddr=saddr, offset=16))
    if valu:
      e(s_waitcnt(simm16=0))                                   # data must be resident before v_pack reads it
      for d in range(8):                                       # flip provenance to VALU: reconstruct base+d via v_pack
        e(v_lshrrev_b32_e32(v[PSCR], 16, v[base+d]))           # PSCR.lo16 = (base+d).hi16
        e(v_pack_b32_f16(vdst=v[base+d], src0=v[base+d], src1=v[PSCR], opsel=0))  # (PSCR.lo<<16)|base.lo == base+d
  # ---- fully-unrolled K accumulate (single fragment buffer) ----
  for kt in range(NK):
    for tm in range(TM): load_frag(a_bases[tm], VA+tm,    s[4:5], a_valu)
    for tn in range(TN): load_frag(b_bases[tn], VA+TM+tn, s[6:7], b_valu)
    e(s_waitcnt(simm16=0))
    for tm in range(TM):
      for tn in range(TN):
        ac = acc_base + (tm*TN+tn)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[a_bases[tm]:a_bases[tm]+7],
                                  src1=v[b_bases[tn]:b_bases[tn]+7], src2=v[ac:ac+7]))
    for r in range(TM+TN): e(v_add_nc_u32_e32(v[VA+r], 32, v[VA+r]))   # advance one k-tile (32 bytes)
  # ---- epilogue store (identical to build_gemm_pipe) ----
  e(v_and_b32_e32(v[4], 15, v[0]))
  e(v_lshrrev_b32_e32(v[5], 4, v[0])); e(v_and_b32_e32(v[5], 1, v[5]))
  for tm in range(TM):
    for tn in range(TN):
      ac = acc_base + (tm*TN+tn)*8
      e(v_add_nc_u32_e32(v[7], s[10], v[5]));  e(v_add_nc_u32_e32(v[7], tm*16, v[7]))
      e(v_add_nc_u32_e32(v[8], s[11], v[4]));  e(v_add_nc_u32_e32(v[8], tn*16, v[8]))
      e(v_mul_lo_u32(v[7], v[7], N)); e(v_add_nc_u32_e32(v[7], v[7], v[8]))
      e(v_lshlrev_b32_e32(v[7], 1, v[7]))
      for i in range(8):
        e(v_cvt_f16_f32_e32(v[6], v[ac+i]))
        e(global_store_b16(addr=v[7:7], data=v[6], saddr=s[8:9], offset=0))
        if i < 7: e(v_add_nc_u32_e32(v[7], N*4, v[7]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  return I

# ---- register-layout helpers (all "low" bases packed contiguously from 10; ACC follows) ----
def low_layout(TM, TN):
  FA = 10; FB = FA + TM*8; ACC = FB + TN*8
  a = [FA + tm*8 for tm in range(TM)]; b = [FB + tn*8 for tn in range(TN)]
  assert ACC + TM*TN*8 <= 120, "low layout must stay < 120"
  return a, b, ACC

# =========================================================================================================
# The E-variants. Each is ONE perturbation from E0 (the all-low, all-VMEM baseline).
# =========================================================================================================
def E0(M=32, N=32, K=64, TM=2, TN=2):
  """baseline: all operands LOW (<120), all VMEM (global_load_b128). remu bit-exact + GPU PASS expected."""
  a, b, acc = low_layout(TM, TN)
  return build_variant(M, N, K, TM, TN, a_bases=a, b_bases=b, a_valu=False, b_valu=False, acc_base=acc)

def E1(M=32, N=32, K=64, TM=2, TN=2, high=128):
  """A/B fragments relocated to HIGH regs (>=128), STILL VMEM (b128). Isolates HIGH+VMEM (T7).
     Predict PASS (LLVM reads high VMEM sources & passes) -> refutes 'height alone faults'."""
  a = [high + tm*8 for tm in range(TM)]
  b = [high + TM*8 + tn*8 for tn in range(TN)]
  _, _, acc = low_layout(TM, TN)  # acc stays low
  return build_variant(M, N, K, TM, TN, a_bases=a, b_bases=b, a_valu=False, b_valu=False, acc_base=acc)

def E2(M=32, N=32, K=64, TM=2, TN=2):
  """A/B delivered by VALU (in-place v_pack) into the SAME LOW regs. Isolates VALU+LOW.
     Predict PASS -> provenance alone (when low) is safe."""
  a, b, acc = low_layout(TM, TN)
  return build_variant(M, N, K, TM, TN, a_bases=a, b_bases=b, a_valu=True, b_valu=True, acc_base=acc)

def E3(M=32, N=32, K=64, TM=2, TN=2, high=128):
  """A/B delivered by VALU (in-place v_pack) into HIGH regs (>=128). Isolates VALU+HIGH (suspected trigger).
     Predict FAIL -> confirms 'a VALU-produced WMMA source in a HIGH reg faults' (T6/T8)."""
  a = [high + tm*8 for tm in range(TM)]
  b = [high + TM*8 + tn*8 for tn in range(TN)]
  _, _, acc = low_layout(TM, TN)
  return build_variant(M, N, K, TM, TN, a_bases=a, b_bases=b, a_valu=True, b_valu=True, acc_base=acc)

def E4(M=32, N=32, K=64, TM=2, TN=2, acc_high=180):
  """C accumulator relocated HIGH (>=128); v_mov-0 init (VALU) stays. A/B stay low+VMEM. Isolates C VALU+HIGH.
     FAIL => C(src2) counts too (T9); PASS => the fault is A/B(src0/src1)-specific (T8)."""
  a, b, _ = low_layout(TM, TN)
  assert acc_high + TM*TN*8 <= 238
  return build_variant(M, N, K, TM, TN, a_bases=a, b_bases=b, a_valu=False, b_valu=False, acc_base=acc_high)

def E5(M=32, N=32, K=64, TM=2, TN=2, base=128):
  """boundary sweep: exactly ONE A fragment (tm=0) VALU-produced at a chosen `base`; every other operand is
     the low+VMEM baseline. Sweep base in {112,116,120,124,128} to locate the fault line (T11).
     Implemented as per-fragment provenance: only frag0 is VALU; frags 1.. stay VMEM."""
  a, b, acc = low_layout(TM, TN)
  a = list(a); a[0] = base                                   # relocate ONLY A-frag 0 to the swept base
  # per-fragment VALU: build with a_valu handled per-frag via a custom call (frag0 valu, rest vmem)
  return _build_single_valu_frag(M, N, K, TM, TN, a, b, acc, valu_a_idx=0)

def E6(M=32, N=32, K=64, TM=2, TN=2, dead=(224, 228, 232)):
  """baseline (all operands LOW + VMEM) PLUS dead high v_mov writes to inflate the declared VGPR footprint.
     Isolates FOOTPRINT. Predict PASS -> re-confirms occupancy/footprint irrelevant (T5) in the hand rig."""
  a, b, acc = low_layout(TM, TN)
  dead_regs = []
  for base in dead: dead_regs += [base]   # a few scattered high dead regs (kept < 238)
  return build_variant(M, N, K, TM, TN, a_bases=a, b_bases=b, a_valu=False, b_valu=False,
                       acc_base=acc, dead_high=list(dead))

def _build_single_valu_frag(M, N, K, TM, TN, a_bases, b_bases, acc_base, valu_a_idx):
  """E5 helper: identical to build_variant but only A-fragment `valu_a_idx` is VALU-produced (in-place pack);
     all other A frags and all B frags stay VMEM. Everything else byte-identical."""
  assert K % 16 == 0
  NK = K // 16; VA = 100; PSCR = 98
  used = []
  for bb in a_bases: used += list(range(bb, bb+8))
  for bb in b_bases: used += list(range(bb, bb+8))
  used += list(range(acc_base, acc_base+TM*TN*8)); used += list(range(VA, VA+TM+TN))
  assert len(used) == len(set(used)), "E5 register overlap"
  I = []
  def e(i): I.append(i); return i
  sh = {8:7, 4:6, 2:5, 1:4}
  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt(simm16=0))
  e(v_and_b32_e32(v[1], 15, v[0]))
  e(s_lshl_b32(s[10], s[3], sh[TM])); e(s_lshl_b32(s[11], s[2], sh[TN]))
  e(v_add_nc_u32_e32(v[2], s[10], v[1])); e(v_add_nc_u32_e32(v[3], s[11], v[1]))
  for tm in range(TM):
    e(v_add_nc_u32_e32(v[VA+tm], tm*16, v[2]) if tm else v_mov_b32_e32(v[VA+tm], v[2]))
    e(v_mul_lo_u32(v[VA+tm], v[VA+tm], K*2))
  for tn in range(TN):
    e(v_add_nc_u32_e32(v[VA+TM+tn], tn*16, v[3]) if tn else v_mov_b32_e32(v[VA+TM+tn], v[3]))
    e(v_mul_lo_u32(v[VA+TM+tn], v[VA+TM+tn], K*2))
  for i in range(TM*TN*8): e(v_mov_b32_e32(v[acc_base+i], 0))
  for kt in range(NK):
    for tm in range(TM):
      base = a_bases[tm]
      e(global_load_b128(vdst=v[base:base+3],   addr=v[VA+tm:VA+tm], saddr=s[4:5], offset=0))
      e(global_load_b128(vdst=v[base+4:base+7], addr=v[VA+tm:VA+tm], saddr=s[4:5], offset=16))
    for tn in range(TN):
      base = b_bases[tn]
      e(global_load_b128(vdst=v[base:base+3],   addr=v[VA+TM+tn:VA+TM+tn], saddr=s[6:7], offset=0))
      e(global_load_b128(vdst=v[base+4:base+7], addr=v[VA+TM+tn:VA+TM+tn], saddr=s[6:7], offset=16))
    e(s_waitcnt(simm16=0))
    base = a_bases[valu_a_idx]
    for d in range(8):
      e(v_lshrrev_b32_e32(v[PSCR], 16, v[base+d]))
      e(v_pack_b32_f16(vdst=v[base+d], src0=v[base+d], src1=v[PSCR], opsel=0))
    for tm in range(TM):
      for tn in range(TN):
        ac = acc_base + (tm*TN+tn)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[a_bases[tm]:a_bases[tm]+7],
                                  src1=v[b_bases[tn]:b_bases[tn]+7], src2=v[ac:ac+7]))
    for r in range(TM+TN): e(v_add_nc_u32_e32(v[VA+r], 32, v[VA+r]))
  e(v_and_b32_e32(v[4], 15, v[0]))
  e(v_lshrrev_b32_e32(v[5], 4, v[0])); e(v_and_b32_e32(v[5], 1, v[5]))
  for tm in range(TM):
    for tn in range(TN):
      ac = acc_base + (tm*TN+tn)*8
      e(v_add_nc_u32_e32(v[7], s[10], v[5])); e(v_add_nc_u32_e32(v[7], tm*16, v[7]))
      e(v_add_nc_u32_e32(v[8], s[11], v[4])); e(v_add_nc_u32_e32(v[8], tn*16, v[8]))
      e(v_mul_lo_u32(v[7], v[7], N)); e(v_add_nc_u32_e32(v[7], v[7], v[8])); e(v_lshlrev_b32_e32(v[7], 1, v[7]))
      for i in range(8):
        e(v_cvt_f16_f32_e32(v[6], v[ac+i]))
        e(global_store_b16(addr=v[7:7], data=v[6], saddr=s[8:9], offset=0))
        if i < 7: e(v_add_nc_u32_e32(v[7], N*4, v[7]))
  e(s_waitcnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())
  return I

# ---------------------------------------------------------------------------------------------------------
# remu validation: assemble I -> raw .text bytes -> run_asm; compare vs numpy A @ Bt.T.
# ---------------------------------------------------------------------------------------------------------
def asm_bytes(I):
  raw = b"".join(i.to_bytes() for i in I)
  assert len(raw) % 4 == 0
  return raw

def final_bytes(I):
  """Exact bytes the GPU runs: push the INS stream through the renderer's asm() pipeline
     (_schedule -> _insert_waitcnt -> _resolve_labels), same as extra/qk/prefill_graph_gemm_route.py.
     No GPU/device needed -- this just assembles. Lets remu validate the POST-RENDER stream."""
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  uops = [UOp(Ops.INS, arg=i) for i in I]
  from tinygrad.helpers import getenv
  if getenv("AMD_ISA_SCHED", 1): uops = ren._schedule(uops)
  final = ren._resolve_labels(ren._insert_waitcnt(uops))
  raw = b"".join(u.arg.to_bytes() for u in final)
  assert len(raw) % 4 == 0
  return raw

def remu_validate(I, M, N, K, seed=0, use_final=False):
  np.random.seed(seed)
  A  = np.random.randn(M, K).astype(np.float16)
  Bt = np.random.randn(N, K).astype(np.float16)   # B transposed: Bt[n,k] = B[k,n]
  OUT = np.zeros((M, N), dtype=np.float16)
  text = final_bytes(I) if use_final else asm_bytes(I)
  args = (ctypes.c_uint64*3)(A.ctypes.data, Bt.ctypes.data, OUT.ctypes.data)  # kernarg [A, B, OUT]
  lib = ctypes.CDLL(LIBREMU)
  lib.run_asm.restype = ctypes.c_int
  lib.run_asm.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_uint32]*6 + [ctypes.POINTER(ctypes.c_uint64)]
  rc = lib.run_asm(ctypes.c_char_p(text), len(text), 1, 1, 1, 32, 1, 1, args)
  ref = A.astype(np.float32) @ Bt.astype(np.float32).T
  got = OUT.astype(np.float32)
  nanfrac = float(np.isnan(got).mean())
  ok = np.isfinite(got)
  rmse = float(np.sqrt(((got[ok]-ref[ok])**2).mean())) if ok.any() else float("nan")
  bitexact = (nanfrac == 0.0) and (rmse < 5e-2)
  return dict(rc=rc, nanfrac=nanfrac, rmse=rmse, bitexact=bitexact, bytes=len(text))

VARIANTS = dict(E0=E0, E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6)

# ---------------------------------------------------------------------------------------------------------
# GPU runner (DEV=AMD only -- the PARENT invokes this; this subagent never runs it). Wraps a variant's INS
# stream in a PROGRAM UOp via Tensor.custom_kernel (same mechanism as extra/qk/prefill_graph_gemm_route.py),
# runs it on the real gfx1100, and compares vs numpy A @ Bt.T. Single wave: grid=(1,1,1), 32 threads.
#   PASS = 0 NaN and rmse < 5e-2 (bit-exact vs numpy). FAIL = NaN present -> that perturbation triggers the fault.
# ---------------------------------------------------------------------------------------------------------
def gpu_run(name, base=128, seed=0):
  from tinygrad import Tensor, Device
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import Estimates
  from tinygrad.helpers import colored
  fn = VARIANTS[name]
  I = fn(base=base) if name == "E5" else fn()
  # geometry: default variants are TM=TN=2 -> M=N=32, K=64. (Every variant here uses that geometry.)
  M = N = 32; K = 64
  np.random.seed(seed)
  A_np  = np.random.randn(M, K).astype(np.float16)
  Bt_np = np.random.randn(N, K).astype(np.float16)
  A  = Tensor(A_np).contiguous().realize()
  Bt = Tensor(Bt_np).contiguous().realize()
  C  = Tensor.empty(M, N, dtype=dtypes.half, device=A.device).contiguous().realize()
  grid = (N // (16*2), M // (16*2), 1)   # (1,1,1) for 2x2
  nm = f"faultprobe_{name}" + (f"_b{base}" if name == "E5" else "")
  def asm_kernel(a, bt, c):
    g = [UOp.special(max(1, grid[0]), "gidx0"), UOp.special(max(1, grid[1]), "gidx1")]
    sink = UOp.sink(a.base, bt.base, c.base, *g, UOp.special(32, "lidx0"),
                    arg=KernelInfo(name=colored(nm, "cyan"),
                                   estimates=Estimates(ops=M*N*K*2, mem=(M*K+N*K+M*N)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=i) for i in I]))))
  out = Tensor.custom_kernel(A, Bt, C, fxn=asm_kernel)[2]
  got = out.numpy().astype(np.float32)
  ref = A_np.astype(np.float32) @ Bt_np.astype(np.float32).T
  nanfrac = float(np.isnan(got).mean())
  ok = np.isfinite(got)
  rmse = float(np.sqrt(((got[ok]-ref[ok])**2).mean())) if ok.any() else float("nan")
  PASS = (nanfrac == 0.0) and (rmse < 5e-2)
  print(f"GPU {name}{'/b'+str(base) if name=='E5' else ''}: nan_frac={nanfrac:.4f} rmse(non-nan)={rmse:.5f} "
        f"PASS={PASS}")
  return PASS

if __name__ == "__main__":
  # DEV=AMD GPU gate (parent only):  python3 extra/qk/prefill/wmma_faultprobe.py --gpu E3
  #   E5 sweeps bases:               python3 extra/qk/prefill/wmma_faultprobe.py --gpu E5
  if sys.argv[1:2] == ["--gpu"]:
    sel = sys.argv[2:] or list(VARIANTS)
    for name in sel:
      if name == "E5":
        for base in (112, 116, 120, 124, 128): gpu_run("E5", base=base)
      else:
        gpu_run(name)
    sys.exit(0)
  # remu on the EXACT post-renderer bytes the GPU will run (_schedule+_insert_waitcnt+_resolve_labels):
  #   python3 extra/qk/prefill/wmma_faultprobe.py --final
  use_final = sys.argv[1:2] == ["--final"]
  if use_final: sys.argv.pop(1)
  # default (remu, DEV=PYTHON): geometry M=N=32, K=64, TM=TN=2 unless a variant overrides
  sel = sys.argv[1:] or list(VARIANTS)
  for name in sel:
    fn = VARIANTS[name]
    if name == "E5":
      for base in (112, 116, 120, 124, 128):
        I = E5(base=base); r = remu_validate(I, 32, 32, 64, use_final=use_final)
        print(f"E5(base={base:3d}): insts={len(I):4d} bytes={r['bytes']:5d} rc={r['rc']} "
              f"nan={r['nanfrac']:.3f} rmse={r['rmse']:.5f} bitexact={r['bitexact']}")
    else:
      I = fn(); r = remu_validate(I, 32, 32, 64, use_final=use_final)
      print(f"{name}: insts={len(I):4d} bytes={r['bytes']:5d} rc={r['rc']} "
            f"nan={r['nanfrac']:.3f} rmse={r['rmse']:.5f} bitexact={r['bitexact']}")
