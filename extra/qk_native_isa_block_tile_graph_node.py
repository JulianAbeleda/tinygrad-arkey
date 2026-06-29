"""Phase H route-binding primitive: the generated decode block tile compiled by the NATIVE AMD ISA backend
(AMDISARenderer) and injected into a graph as a precompiled Ops.PROGRAM node -- the native analogue of
extra/qk_owned_flash_decode_graph_node.py (which injects a hand-AMDGCN ELF).

KEYSTONE (proven): an ELF produced by AMDISARenderer loads + runs in the HIP (DEV=AMD) device context. So the
attention tile can be the native candidate while the rest of the model stays on HIP -- no whole-model DEV=AMD:ISA
(which is blocked by broad model op coverage, e.g. the 64-bit RNG/sampling path's ulong->float).

Mechanism: build the tile UOp AST (the qk_flash_decode fxn) with placeholders, compile via to_program(ast,
AMDISARenderer(target)) to get the ELF BINARY + ProgramInfo, then wrap as an Ops.PROGRAM (BINARY present + explicit
ProgramInfo -> codegen treated as complete) and hand it to Tensor.custom_kernel.

Scope: FIXED context (concrete Tc). Variable in-model context (start_pos runtime var) needs DEFINE_VAR coverage in
AMDISARenderer (isel_var is scaffolded but has an unresolved regalloc desync) -- the remaining Phase H work.
"""
from __future__ import annotations
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo
from tinygrad.renderer import Estimates
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.codegen import to_program
from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf
from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel

from tinygrad.helpers import Context
_ISA_REN = None
def _isa_renderer():
  # reuse the AMD device's target (arch gfx1100) so the ELF matches the runtime; the device itself stays on HIP.
  # Cached + ALLOW_DEVICE_USAGE-wrapped: the injector runs during the model's forward graph build (device usage
  # disallowed), but reading the already-initialized device's renderer target allocates nothing.
  global _ISA_REN
  if _ISA_REN is None:
    with Context(ALLOW_DEVICE_USAGE=1): _ISA_REN = AMDISARenderer(Device["AMD"].renderer.target)
  return _ISA_REN

import functools

@functools.lru_cache(maxsize=None)
def _compile(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Compile the block tile via AMDISARenderer (cached: the runtime var keeps Tc symbolic -> one compile).
  Tc may be an int (fixed context, keystone) or a UOp expression (e.g. vsp+T, runtime start_pos). Returns
  (elf_bytes, global_size, local_size, vars, group_segment_size)."""
  W = Hd + 2
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
  # build PARAM placeholders directly (NOT Tensor.empty) -- the injector runs during the model's forward graph build,
  # where device usage is disallowed; UOp.placeholder is a pure UOp (no buffer), and to_program only renders.
  phs = [UOp.placeholder((Hq*S*W,), dtypes.float32, 0),          # slot 0: out (partials)
         UOp.placeholder((Hq*Hd,), dtypes.float32, 1),           # slot 1: q (flat)
         UOp.placeholder((2,1,Hkv,MAXC,Hd), dtypes.float32, 2)]  # slot 2: cache (5D)
  prg = to_program(fxn(*phs), _isa_renderer())
  elf = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  p = prg.arg
  return elf, p.global_size, p.local_size, p.vars, group_segment_fixed_size_from_elf(elf)

def compile_block_tile_isa(Hd, Hq, Hkv, MAXC, L, S, Tc):
  elf, g, b, v, gseg = _compile(Hd, Hq, Hkv, MAXC, L, S, Tc)
  return elf, g, b, gseg   # back-compat (keystone gate)

def native_isa_block_tile(out_t:Tensor, q_t:Tensor, cache_t:Tensor, Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Run the native-ISA-compiled block tile via Ops.PROGRAM injection (DEV=AMD / HIP runtime). out_t:[Hq*S*(Hd+2)]
  fp32 partials, q_t:[Hq*Hd] fp32, cache_t:[2,1,Hkv,MAXC,Hd] fp32. Tc int (fixed) or UOp (runtime start_pos var).
  Returns the per-split partials tensor (slot 0). gmax + combine stay on HIP (caller)."""
  elf, gsize, lsize, varz, gseg = _compile(Hd, Hq, Hkv, MAXC, L, S, Tc)
  def inject(*ph):
    return UOp(Ops.PROGRAM,
               src=(UOp.sink(*ph, *varz, arg=KernelInfo(name="native_block_tile",
                                                        estimates=Estimates(ops=Hq*MAXC*Hd*2, mem=Hkv*MAXC*Hd*4))),
                    UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=""),
                    UOp(Ops.BINARY, arg=elf)),
               arg=ProgramInfo(name="native_block_tile", global_size=gsize, local_size=lsize,
                               vars=varz, globals=(0,1,2), outs=(0,), ins=(1,2), aux=()))
  return out_t.custom_kernel(q_t, cache_t, fxn=inject)[0]
