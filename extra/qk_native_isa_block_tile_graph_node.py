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

def _isa_renderer():
  # reuse the AMD device's target (arch gfx1100) so the ELF matches the runtime; the device itself stays on HIP.
  return AMDISARenderer(Device["AMD"].renderer.target)

def compile_block_tile_isa(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Compile the block tile via AMDISARenderer. Returns (elf_bytes, ProgramInfo, group_segment_size)."""
  W = Hd + 2
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
  srcs = [Tensor.empty(Hq*S*W, dtype=dtypes.float32).uop.contiguous(),     # slot 0: out
          Tensor.empty(Hq*Hd, dtype=dtypes.float32).uop.contiguous(),      # slot 1: q (flat)
          Tensor.empty((2,1,Hkv,MAXC,Hd), dtype=dtypes.float32).uop.contiguous()]  # slot 2: cache (5D)
  phs = [UOp.placeholder_like(s, slot=i) for i,s in enumerate(srcs)]
  prg = to_program(fxn(*phs), _isa_renderer())
  elf = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  return elf, prg.arg, group_segment_fixed_size_from_elf(elf)

def native_isa_block_tile(out_t:Tensor, q_t:Tensor, cache_t:Tensor, Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc):
  """Run the native-ISA-compiled block tile via Ops.PROGRAM injection (DEV=AMD / HIP runtime). out_t:[Hq*S*(Hd+2)]
  fp32, q_t:[Hq*Hd] fp32, cache_t:[2,1,Hkv,MAXC,Hd] fp32. Returns the per-split partials tensor (slot 0)."""
  W = Hd + 2
  elf, pinfo, gseg = compile_block_tile_isa(Hd, Hq, Hkv, MAXC, L, S, Tc)
  def inject(*ph):
    return UOp(Ops.PROGRAM,
               src=(UOp.sink(*ph, arg=KernelInfo(name="native_block_tile",
                                                 estimates=Estimates(ops=Hq*MAXC*Hd*2, mem=Hkv*MAXC*Hd*4))),
                    UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=()), UOp(Ops.SOURCE, arg=""),
                    UOp(Ops.BINARY, arg=elf)),
               arg=ProgramInfo(name="native_block_tile", global_size=pinfo.global_size, local_size=pinfo.local_size,
                               vars=(), globals=(0,1,2), outs=(0,), ins=(1,2), aux=()))
  return out_t.custom_kernel(q_t, cache_t, fxn=inject)[0]
