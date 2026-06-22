#!/usr/bin/env python3
"""Stateful KV-cache append as an opaque graph node (Design A, docs/kv-cache-stateful-jit-capability-scope-20260622.md).

`kv_append_node(cache_kv, k, v, start_pos)` writes the current token's (K,V) into the persistent `cache_kv`
slice IN PLACE via an opaque RDNA3 custom_kernel (start_pos as a runtime scalar var, NOT baked into a captured
index), and returns `cache_kv.after(append_kernel)` -- the SAME buffer, ordered after the write, with NO
full-`max_context` materialization. This is the bounded core/JIT capability that the functional `.after()`-on-full-
buffer path (model.py:952) lacks; the microprobe (extra/qk_kv_append_microprobe.py) proved byte-correct symbolic
append + capture/replay with changing start_pos.

Strictly shape/arch-guarded: Qwen3-8B/gfx1100 decode shape (B==1, T==1, KvH==8, Hd==128, symbolic start_pos).
Default-off; the caller falls back to the canonical path on any mismatch/exception.
"""
from __future__ import annotations
import functools

KVH, HD = 8, 128
NELEM = 2 * KVH * HD   # one (K,V) append, B=1 T=1 = 2048 fp16

def _append_program(cache_f, src_f, start_pos_var, MAXC):
  """opaque RDNA3 kernel: 1024 threads, each stores one b32 (2 fp16) from src[tid] into cache at the start_pos slice.
  cache[2,1,KVH,MAXC,HD]; dest_fp16(tid)=kv*(KVH*MAXC*HD)+h*(MAXC*HD)+start_pos*HD+2*d2; kv=tid>>9,h=(tid>>6)&7,d2=tid&63."""
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.renderer import Estimates
  from tinygrad.runtime.autogen.amd.rdna3.ins import (s_load_b128, s_load_b32, s_waitcnt_lgkmcnt, s_mov_b32,
    s_lshl_b32, v_lshrrev_b32_e32, v_and_b32_e32, v_mul_lo_u32, v_lshlrev_b32_e32, v_add_nc_u32_e32,
    global_load_b32, s_waitcnt_vmcnt, global_store_b32, s_endpgm)
  from tinygrad.renderer.amd.dsl import s, v, NULL
  KVHMAXCHD, MAXCHD = KVH * MAXC * HD, MAXC * HD
  threads = UOp.special(1024, "lidx0")
  insts = [
    s_load_b128(s[4:7], s[0:1]),                 # s4:5=cache(out) ptr, s6:7=src(in) ptr
    s_load_b32(s[8], s[0:1], offset=0x10),       # s8 = start_pos (runtime scalar)
    s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
    s_mov_b32(s[10], KVHMAXCHD), s_mov_b32(s[11], MAXCHD),
    v_lshrrev_b32_e32(v[1], 9, v[0]),                                  # kv = tid>>9
    v_lshrrev_b32_e32(v[2], 6, v[0]), v_and_b32_e32(v[2], 7, v[2]),    # h  = (tid>>6)&7
    v_and_b32_e32(v[3], 63, v[0]),                                     # d2 = tid&63
    v_mul_lo_u32(v[4], v[1], s[10]),                                   # kv*KVHMAXCHD
    v_mul_lo_u32(v[5], v[2], s[11]), v_add_nc_u32_e32(v[4], v[4], v[5]),         # + h*MAXCHD
    s_lshl_b32(s[9], s[8], 7), v_add_nc_u32_e32(v[4], s[9], v[4]),               # + start_pos*HD
    v_lshlrev_b32_e32(v[6], 1, v[3]), v_add_nc_u32_e32(v[4], v[4], v[6]),        # + 2*d2 -> dest_fp16
    v_lshlrev_b32_e32(v[4], 1, v[4]),            # dest_byte = dest_fp16*2
    v_lshlrev_b32_e32(v[7], 2, v[0]),            # src_byte  = tid*4
    global_load_b32(v[8], v[7], saddr=s[6:7]),
    s_waitcnt_vmcnt(sdst=NULL, simm16=0),
    global_store_b32(addr=v[4], data=v[8], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(cache_f.base, src_f.base, start_pos_var, threads,
                  arg=KernelInfo("kv_append", estimates=Estimates(ops=NELEM, mem=NELEM * 4)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))))

def kv_append_node(cache_kv, k, v):
  """cache_kv:[2,B=1,KvH,MAXC,Hd] fp16 (persistent), k/v:[B,KvH,T=1,Hd] fp16 (post-rope). Returns cache_kv.after(append)
  reshaped to [2,1,KvH,MAXC,Hd] -- the in-place-updated cache ordered after the opaque write, no full-buffer copy."""
  from tinygrad import Tensor, UOp
  MAXC = cache_kv.shape[3]
  spvar = UOp.variable("start_pos", 0, MAXC - 1)   # unbound twin; value flows via var_vals (model carries start_pos)
  src = Tensor.stack(k, v).contiguous()            # [2,1,KvH,1,Hd] -> flat kv*1024+h*128+d (matches kernel src layout)
  # pass cache_kv as its realized BASE (not a .flatten() view): custom_kernel forces .contiguous(), which on a reshape
  # view materializes a FRESH buffer (breaking in-place persistence); on the realized contiguous base it is a no-op.
  out = Tensor.custom_kernel(cache_kv, src,
                             fxn=functools.partial(_append_program, start_pos_var=spvar, MAXC=MAXC))[0]
  # tell @function(precompile=True) the cache buffer was mutated (the opaque kernel's in-place write is NOT a tinygrad
  # STORE, so @function would otherwise treat cache_kv as read-only and the append would not persist across calls).
  # Mirror Tensor.assign's simple-assign branch: repoint the cache tensor's uop at the ordered-after node.
  cache_kv.uop = out.uop
  return out
