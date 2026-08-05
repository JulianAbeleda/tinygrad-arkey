"""Research-only exact Q4_K GEMV with group-factorized fp16 activation math.

This is intentionally not another vector spelling of the installed G3 body
and not a Q8 approximation.  Four 8-lane subgroups own four Q4_K blocks at a
time; each lane owns one complete 32-value quant group.  The exact dequant dot

  sum_i ((d*sc*q_i - dmin*mn) * x_i)

is factorized as

  (d*sc) * sum_i(q_i*x_i) - (dmin*mn) * sum_i(x_i).

The production packed Q4_K words and fp16 activation are consumed directly.
Only fp32 association changes.  Runtime policy never imports this module.
"""
from __future__ import annotations

import numpy as np

from tinygrad import dtypes
from tinygrad.codegen.late.warp_reduce import _staged_shfl, _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import Q4K_WORDS_PER_BLOCK, _f16_word
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

QK = 256


def emit_q4k_exact_group_factorized(rows:int, k:int):
  """One warp/output, four 8-lane block subgroups, exact direct-fp16 ABI."""
  if rows <= 0 or k <= 0 or k % QK or (k//QK) % 4:
    raise ValueError("factorized Q4_K requires positive rows and K/256 divisible by four")
  kblocks, blocks_per_group = k//QK, (k//QK)//4

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row,lane=UOp.special(rows,"gidx0"),UOp.special(32,"lidx0")
    block_group,group=lane//8,lane%8
    lblk=UOp.range(blocks_per_group,0,axis_type=AxisType.LOOP)
    block=block_group*blocks_per_group+lblk
    base=(row*kblocks+block)*Q4K_WORDS_PER_BLOCK

    # Dynamic group extraction means each lane materializes only its own scale
    # and minimum. The four header words are shared by the adjacent group lanes
    # through the read-only cache; no eight-arm static expression tree exists.
    w0,w1,w2,w3=words[base],words[base+1],words[base+2],words[base+3]
    d,dmin=_f16_word(w0,False),_f16_word(w0,True)
    g4=group%4
    b1=w1.rshift(g4*8).bitwise_and(0xff)
    b2=w2.rshift(g4*8).bitwise_and(0xff)
    hb=w3.rshift(g4*8).bitwise_and(0xff)
    sc=(group<4).where(b1.bitwise_and(63),hb.bitwise_and(15).bitwise_or(b1.rshift(6).lshift(4)))
    mn=(group<4).where(b2.bitwise_and(63),hb.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))

    qx=UOp.const(dtypes.float32,0.0)
    sx=UOp.const(dtypes.float32,0.0)
    for word in range(8):
      qpack=words[base+4+(group//2)*8+word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      for nib in range(4):
        xv=x[block*QK+group*32+word*4+nib].cast(dtypes.float32)
        q=qpack.rshift(nib*8).bitwise_and(15).cast(dtypes.float32)
        qx=qx+q*xv
        sx=sx+xv
    subtotal=d*sc.cast(dtypes.float32)*qx-dmin*mn.cast(dtypes.float32)*sx
    for slot,off in enumerate((4,2,1),90): subtotal=subtotal+_staged_shfl(subtotal,off,lane,slot)
    owned=(lane%8).eq(0).where(subtotal,UOp.const(dtypes.float32,0.0))

    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0))
    acc=acc.after(acc[0].store(acc.after(lblk)[0]+owned).end(lblk))
    total=_warp_reduce_sum_staged(acc[0],lane,32,slot_base=100)
    return out[row].store(total,lane.eq(0)).sink(
      arg=KernelInfo(name=f"q4k_exact_group_factorized_{rows}_{k}",opts_to_apply=()))
  return kernel


def emit_q4k_exact_four_warp(rows:int, k:int, block_count:UOp|None=None):
  """Four warps/output with direct-fp16, exact Q4_K group factorization.

  This copies the *observed* MMVQ ownership geometry without copying its Q8_1
  representation: a warp owns a disjoint K stripe, a four-lane subgroup owns
  one quant group, and each lane consumes two packed four-value words.  The
  four warp totals rendezvous in 16 bytes of shared memory and one thread
  writes the final row, so the included-cost candidate remains one kernel.
  """
  if rows <= 0 or k <= 0 or k % QK or (k//QK) % 4:
    raise ValueError("four-warp Q4_K requires positive rows and K/256 divisible by four")
  kblocks, blocks_per_warp = k//QK, (k//QK)//4
  block_count = UOp.const(dtypes.int, blocks_per_warp) if block_count is None else block_count

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row,lid=UOp.special(rows,"gidx0"),UOp.special(128,"lidx0")
    warp,lane=lid//32,lid%32
    group,word_base=lane//4,(lane%4)*2
    lblk=UOp.range(block_count,0,axis_type=AxisType.LOOP)
    block=warp*blocks_per_warp+lblk
    base=(row*kblocks+block)*Q4K_WORDS_PER_BLOCK

    w0,w1,w2,w3=words[base],words[base+1],words[base+2],words[base+3]
    d,dmin=_f16_word(w0,False),_f16_word(w0,True)
    g4=group%4
    b1=w1.rshift(g4*8).bitwise_and(0xff)
    b2=w2.rshift(g4*8).bitwise_and(0xff)
    hb=w3.rshift(g4*8).bitwise_and(0xff)
    sc=(group<4).where(b1.bitwise_and(63),hb.bitwise_and(15).bitwise_or(b1.rshift(6).lshift(4)))
    mn=(group<4).where(b2.bitwise_and(63),hb.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))

    qx=UOp.const(dtypes.float32,0.0)
    sx=UOp.const(dtypes.float32,0.0)
    for word_slot in range(2):
      word=word_base+word_slot
      qpack=words[base+4+(group//2)*8+word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      for nib in range(4):
        xv=x[block*QK+group*32+word*4+nib].cast(dtypes.float32)
        q=qpack.rshift(nib*8).bitwise_and(15).cast(dtypes.float32)
        qx=qx+q*xv
        sx=sx+xv
    subgroup=d*sc.cast(dtypes.float32)*qx-dmin*mn.cast(dtypes.float32)*sx
    for slot,off in enumerate((2,1),90): subgroup=subgroup+_staged_shfl(subgroup,off,lane,slot)
    owned=(lane%4).eq(0).where(subgroup,UOp.const(dtypes.float32,0.0))

    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0))
    acc=acc.after(acc[0].store(acc.after(lblk)[0]+owned).end(lblk))
    warp_total=_warp_reduce_sum_staged(acc[0],lane,32,slot_base=100)
    smem=UOp.placeholder((4,),dtypes.float32,230,addrspace=AddrSpace.LOCAL)
    published=smem[warp].store(warp_total,lane.eq(0))
    ready=UOp.barrier(UOp.group(published))
    total=UOp.const(dtypes.float32,0.0)
    for wi in range(4): total=total+smem.after(ready)[wi]
    return out[row].store(total,lid.eq(0)).sink(
      arg=KernelInfo(name=f"q4k_exact_four_warp_{rows}_{k}",opts_to_apply=()))
  return kernel


def emit_q4k_exact_four_warp_runtime_blocks(rows:int, k:int):
  return emit_q4k_exact_four_warp(rows,k,UOp.variable("q4k_exact_blocks",1,(k//QK)//4))


def unpack_q4k_row(words:np.ndarray, k:int) -> np.ndarray:
  """Independent byte-layout oracle: dequantize one packed production row."""
  if words.dtype != np.uint32 or words.size != (k//QK)*Q4K_WORDS_PER_BLOCK: raise ValueError("bad packed row")
  raw=words.astype("<u4",copy=False).view(np.uint8).reshape(k//QK,144)
  out=np.empty(k,dtype=np.float32)
  for b,rb in enumerate(raw):
    d=np.frombuffer(rb[0:2].tobytes(),dtype="<f2")[0].astype(np.float32)
    dm=np.frombuffer(rb[2:4].tobytes(),dtype="<f2")[0].astype(np.float32)
    for g in range(8):
      if g < 4:
        sc=int(rb[4+g])&63; mn=int(rb[8+g])&63
      else:
        sc=(int(rb[12+g-4])&15)|((int(rb[4+g-4])>>6)<<4)
        mn=(int(rb[12+g-4])>>4)|((int(rb[8+g-4])>>6)<<4)
      for p in range(32):
        q=(int(rb[16+(g//2)*32+p])>>((g%2)*4))&15
        out[b*QK+g*32+p]=d*np.float32(sc*q)-dm*np.float32(mn)
  return out


def oracle_gemv(words:np.ndarray, x:np.ndarray, rows:int, k:int) -> np.ndarray:
  """Independent fp64 accumulation oracle for packed Q4_K x fp16."""
  if words.shape != (rows*(k//QK)*Q4K_WORDS_PER_BLOCK,) or x.shape != (k,) or x.dtype != np.float16:
    raise ValueError("oracle shape/dtype mismatch")
  stride=(k//QK)*Q4K_WORDS_PER_BLOCK
  return np.asarray([np.dot(unpack_q4k_row(words[r*stride:(r+1)*stride],k).astype(np.float64),x.astype(np.float64))
                     for r in range(rows)],dtype=np.float32)


__all__=["emit_q4k_exact_group_factorized","emit_q4k_exact_four_warp","emit_q4k_exact_four_warp_runtime_blocks",
         "unpack_q4k_row","oracle_gemv"]
