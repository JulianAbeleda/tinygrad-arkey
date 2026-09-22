"""Closed-default P3b FFN W1/W3 -> Q8_1 cooperative producer.

This is intentionally a research emitter, not a route.  The important change
from the landed W1/W3 kernel is ownership: one CTA owns 32 consecutive hidden
rows, which is exactly one Q8_1 packet.  Eight lanes compute each row, then
the 32 row owners publish fp16-rounded GLU values to a 32-half shared staging
array.  The first warp makes the packet directly from that array; no 12288-row
activation buffer is materialized.

The GEMV inner body is a runtime LOOP over Q4 logical words.  This avoids the
old ``eight groups x eight words`` source unroll trap.  It deliberately has no
promotion hook: source/resource and numerical gates must precede any GPU run.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import ctypes
import pathlib
import numpy as np

from tinygrad import dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import WARP, _staged_shfl, _warp_reduce_sum_staged, warp_reduce_max
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGateUpLaneMap, Q4K_WORDS_PER_BLOCK, _f16_word,
  _q4k_block_dot_packed_load, _q4k_group_params, _silu_uop)
from tinygrad.llm.q4k_ffn_down_mmvq import emit_ffn_w1w3_q8_scalar_packet
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

K, ROWS, PACK = 4096, 12288, 32
LANES_PER_ROW, ROWS_PER_CTA = 8, 32
THREADS, Q4_BLOCKS, WORDS_PER_Q4_BLOCK = LANES_PER_ROW*ROWS_PER_CTA, K//256, 64
Q8_PACKETS, Q8_METADATA_BASE = ROWS//PACK*8, ROWS//PACK*8
DOWN_ROWS, DOWN_K, DOWN_Q8_METADATA_BASE = 4096, ROWS, ROWS//4


@dataclass(frozen=True)
class FFNQ8CooperativePlan:
  """The exact ownership and ABI contract for the experimental producer."""
  rows: int = ROWS
  k: int = K
  rows_per_cta: int = ROWS_PER_CTA
  lanes_per_row: int = LANES_PER_ROW

  def validate(self) -> None:
    if (self.rows, self.k, self.rows_per_cta, self.lanes_per_row) != (ROWS, K, ROWS_PER_CTA, LANES_PER_ROW):
      raise ValueError("P3b is fixed to Qwen Q4/Q4/Q4 FFN 12288x4096 and 32-row packets")
    if self.rows_per_cta % PACK: raise ValueError("one CTA must own integral Q8_1 packets")
    if self.rows_per_cta*self.lanes_per_row != THREADS: raise ValueError("unexpected CTA ownership")

  @property
  def grid_x(self) -> int: return self.rows//self.rows_per_cta
  @property
  def output_words(self) -> int: return Q8_METADATA_BASE + self.grid_x

  def topology(self) -> dict:
    # Current direct FFN-down requires a materializing fp16 input boundary in
    # addition to the landed W1/W3 producer and the down consumer.  The new
    # chain is producer + Q8 consumer. This is a structural, not timing claim.
    return {"baseline_programs_min": 3, "candidate_programs": 2,
            "strictly_smaller": True, "baseline": ["w1w3", "fp16_boundary", "ffn_down"],
            "candidate": ["cooperative_w1w3_q8_provider", "q8_ffn_down_consumer"]}


def _pack4(vs: tuple[UOp, UOp, UOp, UOp]) -> UOp:
  out=UOp.const(dtypes.uint32,0)
  for i,v in enumerate(vs): out=out.bitwise_or(v.cast(dtypes.uint8).cast(dtypes.uint32).lshift(8*i))
  return out


def _q4_params_dynamic(w0:UOp,w1:UOp,w2:UOp,w3:UOp,group:UOp):
  """Q4_K header decode with runtime group ownership, not an eight-arm tree."""
  d,dmin=_f16_word(w0,False),_f16_word(w0,True)
  g4=group%4
  b1=w1.rshift(g4*8).bitwise_and(0xff); b2=w2.rshift(g4*8).bitwise_and(0xff)
  # Groups 4..7 use one full byte each from header word 3: low nibble is
  # scale and high nibble is minimum. The rejected first spelling advanced by
  # four bits and then masked away the minimum nibble.
  highbyte=w3.rshift(g4*8).bitwise_and(0xff)
  sc=(group<4).where(b1.bitwise_and(63),highbyte.bitwise_and(15).bitwise_or(b1.rshift(6).lshift(4)))
  mn=(group<4).where(b2.bitwise_and(63),highbyte.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))
  return d,dmin,sc,mn


def emit_ffn_w1w3_q8_cooperative(plan:FFNQ8CooperativePlan=FFNQ8CooperativePlan()):
  """Emit one 32-row packet provider in a 256-thread CTA.

  The output ABI is the existing shared-Q8 consumer-private layout: 3072
  uint32 int8x4 payload words followed by 384 uint32 ``d|s`` metadata words.
  ``s`` is the fp16 raw sum of the 32 fp16-rounded GLU values, matching the
  llama-CUDA Q8_1 producer convention.
  """
  plan.validate()
  def kernel(out:UOp, gate_words:UOp, up_words:UOp, x:UOp) -> UOp:
    packet=UOp.special(plan.grid_x,"gidx0"); lid=UOp.special(THREADS,"lidx0")
    row_local,lane=lid//LANES_PER_ROW,lid%LANES_PER_ROW
    row=packet*ROWS_PER_CTA+row_local
    # Each lane owns two Q4_K blocks.  Iteration is a logical (group,word)
    # pair; the nibble loop makes the physical Q4 load explicit while keeping
    # source size bounded irrespective of K's group count.
    logical=UOp.range(2*WORDS_PER_Q4_BLOCK,2,axis_type=AxisType.LOOP)
    block=lane*2+logical//WORDS_PER_Q4_BLOCK
    rem=logical%WORDS_PER_Q4_BLOCK; group=rem//8; word=rem%8
    base=(row*Q4_BLOCKS+block)*Q4K_WORDS_PER_BLOCK
    w0g,w1g,w2g,w3g=gate_words[base],gate_words[base+1],gate_words[base+2],gate_words[base+3]
    w0u,w1u,w2u,w3u=up_words[base],up_words[base+1],up_words[base+2],up_words[base+3]
    dg,dmg,scg,mng=_q4_params_dynamic(w0g,w1g,w2g,w3g,group)
    du,dmu,scu,mnu=_q4_params_dynamic(w0u,w1u,w2u,w3u,group)
    qg=gate_words[base+4+(group//2)*8+word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
    qu=up_words[base+4+(group//2)*8+word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
    nib=UOp.range(4,3,axis_type=AxisType.LOOP)
    qgv=qg.rshift(nib*8).bitwise_and(15).cast(dtypes.float32); quv=qu.rshift(nib*8).bitwise_and(15).cast(dtypes.float32)
    xv=x[block*256+group*32+word*4+nib].cast(dtypes.float32)
    cg=(dg*scg.cast(dtypes.float32)*qgv-dmg*mng.cast(dtypes.float32))*xv
    cu=(du*scu.cast(dtypes.float32)*quv-dmu*mnu.cast(dtypes.float32))*xv
    ag=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); au=UOp.placeholder((1,),dtypes.float32,21,addrspace=AddrSpace.REG)
    init=ag[0].store(0.); init=au.after(init)[0].store(0.)
    ug=ag.after(init)[0].store(ag.after(nib)[0]+cg)
    uu=au.after(ug)[0].store(au.after(nib)[0]+cu).end(nib,logical)
    tg=_warp_reduce_sum_staged(ag.after(uu)[0],lane,LANES_PER_ROW,90)
    tu=_warp_reduce_sum_staged(au.after(uu)[0],lane,LANES_PER_ROW,95)
    # This is the required round point: exact GLU arithmetic above, then the
    # fp16 prelude that the existing direct FFN-down path materializes.
    z=_silu_uop(tg).mul(tu).cast(dtypes.float16)
    zsh=UOp.placeholder((PACK,),dtypes.float16,30,addrspace=AddrSpace.LOCAL)
    publish=zsh[row_local].store(z,lane.eq(0)); ready=UOp.barrier(UOp.group(publish))
    active=lid<PACK; safe_lid=active.where(lid,UOp.const(dtypes.weakint,0))
    rounded=zsh.after(ready)[safe_lid].cast(dtypes.float32)
    amax=warp_reduce_max(rounded.abs(),lid,PACK,100); d=amax/UOp.const(dtypes.float32,127.)
    inv=d.eq(0.).where(UOp.const(dtypes.float32,0.),d.reciprocal())
    q=(rounded*inv).round().maximum(UOp.const(dtypes.float32,-128.)).minimum(UOp.const(dtypes.float32,127.)).cast(dtypes.int8)
    # Use the same fixed shuffle association as the established provider.
    q1=_staged_shfl(q,1,lid,110); q2=_staged_shfl(q,2,lid,111); q3=_staged_shfl(q,3,lid,112)
    payload=out[packet*8+safe_lid//4].store(_pack4((q,q1,q2,q3)),active & (safe_lid%4).eq(0))
    xsum=_warp_reduce_sum_staged(rounded,lid,PACK,120)
    dh=d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32); sh=xsum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    meta_idx=active.bitwise_and(safe_lid.eq(0)).where(UOp.const(dtypes.weakint,Q8_METADATA_BASE)+packet,UOp.const(dtypes.weakint,0))
    meta=out[meta_idx].store(dh.bitwise_or(sh.lshift(16)),active & safe_lid.eq(0))
    return UOp.group(payload,meta).sink(arg=KernelInfo(name="ffn_w1w3_q8_cooperative_12288_4096",opts_to_apply=()))
  return kernel


def _i8lane(packet:UOp, lane:int) -> UOp:
  return packet.rshift(lane*8).bitwise_and(255).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32)


def emit_q4_q8_ffn_down_consumer(rows:int=DOWN_ROWS, k:int=DOWN_K, row_tile:int=2):
  """Q4_K FFN-down consumer for the provider's 12288-element private ABI.

  This deliberately mirrors the established Q4/Q8 attention consumer's
  minimum correction and packet addressing.  It is separate only because
  that emitter is fixed at K=4096 and its metadata base is consequently 1024.
  """
  if (rows,k,row_tile) != (DOWN_ROWS,DOWN_K,2): raise ValueError("P3b consumer is fixed to Qwen 4096x12288, row_tile=2")
  q8_packs,q8_meta=k//4,k//4
  def kernel(out:UOp, words:UOp, xp:UOp):
    ro,ri=UOp.range(rows//row_tile,0),UOp.range(row_tile,1,axis_type=AxisType.LOCAL)
    p4,b=UOp.range(8,2,axis_type=AxisType.LOCAL),UOp.range(k//256,3,axis_type=AxisType.REDUCE)
    row=ro*row_tile+ri; base=(row*(k//256)+b)*Q4K_WORDS_PER_BLOCK; contrib=UOp.const(dtypes.float32,0.)
    for group in range(8):
      d,dmin,sc,mn=_q4k_group_params(words,base,group)
      qw=words[base+4+(group//2)*8+p4].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      xv=xp[b*64+group*8+p4]
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xv).cast(dtypes.float32)
      xsum=_i8lane(xv,0)+_i8lane(xv,1)+_i8lane(xv,2)+_i8lane(xv,3)
      q8d=xp[q8_meta+b*8+group].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
      contrib=contrib+q8d*(d*sc.cast(dtypes.float32)*dot-dmin*mn.cast(dtypes.float32)*xsum.cast(dtypes.float32))
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); acc=acc.after(acc[0].store(0.))
    acc=acc.after(acc[0].store(acc.after(b)[0]+contrib).end(b)); total=acc[0]
    for slot,off in enumerate((4,2,1),90): total=total+_staged_shfl(total,off*row_tile,p4,slot)
    return out[row].store(total).end(ro,ri,p4).sink(arg=KernelInfo(name="q4k_q8_ffn_down_4096_12288",opts_to_apply=()))
  return kernel


def fp16_glu(gate:np.ndarray, up:np.ndarray) -> np.ndarray:
  """Independent declared round point for CPU oracle tests."""
  gate=np.asarray(gate,dtype=np.float32); up=np.asarray(up,dtype=np.float32)
  if gate.shape != (ROWS,) or up.shape != (ROWS,): raise ValueError("expected one Qwen FFN hidden vector")
  return (gate/(1.0+np.exp(-gate))*up).astype(np.float16)


def pack_q8_1_private(values:np.ndarray) -> np.ndarray:
  """Reference ABI: uint32 int8x4 payload then uint32 fp16(d)|fp16(raw-sum)."""
  values=np.asarray(values,dtype=np.float16).reshape(-1)
  if values.size != ROWS: raise ValueError("expected 12288 fp16 values")
  out=np.zeros(Q8_METADATA_BASE+ROWS//PACK,dtype=np.uint32)
  for g, block16 in enumerate(values.reshape(-1,PACK)):
    block=block16.astype(np.float32); d32=np.float32(np.max(np.abs(block)))/np.float32(127.); d=np.float16(d32)
    # Quantization uses the unrounded fp32 d; only packet metadata rounds to
    # fp16. This is the CUDA provider ordering.
    inv=np.float32(0.) if d32 == 0 else np.float32(1.)/d32
    scaled=block*inv
    # ggml's CPU reference uses roundf (ties away from zero), unlike NumPy's
    # bankers-rounding default.
    q=np.clip(np.where(scaled >= 0, np.floor(scaled+.5), np.ceil(scaled-.5)),-128,127).astype(np.int8).view(np.uint8)
    out[g*8:g*8+8]=q.reshape(8,4).astype(np.uint32) @ np.array([1,256,65536,16777216],dtype=np.uint32)
    # CUDA reduces this warp with shfl-down 16,8,4,2,1; preserve that
    # association instead of NumPy's implementation-dependent reduction.
    sums=block.copy()
    for off in (16,8,4,2,1): sums[:off]=sums[:off]+sums[off:off+off]
    d_bits=np.asarray(d).view(np.uint16).astype(np.uint32); s_bits=np.asarray(np.float16(sums[0])).view(np.uint16).astype(np.uint32)
    out[Q8_METADATA_BASE+g]=d_bits | (s_bits<<np.uint32(16))
  return out


def llama_q8_1_oracle_private(values:np.ndarray, library:str="/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0") -> np.ndarray:
  """Independent llama CPU-reference packet, transposed into our private ABI.

  The reference produces literal 36-byte blocks (half d, half s, 32 int8
  values); only this adapter changes storage order. It is deliberately not
  used by ``pack_q8_1_private`` so tests retain an external numerical oracle.
  """
  values=np.asarray(values,dtype=np.float16).reshape(-1)
  if values.size != ROWS: raise ValueError("expected 12288 fp16 values")
  lib=ctypes.CDLL(str(pathlib.Path(library)),mode=ctypes.RTLD_LOCAL)
  fn=lib.quantize_row_q8_1_ref; fn.restype=None
  fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64]
  source=np.ascontiguousarray(values.astype(np.float32)); raw=(ctypes.c_uint8*(values.size//PACK*36))()
  fn(source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),raw,values.size)
  blocks=np.frombuffer(bytes(raw),dtype=np.uint8).reshape(-1,36)
  out=np.zeros(Q8_METADATA_BASE+ROWS//PACK,dtype=np.uint32)
  payload=blocks[:,4:36].reshape(-1,4).astype(np.uint32)
  out[:Q8_METADATA_BASE]=payload @ np.array([1,256,65536,16777216],dtype=np.uint32)
  d=blocks[:,:2].copy().reshape(-1).view(np.uint16).astype(np.uint32)
  s=blocks[:,2:4].copy().reshape(-1).view(np.uint16).astype(np.uint32)
  out[Q8_METADATA_BASE:]=d|(s<<np.uint32(16))
  return out


def q4_q8_ffn_down_row_reference(words:np.ndarray, packed:np.ndarray, row:int=0) -> np.float32:
  """Independent CPU interpretation of one Q4/Q8 FFN-down output row.

  It consumes the provider's packed ABI directly, including the Q4 minimum
  correction.  This is intentionally scalar/oracle code, separate from UOps.
  """
  words=np.asarray(words,dtype=np.uint32).reshape(-1); packed=np.asarray(packed,dtype=np.uint32).reshape(-1)
  if not (0 <= row < DOWN_ROWS) or words.size < (row+1)*48*Q4K_WORDS_PER_BLOCK or packed.size != 3456:
    raise ValueError("invalid fixed-shape FFN-down inputs")
  total=np.float32(0.)
  for block in range(48):
    base=(row*48+block)*Q4K_WORDS_PER_BLOCK; hdr=words[base:base+4]
    d=np.frombuffer(np.uint16(hdr[0]&0xffff).tobytes(),dtype=np.float16)[0].astype(np.float32)
    dmin=np.frombuffer(np.uint16(hdr[0]>>16).tobytes(),dtype=np.float16)[0].astype(np.float32)
    def byte(word, index): return np.uint32(word >> np.uint32(index*8)) & np.uint32(255)
    for group in range(8):
      if group < 4:
        sc=byte(hdr[1],group)&63; mn=byte(hdr[2],group)&63
      else:
        hi=byte(hdr[3],group-4); sc=(hi&15)|((byte(hdr[1],group-4)>>6)<<4); mn=(hi>>4)|((byte(hdr[2],group-4)>>6)<<4)
      bits=np.uint16(packed[Q8_METADATA_BASE+block*8+group]&0xffff)
      q8d=np.frombuffer(bits.tobytes(),dtype=np.float16)[0].astype(np.float32)
      for word in range(8):
        qword=words[base+4+(group//2)*8+word] >> np.uint32((group%2)*4)
        xword=packed[block*64+group*8+word]
        for lane in range(4):
          q4=np.float32((qword >> np.uint32(lane*8)) & np.uint32(15))
          q8=np.uint8((xword >> np.uint32(lane*8)) & np.uint32(255)).view(np.int8).astype(np.float32)
          total=np.float32(total + q8d*(d*np.float32(sc)*q4-dmin*np.float32(mn))*q8)
  return total


def q4_w1w3_packet_reference(gate_words:np.ndarray, up_words:np.ndarray, x:np.ndarray, packet:int) -> np.ndarray:
  """CPU oracle for one 32-row cooperative provider CTA, returning fp16 GLU."""
  gate_words=np.asarray(gate_words,dtype=np.uint32).reshape(-1); up_words=np.asarray(up_words,dtype=np.uint32).reshape(-1)
  x=np.asarray(x,dtype=np.float16).reshape(-1)
  if not 0 <= packet < ROWS//PACK or x.size != K: raise ValueError("invalid provider oracle inputs")
  out=np.empty(PACK,dtype=np.float16)
  def half(bits): return np.frombuffer(np.uint16(bits).tobytes(),dtype=np.float16)[0].astype(np.float32)
  def row_dot(words,row):
    lanes=np.zeros(8,dtype=np.float32)
    for lane in range(8):
      acc=np.float32(0.)
      for block in (lane*2,lane*2+1):
        base=(row*16+block)*Q4K_WORDS_PER_BLOCK; hdr=words[base:base+4]
        d,dmin=half(hdr[0]&0xffff),half(hdr[0]>>16)
        def byte(word,index): return np.uint32(word >> np.uint32(index*8))&np.uint32(255)
        for group in range(8):
          if group < 4: sc=byte(hdr[1],group)&63; mn=byte(hdr[2],group)&63
          else:
            hi=byte(hdr[3],group-4); sc=(hi&15)|((byte(hdr[1],group-4)>>6)<<4); mn=(hi>>4)|((byte(hdr[2],group-4)>>6)<<4)
          for word in range(8):
            qword=words[base+4+(group//2)*8+word] >> np.uint32((group%2)*4)
            for nib in range(4):
              q=np.float32((qword >> np.uint32(nib*8))&np.uint32(15)); xv=np.float32(x[block*256+group*32+word*4+nib])
              acc=np.float32(acc+np.float32((d*np.float32(sc)*q-dmin*np.float32(mn))*xv))
      lanes[lane]=acc
    for off in (4,2,1): lanes=np.float32(lanes+lanes[np.arange(8)^off])
    return lanes[0]
  for local in range(PACK):
    row=packet*PACK+local; gate=row_dot(gate_words,row); up=row_dot(up_words,row)
    silu=np.float32(gate*np.float32(1.)/np.float32(1.+np.exp2(np.float32(gate*np.float32(-1./math.log(2.))))))
    out[local]=np.float16(np.float32(silu*up))
  return out
