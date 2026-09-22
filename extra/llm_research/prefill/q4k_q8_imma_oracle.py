#!/usr/bin/env python3
"""Independent, GPU-free Q4_K x live-CUDA-Q8_1 algebra/IMMA mapping oracle."""
from __future__ import annotations
import numpy as np

def unpack_scales(raw12: bytes):
  s=np.frombuffer(raw12,dtype=np.uint8)
  if len(s)!=12: raise ValueError("Q4_K scale metadata is 12 bytes")
  sc=np.r_[s[:4]&63, (s[8:12]&15)|((s[:4]>>6)<<4)]
  mn=np.r_[s[4:8]&63, (s[8:12]>>4)|((s[4:8]>>6)<<4)]
  return sc.astype(np.int32),mn.astype(np.int32)

def unpack_qs(raw128: bytes):
  b=np.frombuffer(raw128,dtype=np.uint8).reshape(4,32)
  # GGML logical K: group pairs share 32 bytes; low nibble precedes high.
  return np.stack((b&15,b>>4),axis=1).reshape(8,32).astype(np.int8)

def q4k_q8_block(dm, raw12, raw128, q8, d8, raw_sums):
  """Return one 256-K contribution. dm=(D,Dmin); metadata is already fp32."""
  sc,mn=unpack_scales(raw12); q=unpack_qs(raw128)
  q8=np.asarray(q8,dtype=np.int8).reshape(8,32).astype(np.int32)
  d8=np.asarray(d8,dtype=np.float32).reshape(8)
  raw_sums=np.asarray(raw_sums,dtype=np.float32).reshape(8)
  dots=(q.astype(np.int32)*q8).sum(1,dtype=np.int32)
  return np.float32(dm[0])*np.sum(d8*sc*dots,dtype=np.float32)-np.float32(dm[1])*np.sum(mn*raw_sums,dtype=np.float32)

def dense_reference(dm, raw12, raw128, q8, d8, raw_sums):
  sc,mn=unpack_scales(raw12); q=unpack_qs(raw128).astype(np.float32)
  w=np.float32(dm[0])*sc[:,None]*q-np.float32(dm[1])*mn[:,None]
  # Main affine term uses dequantized Q8; min correction uses live raw sums.
  main=np.sum(np.float32(dm[0])*sc*d8*np.sum(q*q8.reshape(8,32),axis=1),dtype=np.float32)
  corr=np.float32(dm[1])*np.sum(mn*raw_sums,dtype=np.float32)
  return main-corr,w

def a_fragment_coord(lane, reg, byte):
  """m16n8k32 row-major A: four b32 registers per lane."""
  return (lane//4 + 8*(reg&1), 4*(lane%4) + 16*(reg//2) + byte)

def b_fragment_coord(lane, reg, byte):
  """m16n8k32 col-major B: two b32 registers per lane; returns (K,N)."""
  return (4*((lane%4)+4*reg)+byte, lane//4)

def d_fragment_coord(lane, reg):
  """Four s32 accumulators per lane."""
  return (lane//4 + 8*(reg//2), 2*(lane%4)+reg%2)

def q4_typed_fragment_value(packed_byte, logical_group):
  """Scalar contract a packed fragment provider must expose after TC permutation."""
  b=int(packed_byte)&255
  return np.int8((b >> 4) & 15 if logical_group & 1 else b & 15)

def q4_broken_postrange_value(packed_byte, logical_group):
  """Observed 2026-08-28 lowering: STACK selector incorrectly becomes group<1."""
  b=int(packed_byte)&255
  return np.int8((b & 15) if logical_group < 1 else (b >> 4))

def adversarial_fixture():
  # Every scale/min high-bit path differs; every nibble and sign quadrant occurs.
  sc=np.array([1,2,3,4,17,34,51,63],np.uint8); mn=np.array([5,6,7,8,18,35,52,62],np.uint8)
  meta=np.empty(12,np.uint8); meta[:4]=(sc[:4]&63)|((sc[4:]>>4)<<6); meta[4:8]=(mn[:4]&63)|((mn[4:]>>4)<<6)
  meta[8:12]=(sc[4:]&15)|((mn[4:]&15)<<4)
  lo=np.arange(128,dtype=np.uint8).reshape(4,32)%16
  hi=(15-lo+np.arange(4,dtype=np.uint8)[:,None])%16
  packed=(lo|(hi<<4)).tobytes()
  q8=((np.arange(256,dtype=np.int16)*37+11)%255-127).astype(np.int8)
  d8=np.array([.125,.25,.5,1,1.5,2,3,4],np.float32)
  sums=np.array([31,-29,23,-19,17,-13,11,-7],np.float32)
  return (np.float32(0.0625),np.float32(0.03125)),meta.tobytes(),packed,q8,d8,sums
