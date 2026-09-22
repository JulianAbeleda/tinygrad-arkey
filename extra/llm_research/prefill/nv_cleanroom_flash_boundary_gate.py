"""Default-off diagnostic gate for the real prefill Flash boundary.

This module deliberately does not launch a replacement kernel.  It records
metadata from graph tensors without materializing host copies and rejects all
but the first exact Qwen3-8B pp512 contract.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, pathlib

@dataclass(frozen=True)
class BoundaryRecord:
  q_shape: tuple
  k_shape: tuple
  v_shape: tuple
  q_dtype: str
  k_dtype: str
  v_dtype: str
  q_strides: tuple
  k_strides: tuple
  v_strides: tuple
  start_pos: int
  ring: bool
  admitted: bool
  reason: str

def _shape(x): return tuple(int(v) for v in x.shape)
def _strides(x):
  st=getattr(x, 'strides', None)
  return tuple(int(v) for v in st) if st is not None else ()

def inspect_boundary(q, k, v, *, start_pos: int, ring: bool = False) -> BoundaryRecord:
  qs,ks,vs=_shape(q),_shape(k),_shape(v)
  exact=(qs==(1,32,512,128) and ks==(1,8,512,128) and vs==ks and
         str(q.dtype) in ('dtypes.float16','float16','<class \'numpy.float16\'>') and
         str(k.dtype) in ('dtypes.float16','float16','<class \'numpy.float16\'>') and
         str(v.dtype) in ('dtypes.float16','float16','<class \'numpy.float16\'>') and
         start_pos==0 and not ring)
  reason='exact_b1_t512_hq32_hkv8_hd128_start0_nonring' if exact else 'contract_mismatch'
  return BoundaryRecord(qs,ks,vs,str(q.dtype),str(k.dtype),str(v.dtype),_strides(q),_strides(k),_strides(v),start_pos,ring,exact,reason)

def write_record(record: BoundaryRecord, path: str|pathlib.Path) -> None:
  pathlib.Path(path).write_text(json.dumps(asdict(record), sort_keys=True)+'\n')

def admitted(record: BoundaryRecord) -> bool:
  return record.admitted
