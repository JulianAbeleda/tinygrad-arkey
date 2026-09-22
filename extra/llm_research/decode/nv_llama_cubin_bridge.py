#!/usr/bin/env python3
"""Guarded research bridge for one pinned llama Q6_K MMVQ cubin on native NV.

This is deliberately not a selector or production route.  The parent process
runs one worker with a timeout; the worker validates the target ELF metadata,
packs the declared CUDA parameter ABI, patches NVProgram's QMD for the target
symbol in the multi-kernel cubin, and executes one guarded correctness case.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, struct, subprocess, sys
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]))

from tinygrad.helpers import round_up
from tinygrad.runtime.support.elf import elf_loader
from extra.llm_research.decode.nv_eiattr_kparam_census import pack_cbuf, parameter_layout, records
from scratchpad.llama_cuda_quantized_live_oracle import (DEFAULT_BASE, DEFAULT_CUBIN, ENTRY_Q6, Q6_BLOCK_BYTES,
  _cpu_quantizers, decode_q6, decode_q8, pack_q6, pack_q8)


def _elf_rows(content:bytes):
  off=0
  while off < len(content):
    typ,param,size=struct.unpack_from("BBH",content,off)
    payload=bytes(content[off+4:off+4+size]) if typ == 4 else b""
    yield typ,param,size,payload
    off += 4+size if typ == 4 else 4


def inspect_symbol(blob:bytes, symbol:str=ENTRY_Q6) -> dict:
  image,sections,relocs=elf_loader(blob,force_section_align=128)
  by_name={section.name:section for section in sections}
  required=(f".text.{symbol}",f".nv.info.{symbol}",f".nv.constant0.{symbol}",f".nv.shared.{symbol}")
  missing=[name for name in required if name not in by_name]
  if missing: raise ValueError(f"missing target ELF sections: {missing}")
  text=by_name[f".text.{symbol}"]
  function_symbol_index=text.header.sh_info
  global_info=by_name[".nv.info"]
  reg_rows=[]; stack_rows=[]
  for typ,param,size,payload in _elf_rows(global_info.content):
    if typ != 4 or size < 8: continue
    owner,value=struct.unpack_from("<II",payload)
    if owner != function_symbol_index: continue
    if param == 0x2f: reg_rows.append(value)
    if param == 0x12: stack_rows.append(value)
  if len(reg_rows) != 1 or len(stack_rows) != 1:
    raise ValueError(f"target global attributes are not unique: regs={reg_rows}, stack={stack_rows}")

  symbol_rows=records(blob).get(symbol)
  if symbol_rows is None: raise ValueError("target has no per-symbol EIATTR records")
  parameter_base,parameter_size,kparams=parameter_layout(symbol_rows)
  constant0_size=by_name[f".nv.constant0.{symbol}"].header.sh_size
  if parameter_base+parameter_size != constant0_size:
    raise ValueError(f"CB0 extent mismatch: {parameter_base}+{parameter_size}!={constant0_size}")
  reloc_types=sorted(set(row[2] for row in relocs))
  if any(kind not in (2,0x38,0x39) for kind in reloc_types):
    raise ValueError(f"NVProgram-unsupported relocation types: {reloc_types}")
  return {"symbol":symbol,"image_bytes":image.nbytes,"text_bytes":text.header.sh_size,
    "function_symbol_index":function_symbol_index,"register_count":reg_rows[0],"stack_bytes":stack_rows[0],
    "shared_section_bytes":by_name[f".nv.shared.{symbol}"].header.sh_size,
    "constant0_size":constant0_size,"parameter_base":parameter_base,"parameter_size":parameter_size,
    "parameter_count":len(kparams),"relocation_types":reloc_types}


def _u32(value:int) -> bytes: return struct.pack("<I",value)
def _u64(value:int) -> bytes: return struct.pack("<Q",value)
def _uint3(x:int,y:int,z:int) -> bytes: return struct.pack("<III",x,y,z)


def mmvq_parameter_region(rows:list[dict], weight_addr:int, activation_addr:int, output_addr:int,
                          nrows:int, k:int, guard:int=0) -> bytes:
  if k <= 0 or k % 256 or nrows <= 0: raise ValueError("invalid MMVQ shape")
  row_blocks,q8_blocks=k//256,k//32
  zero=_uint3(0,0,0); one=_uint3(1,0,1)
  # Exact signature used by the pinned plain llama CUDA MMVQ symbol.  Values
  # are serialized by ordinal, then placed only at EIATTR-declared offsets.
  values={0:_u64(weight_addr),1:_u64(activation_addr),2:_u64(0),
    3:struct.pack("<QQQi4x",0,0,0,0),4:_u64(output_addr+guard*4),5:_u32(k),6:zero,
    7:_u32(row_blocks),8:_u32(q8_blocks),9:_u32(nrows),10:one,
    11:_u32(nrows*row_blocks),12:_u32(q8_blocks),13:_u32(nrows),14:one,
    15:_u32(nrows*row_blocks),16:_u32(q8_blocks),17:_u32(nrows),18:_u32(0)}
  return pack_cbuf(rows,values)


def bind_foreign_cbuf(program, metadata:dict, parameter_region:bytes) -> None:
  """Install a fully packed foreign CB0 and target-specific QMD metadata."""
  base,size,total=metadata["parameter_base"],metadata["parameter_size"],metadata["constant0_size"]
  if len(parameter_region) != size or base+size != total: raise ValueError("foreign parameter extent mismatch")
  prefix=struct.pack(f"<{len(program.cbuf_0)}I",*program.cbuf_0)
  if len(prefix) != base: raise ValueError(f"NVProgram prefix is {len(prefix)} bytes, expected {base}")
  full=prefix+parameter_region
  if len(full) % 4: raise ValueError("foreign CB0 must be word aligned")
  program.cbuf_0=list(struct.unpack(f"<{len(full)//4}I",full))

  old_addr,_old_size=program.constbufs[0]
  program.constbufs[0]=(old_addr,total)
  program.regs_usage=metadata["register_count"]
  if program.qmd.ver >= 4:
    program.qmd.write(register_count=program.regs_usage,constant_buffer_size_shifted4_0=total>>4,
                      constant_buffer_valid_0=1)
  else:
    program.qmd.write(register_count_v=program.regs_usage,constant_buffer_size_shifted4_0=total>>4,
                      constant_buffer_valid_0=1)
  program.max_threads=((65536//round_up(max(1,program.regs_usage)*32,256))//4)*4*32
  program.kernargs_alloc_size=round_up(total,1<<8)+(8<<8)


def _sha256(blob:bytes) -> str: return hashlib.sha256(blob).hexdigest()


def inspect_file(cubin:pathlib.Path) -> dict:
  blob=cubin.read_bytes(); metadata=inspect_symbol(blob)
  rows=records(blob)[ENTRY_Q6]
  probe=mmvq_parameter_region(rows,0x1111222233334444,0x5555666677778888,0x9999aaaabbbbcccc,1024,4096,32)
  return {"schema":"tinygrad.nv_llama_cubin_bridge_inspection.v1","cubin":str(cubin),"sha256":_sha256(blob),
          "metadata":metadata,"packed_parameter_bytes":len(probe),"verdict":"CPU_ABI_PASS"}


def worker(cubin:pathlib.Path, base:pathlib.Path, seed:int=202608056) -> dict:
  from tinygrad.device import Buffer, Device
  from tinygrad.dtype import dtypes
  from tinygrad.runtime.ops_nv import NVProgram

  nrows,k,guard=1024,4096,32
  blob=cubin.read_bytes(); metadata=inspect_symbol(blob); symbol_rows=records(blob)[ENTRY_Q6]
  rng=np.random.default_rng(seed); q4,q6,q8=_cpu_quantizers(base)
  weights_f32=rng.normal(0,.2,(nrows,k)).astype(np.float32)
  activation_f32=rng.normal(0,.2,k).astype(np.float32)
  weights_payload,activation_payload=pack_q6(weights_f32,q6),pack_q8(activation_f32,q8)
  if len(weights_payload) != nrows*(k//256)*Q6_BLOCK_BYTES: raise RuntimeError("Q6 payload size mismatch")
  reference=decode_q6(weights_payload).reshape(nrows,k) @ decode_q8(activation_payload)
  sentinel=np.float32(12345.25); initial=np.full(nrows+2*guard,sentinel,dtype=np.float32)
  weight=Buffer("NV",len(weights_payload),dtypes.uint8,initial_value=bytearray(weights_payload))
  activation=Buffer("NV",len(activation_payload),dtypes.uint8,initial_value=bytearray(activation_payload))
  output=Buffer("NV",initial.size,dtypes.float32,initial_value=bytearray(initial.tobytes()))
  dev=Device["NV"]; dev.synchronize()
  wbuf,abuf,obuf=(buf.ensure_allocated()._buf for buf in (weight,activation,output))
  params=mmvq_parameter_region(symbol_rows,wbuf.va_addr,abuf.va_addr,obuf.va_addr,nrows,k,guard)
  program=NVProgram(dev,ENTRY_Q6,blob); bind_foreign_cbuf(program,metadata,params)
  elapsed=program(global_size=(nrows,1,1),local_size=(32,4,1),wait=True)
  raw=bytearray(output.nbytes); output.copyout(memoryview(raw)); got=np.frombuffer(raw,dtype=np.float32)
  result=got[guard:guard+nrows]; guards_ok=bool(np.all(got[:guard] == sentinel) and np.all(got[-guard:] == sentinel))
  finite=bool(np.isfinite(result).all()); delta=np.abs(result-reference)
  numerical_ok=finite and bool(np.all(delta <= .02)); passed=guards_ok and numerical_ok
  out={"schema":"tinygrad.nv_llama_cubin_bridge_gate.v1","shape":[nrows,k],"seed":seed,
    "cubin_sha256":_sha256(blob),"metadata":metadata,"guards_ok":guards_ok,"finite":finite,
    "max_abs":float(delta.max()) if finite else None,"mean_abs":float(delta.mean()) if finite else None,
    "atol":.02,"numerical_ok":numerical_ok,"pass":passed,"verdict":"PASS" if passed else "FAIL"}
  if passed: out["kernel_time_us"]=float(elapsed)*1e6
  return out


def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--cubin",type=pathlib.Path,default=DEFAULT_CUBIN)
  ap.add_argument("--base",type=pathlib.Path,default=DEFAULT_BASE); ap.add_argument("--seed",type=int,default=202608056)
  ap.add_argument("--timeout",type=float,default=45.); ap.add_argument("--worker",action="store_true")
  ap.add_argument("--inspect-only",action="store_true"); a=ap.parse_args()
  if a.inspect_only:
    print(json.dumps(inspect_file(a.cubin),sort_keys=True)); return
  if a.worker:
    print(json.dumps(worker(a.cubin,a.base,a.seed),sort_keys=True)); return
  cmd=[sys.executable,str(pathlib.Path(__file__).resolve()),"--worker","--cubin",str(a.cubin),"--base",str(a.base),"--seed",str(a.seed)]
  try: completed=subprocess.run(cmd,capture_output=True,text=True,timeout=a.timeout,check=False)
  except subprocess.TimeoutExpired as exc:
    print(json.dumps({"schema":"tinygrad.nv_llama_cubin_bridge_parent.v1","pass":False,"verdict":"TIMEOUT",
      "timeout_s":a.timeout,"stdout":exc.stdout,"stderr":exc.stderr},sort_keys=True)); return
  if completed.returncode != 0:
    print(json.dumps({"schema":"tinygrad.nv_llama_cubin_bridge_parent.v1","pass":False,"verdict":"WORKER_ERROR",
      "returncode":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr},sort_keys=True)); return
  print(completed.stdout.strip())


if __name__ == "__main__": main()
