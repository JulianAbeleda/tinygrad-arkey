#!/usr/bin/env python3
"""Execute exact BoltBeam-lowered route artifacts against independent references."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.route_class_numerics import _make_q4k_words, _make_q6k_halfs
from extra.llm_research.layout import q4_k_reference, q6_k_reference


def _buffer(t:Tensor): return t.uop.buffer.get_buf("NV")


def _program_key(program) -> str: return program.key.hex() if isinstance(program.key,bytes) else str(program.key)


def _runner(ast:UOp, tensors:tuple[Tensor, ...]):
  program=to_program(ast, Device["NV"].renderer)
  runtime=get_runtime("NV", program)
  global_size,local_size=program.arg.launch_dims({})
  buffers=[_buffer(t) for t in tensors]
  return program,lambda wait=False: runtime(*[buffers[i] for i in program.arg.globals],
    global_size=global_size,local_size=local_size,vals=(),wait=wait)


def _timing(run, reps:int) -> list[float]:
  run(True)
  return [run(True)*1e6 for _ in range(reps)]


def _selected_reference(raw:np.ndarray, x:np.ndarray, rows:tuple[int, ...], row_bytes:int, quant:str) -> np.ndarray:
  reference=q4_k_reference if quant == "q4" else q6_k_reference
  values=[]
  for row in rows:
    packed=np.ascontiguousarray(raw[row*row_bytes:(row+1)*row_bytes])
    dense=reference(Tensor(packed,dtype=dtypes.uint8),x.size).numpy().astype(np.float32).reshape(-1)
    values.append(float(dense @ x.astype(np.float32)))
  return np.asarray(values,dtype=np.float32)


def qualify_q6_v(reps:int) -> dict:
  rows,k,seed=1024,4096,202609011
  selected=(0,1,7,127,511,1023)
  halfs_np=_make_q6k_halfs(rows,k,seed)
  raw=np.ascontiguousarray(halfs_np.view(np.uint8).reshape(-1))
  x_np=np.random.default_rng(seed+1).normal(0,.2,k).astype(np.float16)
  out=Tensor.empty(rows,dtype=dtypes.float32,device="NV").realize()
  halfs=Tensor(halfs_np.copy(),dtype=dtypes.uint16,device="NV").realize()
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device="NV").realize()
  emitter,ticket=lower_authorized_candidate({"family":"q6_v.v1"},
    (("decode_q6k_v_four_warp_fp16_geometry","q6_v_four_warp_fp16"),))
  ast=emitter(UOp.placeholder((rows,),dtypes.float32,0),
    UOp.placeholder(halfs_np.shape,dtypes.uint16,1),UOp.placeholder((k,),dtypes.float16,2))
  program,run=_runner(ast,(out,halfs,x)); timing=_timing(run,reps); got=out.numpy()[list(selected)]
  ref=_selected_reference(raw,x_np,selected,(k//256)*210,"q6")
  err=np.abs(got-ref); atol=np.maximum(2e-4,np.abs(ref)*2e-4)
  return {"route_id":"decode_q6k_v_four_warp_fp16_geometry","candidate_family":"q6_v.v1",
    "tickets":[item.to_dict() for item in ticket.tickets],"program_key":_program_key(program),"selected_rows":selected,
    "payload_sha256":hashlib.sha256(raw.tobytes()+x_np.tobytes()).hexdigest(),
    "correctness":{"pass":bool(np.all(err <= atol)),"max_abs":float(err.max()),
      "max_allowed":float(atol.max()),"finite":bool(np.isfinite(got).all())},
    "timing_us":{"samples":timing,"median":statistics.median(timing)}}


def qualify_q4_down(reps:int) -> dict:
  rows,k,seed=4096,12288,202609012
  selected=(0,1,7,127,2047,4095)
  words_np,raw_matrix=_make_q4k_words(rows,k,seed)
  raw=np.ascontiguousarray(raw_matrix.reshape(-1))
  x_np=np.random.default_rng(seed+1).normal(0,.2,k).astype(np.float16)
  residual_np=np.random.default_rng(seed+2).normal(0,.1,rows).astype(np.float32)
  out=Tensor.empty(rows,dtype=dtypes.float32,device="NV").realize()
  words=Tensor(words_np.copy(),dtype=dtypes.uint32,device="NV").realize()
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device="NV").realize()
  residual=Tensor(residual_np.copy(),dtype=dtypes.float32,device="NV").realize()
  candidate={"family":"q4_ffn_down.v1","block_count":3,"resadd":True,"load_style":"vector"}
  emitter,ticket=lower_authorized_candidate(candidate,
    (("decode_q4k_ffn_down_fp16_geometry","q4_ffn_down_fp16"),("decode_ffn_down_resadd","q4_ffn_down_resadd")))
  ast=emitter(UOp.placeholder((rows,),dtypes.float32,0),
    UOp.placeholder(words_np.shape,dtypes.uint32,1),UOp.placeholder((k,),dtypes.float16,2),
    UOp.placeholder((rows,),dtypes.float32,3))
  program,run=_runner(ast,(out,words,x,residual)); timing=_timing(run,reps); got=out.numpy()[list(selected)]
  ref=_selected_reference(raw,x_np,selected,(k//256)*144,"q4")+residual_np[list(selected)]
  err=np.abs(got-ref); atol=np.maximum(2e-3,np.abs(ref)*5e-4)
  return {"route_id":"decode_q4k_ffn_down_fp16_geometry","candidate_family":"q4_ffn_down.v1",
    "tickets":[item.to_dict() for item in ticket.tickets],"program_key":_program_key(program),"selected_rows":selected,
    "payload_sha256":hashlib.sha256(raw.tobytes()+x_np.tobytes()+residual_np.tobytes()).hexdigest(),
    "correctness":{"pass":bool(np.all(err <= atol)),"max_abs":float(err.max()),
      "max_allowed":float(atol.max()),"finite":bool(np.isfinite(got).all())},
    "timing_us":{"samples":timing,"median":statistics.median(timing)}}


def qualify_q4_w1w3(reps:int) -> list[dict]:
  rows,k,seed=12288,4096,202609014
  selected=(0,1,7,127,4095,12287)
  gate_words,gate_raw_matrix=_make_q4k_words(rows,k,seed);up_words,up_raw_matrix=_make_q4k_words(rows,k,seed+1)
  gate_raw=np.ascontiguousarray(gate_raw_matrix.reshape(-1));up_raw=np.ascontiguousarray(up_raw_matrix.reshape(-1))
  x_np=np.random.default_rng(seed+2).normal(0,.2,k).astype(np.float16)
  gate=Tensor(gate_words.copy(),dtype=dtypes.uint32,device="NV").realize()
  up=Tensor(up_words.copy(),dtype=dtypes.uint32,device="NV").realize();x=Tensor(x_np.copy(),dtype=dtypes.float16,device="NV").realize()
  gate_ref=_selected_reference(gate_raw,x_np,selected,(k//256)*144,"q4")
  up_ref=_selected_reference(up_raw,x_np,selected,(k//256)*144,"q4")
  reference=(gate_ref/(1.0+np.exp(-gate_ref))*up_ref).astype(np.float32);results=[]
  for store_fp16 in (False,True):
    dtype=dtypes.float16 if store_fp16 else dtypes.float32;out=Tensor.empty(rows,dtype=dtype,device="NV").realize()
    authorities=(("decode_q4k_w1w3_fusion","q4_w1w3_fused"),)+(
      (("decode_q4k_w1w3_fp16_store","q4_w1w3_fused_fp16"),) if store_fp16 else ())
    candidate={"family":"q4_w1w3.v1","rows":rows,"k":k,"load_style":"vector","store_fp16":store_fp16}
    emitter,ticket=lower_authorized_candidate(candidate,authorities)
    ast=emitter(UOp.placeholder((rows,),dtype,0),UOp.placeholder(gate_words.shape,dtypes.uint32,1),
      UOp.placeholder(up_words.shape,dtypes.uint32,2),UOp.placeholder((k,),dtypes.float16,3))
    program,run=_runner(ast,(out,gate,up,x));timing=_timing(run,reps);got=out.numpy()[list(selected)].astype(np.float32)
    ref=reference.astype(np.float16).astype(np.float32) if store_fp16 else reference
    err=np.abs(got-ref);atol=np.maximum(3e-3,np.abs(ref)*1e-3)
    results.append({"route_id":"decode_q4k_w1w3_fp16_store" if store_fp16 else "decode_q4k_w1w3_fusion",
      "candidate_family":"q4_w1w3.v1","store_fp16":store_fp16,"tickets":[item.to_dict() for item in ticket.tickets],
      "program_key":_program_key(program),"selected_rows":selected,
      "payload_sha256":hashlib.sha256(gate_raw.tobytes()+up_raw.tobytes()+x_np.tobytes()).hexdigest(),
      "correctness":{"pass":bool(np.all(err<=atol)),"max_abs":float(err.max()),"max_allowed":float(atol.max()),
        "finite":bool(np.isfinite(got).all())},"timing_us":{"samples":timing,"median":statistics.median(timing)}})
  return results


def qualify_q4_g3(reps:int) -> dict:
  rows,k,seed=4096,4096,202609015;selected=(0,1,7,127,2047,4095)
  words_np,raw_matrix=_make_q4k_words(rows,k,seed);raw=np.ascontiguousarray(raw_matrix.reshape(-1))
  x_np=np.random.default_rng(seed+1).normal(0,.2,k).astype(np.float16)
  out=Tensor.empty(rows,dtype=dtypes.float32,device="NV").realize();words=Tensor(words_np.copy(),dtype=dtypes.uint32,device="NV").realize()
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device="NV").realize()
  candidate={"family":"q4_g3_route.v1","rows":rows,"k":k,"load_style":"vector","epilogue_kind":"","epilogue_binding":None}
  emitter,ticket=lower_authorized_candidate(candidate,(("decode_q4k_g3_generated","q4_g3_gemv"),))
  ast=emitter(UOp.placeholder((rows,),dtypes.float32,0),UOp.placeholder(words_np.shape,dtypes.uint32,1),
    UOp.placeholder((k,),dtypes.float16,2));program,run=_runner(ast,(out,words,x));timing=_timing(run,reps)
  got=out.numpy()[list(selected)];ref=_selected_reference(raw,x_np,selected,(k//256)*144,"q4")
  err=np.abs(got-ref);atol=np.maximum(2e-3,np.abs(ref)*5e-4)
  return {"route_id":"decode_q4k_g3_generated","candidate_family":"q4_g3_route.v1",
    "tickets":[item.to_dict() for item in ticket.tickets],"program_key":_program_key(program),"selected_rows":selected,
    "payload_sha256":hashlib.sha256(raw.tobytes()+x_np.tobytes()).hexdigest(),
    "correctness":{"pass":bool(np.all(err<=atol)),"max_abs":float(err.max()),"max_allowed":float(atol.max()),
      "finite":bool(np.isfinite(got).all())},"timing_us":{"samples":timing,"median":statistics.median(timing)}}


def main() -> int:
  parser=argparse.ArgumentParser();parser.add_argument("--route",choices=("q6-v","q4-down","q4-w1w3","q4-g3","all"),default="all")
  parser.add_argument("--reps",type=int,default=11);parser.add_argument("--out",required=True);args=parser.parse_args()
  if not str(Device.DEFAULT).startswith("NV"): raise RuntimeError(f"native NV required, got {Device.DEFAULT}")
  rows=[]
  if args.route in ("q6-v","all"): rows.append(qualify_q6_v(args.reps))
  if args.route in ("q4-down","all"): rows.append(qualify_q4_down(args.reps))
  if args.route in ("q4-w1w3","all"): rows.extend(qualify_q4_w1w3(args.reps))
  if args.route in ("q4-g3","all"): rows.append(qualify_q4_g3(args.reps))
  report={"schema":"tinygrad.boltbeam_route_numerical_qualification.v1","device":str(Device.DEFAULT),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"routes":rows,
    "pass":all(row["correctness"]["pass"] for row in rows)}
  text=json.dumps(report,indent=2,sort_keys=True);out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(text+"\n");print(text)
  return 0 if report["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
