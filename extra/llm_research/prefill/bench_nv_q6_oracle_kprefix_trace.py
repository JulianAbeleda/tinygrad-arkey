#!/usr/bin/env python3
"""Real-Qwen K256 prefix localization and trusted weight-scale repair gate."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, statistics, time
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record,_run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import ROWS,COLS,SHARED_BYTES,q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

M,N,K,K256=512,4096,12288,48
DEPTHS=(1,2,4,8,16,24,32,40,48)
TRACE_HEADER,TRACE_STRIDE=16,248
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
CONTRACTS={"baseline":"legacy","repair":"trusted_fp16","packed":"trusted_fp16_packed"}

def _buf(t): return t.uop.buffer.get_buf("NV")
def _stats(x): return {"samples_us":x,"min_us":min(x),"median_us":statistics.median(x),"max_us":max(x)}
def _paired(lhs,rhs):
  delta=[a-b for a,b in zip(lhs,rhs)]
  return {**_stats(delta),"wins":sum(a<b for a,b in zip(lhs,rhs)),"pairs":len(delta)}
def _resource_signature(compiler):
  summary=compiler["sass"];resources=summary.get("resources") or {};families=summary.get("families",{})
  return {"stack_bytes":resources.get("stack_bytes"),"local_static_bytes":resources.get("local_static_bytes"),
    "LDL":families.get("LDL",0),"STL":families.get("STL",0)}
def _u32(x): return np.asarray(x,dtype=np.float32).view(np.uint32)
def _i32_bits(x): return np.asarray(x,dtype=np.uint32).view(np.int32).item()

def _decode(raw):
  q=np.empty((16,16),np.int8)
  for g in range(16):
    half,pg=g//8,g%8; ql=half*64+(pg%4)*16; qh=half*32+(pg%2)*16
    q[g]=((((raw[ql:ql+16]>>(4 if pg>=4 else 0))&15)|(((raw[128+qh:128+qh+16]>>((pg//2)*2))&3)<<4)).astype(np.int16)-32).astype(np.int8)
  return q,raw[192:208].view(np.int8),raw[208:210].view(np.float16)[0]

def _published(raw,mode):
  out=np.full(76,0xdeadbeef,np.uint32)
  for tx in range(32):
    ql=int.from_bytes(raw[4*tx:4*tx+4].tobytes(),"little"); qi=(tx//16)*8+tx%8
    qh=int.from_bytes(raw[128+4*qi:132+4*qi].tobytes(),"little"); sh=(tx&8)>>2
    q0=(ql&0x0f0f0f0f)|(((qh>>sh)<<4)&0x30303030); q1=((ql>>4)&0x0f0f0f0f)|((qh>>sh)&0x30303030)
    def sub(v): return int.from_bytes((np.frombuffer(v.to_bytes(4,"little"),np.uint8).astype(np.int16)-32).astype(np.int8).tobytes(),"little")
    kq=2*tx-tx%16; out[kq]=sub(q0); out[kq+16]=sub(q1)
  db=raw[208:210].view(np.uint16)[0];trusted=mode in ("repair","packed")
  out[64]=_u32(np.float32(raw[208:210].view(np.float16)[0])) if trusted else db
  if mode == "packed":
    d=np.float32(raw[208:210].view(np.float16)[0]);scales=raw[192:208].view(np.int8).astype(np.float32)
    halves=np.float16(np.float32(d*scales)).view(np.uint16).astype(np.uint32)
    out[65:73]=halves[0::2]|(halves[1::2]<<16)
  else: out[65:69]=raw[192:208].copy().view(np.uint32)
  return out

def _record_prefix(q,scales,depth):
  broad=np.concatenate([broad_record(np.ascontiguousarray(q[:,e*256:(e+1)*256].T),
    np.ascontiguousarray(scales[:,e*8:(e+1)*8].T)) for e in range(depth)]).reshape(-1)
  sums=q.reshape(ROWS,depth*8,32).astype(np.int32).sum(2).astype(np.float32)
  wide=np.frombuffer(q.tobytes()+scales.tobytes()+sums.tobytes(),np.uint32).copy()
  return broad,wide

def _render(depth,mode,trace,probe,root,full_route=False):
  root.mkdir(parents=True,exist_ok=True); ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  trace_u=ph(TRACE_HEADER+depth*TRACE_STRIDE,dtypes.uint32,3) if trace else None
  replicas,tiles_m,tiles_n=(128,4,32) if full_route else (1,1,1)
  ast=q6_oracle_broad_cta_kernel(ph((M*N if full_route else ROWS*COLS),dtypes.float32,0),
    ph((N if full_route else ROWS)*depth*105,dtypes.uint16,1),
    ph(tiles_m*depth*2*COLS*36,dtypes.uint32,2),replicas=replicas,depth=depth,tile_grid=(tiles_m,tiles_n),prefetch_second_panel=True,
    oracle_publisher=True,weight_scale_contract=CONTRACTS[mode],
    trace=trace_u,trace_config=probe if trace else None)
  started=time.perf_counter(); p=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in p.src if x.op is Ops.SOURCE); render_ms=(time.perf_counter()-started)*1e3
  name=f"nv_q6_prefix_{mode}_d{depth}{'_trace' if trace else ''}"; match=re.search(r'__launch_bounds__\(256\) (\w+)\(',source)
  if match is None: raise RuntimeError("rendered symbol missing")
  source_path=root/f"{name}.cu";source_path.write_text(source);started=time.perf_counter();binary=Device["NV"].compiler.compile(source)
  compile_ms=(time.perf_counter()-started)*1e3;cubin=root/f"{name}.cubin";cubin.write_bytes(binary)
  census=analyze_cubin(cubin,root/"sass",match.group(1));return match.group(1),binary,{"source":str(source_path),
    "cubin":str(cubin),"sha256":hashlib.sha256(binary).hexdigest(),"render_ms":render_ms,"compile_ms":compile_ms,
    "sass":census["summary"],"sass_artifacts":{k:census[k] for k in ("sass_json","disassembly","resources")}}

def _wide_expected(depth,halfs,record,root,full_route=False):
  m,n=(M,N) if full_route else (ROWS,COLS)
  root.mkdir(parents=True,exist_ok=True); result=_run("wide",m,n,depth*256,halfs,record,1,root,(128,128,2,4,256))
  src=(root/"wide.cu").read_text(); binary=Device["NV"].compiler.compile(src); symbol_match=re.search(r'__launch_bounds__\(256\) (\w+)\(',src)
  if symbol_match is None: raise RuntimeError("trusted wide symbol missing")
  out=Tensor.full((m,n),float("nan"),device="NV").contiguous().realize()
  grid=(32,4,1) if full_route else (1,1,1)
  NVProgram(Device["NV"],symbol_match.group(1),binary)(_buf(out),_buf(record),_buf(halfs),global_size=grid,local_size=(32,2,4),wait=True)
  return out.numpy(),result

def _mapping(row,col):
  band=row//32;n=(row%32)//16;lr=row%8;r=2*((row%16)//8)+(col&1);cg=col//16
  wp=(col%16)//8;lc=(col%8)//2;warp=2*band+wp;lane=4*lr+lc
  return np.asarray((0x51365452,1,row,col,32*warp+lane,warp,lane,band,wp,lr,lc,cg,n,r,8*cg+4*n+r,248),np.uint32)

def _trace_compare(depth,mode,tr,raw_row,q,scales,probe,broad_flat,got):
  row,col=probe; mapping=_mapping(row,col); trusted=mode in ("repair","packed");packed=mode=="packed"
  host_q6=np.stack([_published(raw_row[e],mode) for e in range(depth)])
  dev_q6=np.stack([tr[TRACE_HEADER+e*TRACE_STRIDE:TRACE_HEADER+e*TRACE_STRIDE+76] for e in range(depth)])
  records=np.asarray(broad_flat,np.uint32).reshape(depth,2,ROWS,36)
  host_q8=records[:,:,col,:];dev_q8=np.stack([[tr[TRACE_HEADER+e*TRACE_STRIDE+76+p*36:TRACE_HEADER+e*TRACE_STRIDE+112+p*36]
    for p in range(2)] for e in range(depth)])
  host_values=np.stack([_decode(raw_row[e])[0].reshape(-1) for e in range(depth)])
  dev_values=dev_q6[:,:64].copy().view(np.int8).reshape(depth,256)
  host_scales=np.stack([_decode(raw_row[e])[1] for e in range(depth)])
  dev_scales=None if packed else dev_q6[:,65:69].copy().view(np.int8).reshape(depth,16)
  d_bits=np.asarray([_u32(np.float32(_decode(raw_row[e])[2])) for e in range(depth)],np.uint32).reshape(-1)
  dev_d=np.asarray([tr[TRACE_HEADER+e*TRACE_STRIDE+148] for e in range(depth)],np.uint32)
  carrier_ok=yscale_ok=imma_ok=correction_ok=weight_table_ok=weighted_terms_ok=fp_ok=chain_ok=True
  acc=np.float32(0.0);last_bits=np.uint32(0)
  band,n,lr,cg,r,lc=int(mapping[7]),int(mapping[12]),int(mapping[9]),int(mapping[11]),int(mapping[13]),int(mapping[10])
  for e in range(depth):
    q6,qs,wdh=_decode(raw_row[e]);wd=np.float32(wdh)
    for kp in range(2):
      for p in range(4):
        g0,g1=kp*8+2*p,kp*8+2*p+1;tb=TRACE_HEADER+e*TRACE_STRIDE+149+kp*48+p*12;v=tr[tb:tb+12]
        qv0=records[e,kp,(cg*16+int(mapping[8])*8+lr),4+(2*p)*4+lc]
        qv1=records[e,kp,(cg*16+int(mapping[8])*8+lr),4+(2*p+1)*4+lc]
        carrier_ok &= bool(v[0]==qv0 and v[1]==qv1)
        ys=np.float32(scales[col,e*8+kp*4+p]);yscale_ok &= bool(v[2]==_u32(ys))
        z0=int(q6[g0].astype(np.int32)@q[col,e*256+g0*16:e*256+(g0+1)*16].astype(np.int32))
        z1=int(q6[g1].astype(np.int32)@q[col,e*256+g1*16:e*256+(g1+1)*16].astype(np.int32))
        s0,s1=int(qs[g0]),int(qs[g1]);imma_ok &= bool(_i32_bits(v[5])==z0 and _i32_bits(v[6])==z1)
        ws0=np.float32(np.float16(np.float32(wd*np.float32(s0))));ws1=np.float32(np.float16(np.float32(wd*np.float32(s1))))
        term0=np.float32(ws0*np.float32(z0));term1=np.float32(ws1*np.float32(z1));weighted=np.float32(term0+term1)
        if packed:
          weight_table_ok &= bool(v[3]==_u32(ws0) and v[4]==_u32(ws1))
          weighted_terms_ok &= bool(v[7]==_u32(term0) and v[8]==_u32(term1) and v[9]==_u32(weighted))
        else:
          correction_ok &= bool(_i32_bits(v[3])==s0 and _i32_bits(v[4])==s1 and _i32_bits(v[7])==s0*z0 and
            _i32_bits(v[8])==s1*z1 and _i32_bits(v[9])==s0*z0+s1*z1)
        before=np.uint32(v[10]);after=np.uint32(v[11]);chain_ok &= bool(before==last_bits)
        if trusted:
          acc=np.float32(acc+np.float32(ys*weighted));fp_ok &= bool(_u32(acc)==after)
        last_bits=after
      phase_bit=tr[TRACE_HEADER+e*TRACE_STRIDE+245+kp]
      chain_ok &= bool(phase_bit==last_bits)
    chain_ok &= bool(tr[TRACE_HEADER+e*TRACE_STRIDE+247]==last_bits)
  checks={"mapping_exact":bool(np.array_equal(tr[:16],mapping)),"q6_publication_exact":bool(np.array_equal(host_q6,dev_q6)),
    "q8_both_panels_exact":bool(np.array_equal(host_q8,dev_q8)),"decoded_q6_values_exact":bool(np.array_equal(host_values,dev_values)),
    "d_fp32_bits_exact":bool(np.array_equal(d_bits,dev_d)),"traced_q8_carriers_exact":carrier_ok,"q8_scale_bits_exact":yscale_ok,
    "imma_c0_c1_exact":imma_ok,"accumulator_chain_bits_exact":chain_ok,"trusted_fp32_steps_exact":bool(fp_ok if trusted else True),
    "final_accumulator_output_bits_exact":bool(last_bits==_u32(got[col,row]))}
  if packed:
    checks.update({"packed_weight_scale_table_exact":bool(np.array_equal(host_q6[:,65:73],dev_q6[:,65:73]) and weight_table_ok),
      "weighted_fp32_terms_exact":weighted_terms_ok})
  else:
    checks.update({"decoded_q6_scales_exact":bool(np.array_equal(host_scales,dev_scales)),"integer_corrections_exact":correction_ok})
  checks["all_required_exact"]=bool(all(checks.values()))
  arrays={"trace":tr,"host_q6":host_q6,"device_q6":dev_q6,"host_q8":host_q8,"device_q8":dev_q8,
    "host_q6_values":host_values,"device_q6_values":dev_values,"host_q6_scales":host_scales}
  if packed: arrays.update({"host_weight_scale_table":host_q6[:,65:73],"device_weight_scale_table":dev_q6[:,65:73]})
  else: arrays["device_q6_scales"]=dev_scales
  return checks,arrays

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depths",default=",".join(map(str,DEPTHS)));ap.add_argument("--rounds",type=int,default=31)
  ap.add_argument("--out",type=pathlib.Path,required=True);ap.add_argument("--artifacts",type=pathlib.Path,required=True);a=ap.parse_args()
  depths=tuple(map(int,a.depths.split(",")));a.artifacts.mkdir(parents=True,exist_ok=True)
  path=pathlib.Path(a.model);meta=read_metadata(path);info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K:raise RuntimeError(info)
  full=packed_u16_slice(path,meta,info).numpy().reshape(N,K256,105);raw=full.view(np.uint8).reshape(N,K256,210)
  wide_host,q_all,s_all=wide_record(M,K);q_tile=np.ascontiguousarray(q_all[:ROWS]);s_tile=np.ascontiguousarray(s_all[:ROWS])
  d=raw[:,:,208:210].copy().view(np.float16).astype(np.float32).reshape(N,K256,1);sc=raw[:,:,192:208].view(np.int8).astype(np.float32)
  broad=d*sc;rounded=np.float16(broad).astype(np.float32);mask=broad.view(np.uint32)!=rounded.view(np.uint32)
  pop={"elements":int(mask.size),"different":int(mask.sum()),"fraction":float(mask.mean()),
    "max_abs":float(np.abs(broad-rounded).max()),"mean_abs":float(np.abs(broad-rounded).mean())}
  loc=np.argwhere(mask[:ROWS].transpose(1,0,2))[0];epoch,row,group=map(int,loc);qq,_,_=_decode(raw[row,epoch])
  dots=q_tile[:,epoch*256+group*16:epoch*256+(group+1)*16].astype(np.int32)@qq[group].astype(np.int32)
  col=int(np.flatnonzero(dots)[0]);probe=(row,col)
  prefixes=[];first=None
  for depth in depths:
    root=a.artifacts/f"depth_{depth:02d}"; hp=np.ascontiguousarray(full[:ROWS,:depth]).reshape(-1)
    bq,wq=_record_prefix(q_tile[:,:depth*256],s_tile[:,:depth*8],depth)
    ht=Tensor(hp,device="NV").contiguous().realize();bt=Tensor(bq,device="NV").contiguous().realize();wt=Tensor(wq,device="NV").contiguous().realize()
    expected,wide_result=_wide_expected(depth,ht,wt,root/"wide")
    arms={}
    outputs={}
    for mode in ("baseline","repair","packed"):
      name,binary,compiler=_render(depth,mode,True,probe,root/mode);out=Tensor.full((ROWS,COLS),float("nan"),device="NV").contiguous().realize()
      trace=Tensor.zeros((TRACE_HEADER+depth*TRACE_STRIDE,),dtype=dtypes.uint32,device="NV").contiguous().realize()
      NVProgram(Device["NV"],name,binary,shared_mem=LAUNCH_SHARED_BYTES)(_buf(out),_buf(ht),_buf(bt),_buf(trace),
        global_size=(1,1,1),local_size=(256,1,1),wait=True,timeout=120000)
      got=out.numpy();outputs[mode]=got;tr=trace.numpy();diff=np.abs(got-expected);close=np.isclose(got,expected,rtol=2e-5,atol=2e-3)
      checks,arrays=_trace_compare(depth,mode,tr,raw[row,:depth],q_tile[:,:depth*256],s_tile[:,:depth*8],probe,bq,got)
      np.savez_compressed(root/mode/"trace.npz",**arrays)
      arms[mode]={"correctness":{"finite":bool(np.isfinite(got).all()),"max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
        "failing_count":int(np.count_nonzero(~close)),"reference_passed":bool(close.all()),"trace_checks":checks,
        "passed":bool(close.all() and checks["all_required_exact"])},
        "compiler":compiler,"trace":{"artifact":str(root/mode/"trace.npz"),"contract":CONTRACTS[mode],
          "production_observed":["q6_shared","d_fp32","q8_panel0","q8_panel1","q8_carriers","q8_scale","imma_c0_c1","mapping","accumulator_bits"]+
          (["packed_weight_scales","weighted_fp32_terms"] if mode=="packed" else ["signed_scales","integer_correction_dot"]),
          "a_carrier_observation":"indirect_exact_via_q6_publication_plus_actual_imma",
          "host_replayed":["canonical_q6_decode","shared_publication","weight_scale_rounding"]}}
    packed_equal=bool(np.array_equal(outputs["packed"].view(np.uint32),outputs["repair"].view(np.uint32)))
    arms["packed"]["correctness"]["explicit_repair_bit_equal"]=packed_equal
    arms["packed"]["correctness"]["passed"] &= packed_equal
    divergence=("Q6_PUBLICATION" if not arms["baseline"]["correctness"]["trace_checks"]["q6_publication_exact"] else
      ("WEIGHT_SCALE_CONTRACT" if not arms["baseline"]["correctness"]["passed"] and arms["repair"]["correctness"]["passed"] else
       ("PACKED_PUBLICATION" if not arms["packed"]["correctness"]["trace_checks"]["packed_weight_scale_table_exact"] else
        ("PACKED_CONSUMER" if not arms["packed"]["correctness"]["passed"] else None))))
    if first is None and divergence:first={"depth":depth,"stage":divergence}
    prefixes.append({"depth":depth,"wide":wide_result,"arms":arms,"first_divergence":divergence})
  repair_prefix_pass=all(x["arms"]["repair"]["correctness"]["passed"] for x in prefixes)
  packed_prefix_pass=all(x["arms"]["packed"]["correctness"]["passed"] for x in prefixes)
  packed_trace_pass=all(x["arms"]["packed"]["correctness"]["trace_checks"]["all_required_exact"] for x in prefixes)
  prefix_bit_equal=all(x["arms"]["packed"]["correctness"]["explicit_repair_bit_equal"] for x in prefixes)
  full_ab=None
  if repair_prefix_pass and packed_prefix_pass:
    bfull=np.concatenate([broad_record(np.ascontiguousarray(q_all[m:m+ROWS,e*256:(e+1)*256].T),
      np.ascontiguousarray(s_all[m:m+ROWS,e*8:(e+1)*8].T)) for m in range(0,M,ROWS) for e in range(K256)]).reshape(-1)
    hfull=Tensor(full.reshape(-1),device="NV").contiguous().realize();qfull=Tensor(bfull,device="NV").contiguous().realize()
    whost=Tensor(wide_host,device="NV").contiguous().realize();expected,full_wide=_wide_expected(K256,hfull,whost,a.artifacts/"full_wide",True)
    states={};samples={x:[] for x in ("baseline","repair","packed")}
    for mode in samples:
      name,binary,comp=_render(K256,mode,False,probe,a.artifacts/f"full_{mode}",True);states[mode]=(NVProgram(Device["NV"],name,binary,shared_mem=LAUNCH_SHARED_BYTES),
        Tensor.full((M,N),float("nan"),device="NV").contiguous().realize(),comp)
    full_correctness={};full_outputs={}
    for mode,(p,o,_) in states.items():
      p(_buf(o),_buf(hfull),_buf(qfull),global_size=(128,1,1),local_size=(256,1,1),wait=True,timeout=120000)
      got=o.numpy();full_outputs[mode]=got;diff=np.abs(got-expected);close=np.isclose(got,expected,rtol=2e-5,atol=2e-3)
      full_correctness[mode]={"finite":bool(np.isfinite(got).all()),"max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
        "failing_count":int(np.count_nonzero(~close)),"passed":bool(close.all())}
    full_bit_equal=bool(np.array_equal(full_outputs["packed"].view(np.uint32),full_outputs["repair"].view(np.uint32)))
    full_correctness["packed"]["explicit_repair_bit_equal"]=full_bit_equal
    full_correctness["packed"]["passed"] &= full_bit_equal
    orders=(("baseline","repair","packed"),("repair","packed","baseline"),("packed","baseline","repair"),
      ("packed","repair","baseline"),("repair","baseline","packed"),("baseline","packed","repair"))
    for r in range(a.rounds):
      for mode in orders[r%len(orders)]:
        p,o,_=states[mode];samples[mode].append(p(_buf(o),_buf(hfull),_buf(qfull),global_size=(128,1,1),local_size=(256,1,1),wait=True)*1e6)
    work=M*N*K
    paired={"packed_minus_repair":_paired(samples["packed"],samples["repair"]),
      "packed_minus_legacy":_paired(samples["packed"],samples["baseline"])}
    resource_signatures={m:_resource_signature(states[m][2]) for m in states}
    explicit_resources=resource_signatures["repair"];packed_resources=resource_signatures["packed"]
    resource_comparable=all(explicit_resources[k] is not None and packed_resources[k] is not None for k in explicit_resources)
    resource_no_regression=bool(resource_comparable and all(packed_resources[k] <= explicit_resources[k] for k in explicit_resources))
    full_ab={"wide":full_wide,"correctness":full_correctness,"timing":{m:{**_stats(v),"median_us_per_output_element_k":statistics.median(v)/work,
      "giga_output_element_k_per_s":work/(statistics.median(v)*1e3)} for m,v in samples.items()},
      "paired":paired,"compiler":{m:states[m][2] for m in states},"resources":resource_signatures,
      "resource_no_regression_vs_repair":resource_no_regression,"order_schedule":[list(x) for x in orders],"rounds":a.rounds,
      "repair_passed":bool(full_correctness["repair"]["passed"]),"packed_passed":bool(full_correctness["packed"]["passed"])}
  full_reference_pass=bool(full_ab is not None and full_ab["repair_passed"] and full_ab["packed_passed"])
  performance_pass=bool(full_ab is not None and a.rounds==31 and
    full_ab["timing"]["repair"]["median_us"]-full_ab["timing"]["packed"]["median_us"] >= 3.0 and
    full_ab["paired"]["packed_minus_repair"]["wins"] >= 24)
  resource_pass=bool(full_ab is not None and full_ab["resource_no_regression_vs_repair"])
  promotion=bool(repair_prefix_pass and packed_prefix_pass and packed_trace_pass and prefix_bit_equal and full_reference_pass and performance_pass and resource_pass)
  verdict=("PROMOTE_TRUSTED_FP16_PACKED" if promotion else ("PACKED_CORRECT_NOT_PROMOTED" if packed_prefix_pass and full_reference_pass else
    ("FAIL_PACKED_FULL" if packed_prefix_pass else "FAIL_PACKED_PREFIX")))
  result={"schema":"tinygrad.nv_q6_oracle_kprefix_trace.v2","depths":depths,"tile":[0,0],"probe":{"row":row,"col":col,
    "earliest_rounding_epoch":epoch,"group":group},"scale_rounding_population":pop,"prefixes":prefixes,"first_divergence":first,
    "repair_prefix_pass":repair_prefix_pass,"packed_prefix_pass":packed_prefix_pass,"packed_trace_pass":packed_trace_pass,
    "prefix_explicit_repair_bit_equal":prefix_bit_equal,"performance_pass":performance_pass,"resource_pass":resource_pass,
    "promotion_eligible":promotion,"full_ab":full_ab,"verdict":verdict,"passed":promotion}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
