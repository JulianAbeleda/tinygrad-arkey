#!/usr/bin/env python3
"""Fail-closed Gate13 for const/restrict Q8 RegionLoad panel1 copies."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, sys

from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, ParamArg, PostBarrierRegion, RegionLoad

import extra.llm_research.prefill.bench_nv_q6_oracle_region_load_panel1 as gate12
import extra.llm_research.prefill.bench_nv_q6_oracle_true_late_panel1 as gate
import extra.llm_research.prefill.nv_q6_oracle_broad_cta as impl


ANCHOR_SHA="6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137"
FIXUP_SHA="483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514"
GATE12_REGION_SOURCE_SHA="206ebe0ea6214fccfa6c389c19e6b4e6f1d9e0fcc38557495552710555e90017"
BASELINE_US=256.256
CANDIDATE="const_restrict_region_q8_panel1"
TARGET=Target.parse("NV:CUDA:sm_120")
EXPECTED={"IMMA":256,"LDSM":32,"LDS":176,"LDG":109,"STS":73,"STG":64,"BAR":4,
          "I2FP":1024,"FMUL":1544,"FADD":1024,"FFMA":0}
Q8_QUALIFIER_RE=re.compile(r"const unsigned int \*__restrict__ (data2_\d+)")


def _builder(*args,q8_panel1_schedule="early",**kwargs):
  candidate=q8_panel1_schedule != "early"
  return impl.q6_oracle_broad_cta_kernel(*args,region_load_q8_panel1=candidate,
    const_restrict_q8=candidate,**kwargs)


def _ast_flags(region:bool,qualified:bool):
  original=gate.q6_oracle_broad_cta_kernel
  def builder(*args,q8_panel1_schedule="early",**kwargs):
    return impl.q6_oracle_broad_cta_kernel(*args,region_load_q8_panel1=region,
      const_restrict_q8=qualified,**kwargs)
  gate.q6_oracle_broad_cta_kernel=builder
  try: return gate._ast("true_late_tail" if region else "early")
  finally: gate.q6_oracle_broad_cta_kernel=original


def _source(ast) -> str:
  program=to_program(ast,CUDARenderer(TARGET))
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def _source_proof() -> dict:
  default,region,candidate=(_source(_ast_flags(False,False)),_source(_ast_flags(True,False)),
                            _source(_ast_flags(True,True)))
  match=Q8_QUALIFIER_RE.search(candidate);q8_name=match.group(1) if match else ""
  normalized=candidate if match is None else candidate.replace(match.group(0),f"unsigned int* {q8_name}")
  offsets=tuple(4608+i*256 for i in range(18))
  copies=[line.strip() for line in candidate.splitlines() if "buf0" in line and q8_name in line and "=" in line]
  lhs_writes=[line for line in candidate.splitlines() if "=" in line and q8_name in line.split("=",1)[0]]
  signature_match=re.search(r'extern "C" __global__ void __launch_bounds__\(256\) \w+\([^)]*\)',candidate)
  signature=signature_match.group(0) if signature_match else ""
  gates={
    "one_q8_qualifier":match is not None and candidate.count("__restrict__")==1,
    "only_qualifier_source_delta":normalized==region,
    "gate12_region_control_frozen":hashlib.sha256(region.encode()).hexdigest()==GATE12_REGION_SOURCE_SHA,
    "launch_bounds_256_unchanged":"__launch_bounds__(256)" in signature and "__launch_bounds__(256,1)" not in signature,
    "sm120_unchanged":TARGET.arch=="sm_120",
    "direct_copies_18":len(copies)==18 and all(sum(f"+{off}" in line for line in copies)==1 for off in offsets),
    "barriers_unchanged":candidate.count("__syncthreads();")==region.count("__syncthreads();")==default.count("__syncthreads();")==4,
    "no_q8_stores":not lhs_writes,
  }
  return {"gates":gates,"default_source_sha256":hashlib.sha256(default.encode()).hexdigest(),
    "region_source_sha256":hashlib.sha256(region.encode()).hexdigest(),
    "candidate_source_sha256":hashlib.sha256(candidate.encode()).hexdigest(),"q8_parameter":q8_name,
    "direct_assignments":len(copies),"source_barriers":candidate.count("__syncthreads();")}


def _semantic(ast) -> dict:
  topo=ast.toposort();base=gate12._semantic(ast)
  qualified=[x for x in topo if x.op is Ops.PARAM and isinstance(x.arg,ParamArg) and x.arg.const_restrict]
  markers=[x for x in topo if x.op is Ops.AFTER and isinstance(x.arg,RegionLoad)]
  loads=[x for x in topo if x.op is Ops.LOAD and any(y in markers for y in x.src[1:])]
  owners={x.src[0].src[0] for x in loads}
  return {**base,"qualified_params":len(qualified),"qualified_is_region_owner":len(qualified)==1 and owners=={qualified[0]},
    "qualified_written":bool(qualified and any(qualified[0] in x.src[0].pointer_base_params() for x in topo if x.op is Ops.STORE))}


def _panel_constant(disassembly:str) -> dict:
  insns=gate._sass_instructions(disassembly)
  q8=[x for x in insns if x["opcode"].startswith("LDG.E") and ".U16" not in x["opcode"]]
  panel=q8[-18:] if len(q8)==36 else []
  bars=[x for x in insns if x["opcode"].startswith("BAR.SYNC")]
  return {"panel_constant_loads":sum(x["opcode"]=="LDG.E.CONSTANT" for x in panel),
    "panel_load_opcodes":[x["opcode"] for x in panel],"panel_load_pcs":[f"0x{x['pc']:x}" for x in panel],
    "barrier_ordinals":[x["ordinal"] for x in bars],"barrier_pcs":[f"0x{x['pc']:x}" for x in bars]}


def _write(path:pathlib.Path,data:dict):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


def _ledger(path:pathlib.Path,result:dict):
  path.parent.mkdir(parents=True,exist_ok=True)
  static=result.get("static_source_sass",result);verdict=result.get("gate13_verdict",result.get("verdict","UNKNOWN"))
  lines=["# NV Q6 const/restrict RegionLoad panel1 Gate13 decision", "", "## Verdict", "", f"`{verdict}`", "",
    f"Frozen main: `{ANCHOR_SHA}`", f"Frozen fixup: `{FIXUP_SHA}`", f"Admitted timing reference: `{BASELINE_US:.3f} us`", ""]
  if "panel" in static:
    lines += ["## Static gate", "", f"Candidate cubin: `{static.get('candidate_cubin_sha256')}`",
      f"Panel constant loads: `{static['panel_constant'].get('panel_constant_loads')}/18`",
      f"Panel LDG/STS and span: `{static['panel'].get('panel1_loads')}/{static['panel'].get('panel1_stores')}` / "
      f"`{static['panel'].get('panel1_load_to_store_span_instructions')}`", f"Families: `{json.dumps(static.get('families',{}),sort_keys=True)}`",
      f"Resources: `{json.dumps(static.get('resources',{}),sort_keys=True)}`", ""]
  if result.get("gpu_work_started"):
    cand=result["timing"][CANDIDATE];anchor=result["timing"][gate.ANCHOR];paired=result["paired"]
    lines += ["## Correctness and locked R31", "", f"Correctness passed: `{result.get('correctness_passed')}`",
      f"Partial/final bit exact: `{result['exactness']['partials_bit_exact']}` / `{result['exactness']['final_bit_exact']}`",
      f"Anchor main/fixup/total medians: `{anchor['main']['r31']['median_us']:.3f}` / `{anchor['fixup']['r31']['median_us']:.3f}` / `{anchor['total']['r31']['median_us']:.3f} us`",
      f"Candidate main/fixup/total medians: `{cand['main']['r31']['median_us']:.3f}` / `{cand['fixup']['r31']['median_us']:.3f}` / `{cand['total']['r31']['median_us']:.3f} us`",
      f"Paired main median/wins: `{paired['main']['r31']['median_us']:.3f} us`, `{paired['main']['candidate_wins']}/31`",
      f"Paired total median/wins: `{paired['total']['r31']['median_us']:.3f} us`, `{paired['total']['candidate_wins']}/31`", ""]
  lines += [f"GPU lock held: `{result.get('gpu_lock',{}).get('acquired',result.get('gpu_lock_held',False))}`", ""]
  path.write_text("\n".join(lines))


def _static(args) -> int:
  result={"schema":"tinygrad.nv_q6_const_restrict_region_panel1_gate13.static.v1","frozen":{"main_sha256":ANCHOR_SHA,
    "fixup_sha256":FIXUP_SHA,"baseline_us":BASELINE_US},"gpu_lock_held":os.getenv("NV_Q6_GPU_LOCK_HELD")=="1",
    "gpu_work_started":False,"source":_source_proof()}
  if not result["gpu_lock_held"]:
    result|={"verdict":"BLOCKED_GPU_LOCK_NOT_HELD","promotion_eligible":False};_write(args.out,result);_ledger(args.ledger,result);return 2
  if not all(result["source"]["gates"].values()):
    result|={"verdict":"REJECT_SOURCE","promotion_eligible":False};_write(args.out,result);_ledger(args.ledger,result);return 2
  original=gate.q6_oracle_broad_cta_kernel;gate.q6_oracle_broad_cta_kernel=_builder
  try:
    anchor_ast=gate._ast("early");candidate_ast=gate._ast(CANDIDATE)
    result["semantic"]=_semantic(candidate_ast)
    result["compile_elapsed_s"]={"anchor":gate12._compile_bounded(anchor_ast,"anchor",args.artifacts,args.compile_bound),
      "candidate":gate12._compile_bounded(candidate_ast,"candidate",args.artifacts,args.compile_bound)}
  except Exception as exc:
    result|={"verdict":"REJECT_COMPILE","promotion_eligible":False,"error":f"{type(exc).__name__}: {exc}"};_write(args.out,result);_ledger(args.ledger,result);return 2
  finally: gate.q6_oracle_broad_cta_kernel=original
  abase,ajson,_=gate12._artifact(args.artifacts,"anchor");cbase,cjson,cdis=gate12._artifact(args.artifacts,"candidate")
  asha=hashlib.sha256((abase/"anchor.cubin").read_bytes()).hexdigest();csha=hashlib.sha256((cbase/"candidate.cubin").read_bytes()).hexdigest()
  panel=gate._classify_q8_panel1(cdis);constant=_panel_constant(cdis);fam=cjson["families"];res=cjson["resources"]
  bars=constant["barrier_ordinals"];load=panel.get("panel1_first_load_ordinal");store=panel.get("panel1_first_store_ordinal")
  ordering=bool(len(bars)==4 and load is not None and store is not None and bars[0]<load<bars[1]<store<bars[2])
  semantic_expected={"regions":1,"markers":1,"loads":18,"publications":18,"ends":1,"publication_roots":18,
    "global_immutable":True,"index_markers":0,"qualified_params":1,"qualified_is_region_owner":True,"qualified_written":False}
  gates={"anchor_byte_identical":asha==ANCHOR_SHA,"semantic_exact":result["semantic"]==semantic_expected,
    "panel_exact":panel.get("classified") and panel.get("panel1_loads")==18 and panel.get("panel1_stores")==18,
    "panel_constant_18":constant["panel_constant_loads"]==18,"llama_barrier_shape":ordering,
    "span_le_160":panel.get("panel1_load_to_store_span_instructions",10**9)<=160,
    "families_exact":all(fam.get(k,0)==v for k,v in EXPECTED.items()),"instruction_le_5144":cjson["instruction_total"]<=5144,
    "lop3_anchor":fam.get("LOP3")==ajson["families"].get("LOP3"),
    "no_forbidden":all(fam.get(k,0)==0 for k in ("MEMBAR","ATOM","LDL","STL")),
    "resources":res.get("registers",10**9)<=255 and res.get("stack_bytes")==0 and res.get("local_static_bytes")==0 and res.get("shared_static_bytes")==1024}
  result|={"anchor_rebuild_sha256":asha,"candidate_cubin_sha256":csha,
    "candidate_source_sha256":hashlib.sha256((cbase/"candidate.cu").read_bytes()).hexdigest(),"candidate_symbol":cjson["symbol"],
    "instruction_total":cjson["instruction_total"],"families":fam,"resources":res,"panel":panel,"panel_constant":constant,"gates":gates}
  result["promotion_eligible"]=all(gates.values());result["verdict"]="PASS_SOURCE_SASS" if result["promotion_eligible"] else "REJECT_SOURCE_SASS"
  _write(args.out,result);_ledger(args.ledger,result);print(json.dumps({"verdict":result["verdict"],"gates":gates,"panel":panel,"constant":constant},sort_keys=True))
  return 0 if result["promotion_eligible"] else 2


def _full(args) -> int:
  static=json.loads(args.static_result.read_text())
  if not static.get("promotion_eligible"):
    result={"schema":"tinygrad.nv_q6_const_restrict_region_panel1_gate13.v1","gate13_verdict":"BLOCKED_STATIC_NOT_PASS",
      "static_source_sass":static,"gpu_work_started":False,"gpu_lock":{"acquired":os.getenv("NV_Q6_GPU_LOCK_HELD")=="1"}}
    _write(args.out,result);_ledger(args.ledger,result);return 2
  if os.getenv("NV_Q6_GPU_LOCK_HELD") != "1":
    result={"schema":"tinygrad.nv_q6_const_restrict_region_panel1_gate13.v1","gate13_verdict":"BLOCKED_GPU_LOCK_NOT_HELD",
      "static_source_sass":static,"gpu_work_started":False,"gpu_lock":{"acquired":False}}
    _write(args.out,result);_ledger(args.ledger,result);return 2
  original_builder,original_signature,original_binary=gate.q6_oracle_broad_cta_kernel,gate._signature,gate._binary_gates
  original_candidate,original_arms=gate.CANDIDATE,gate.ARMS
  def signature(artifact):
    sig=original_signature(artifact);fam=artifact["sass"].get("families",{});dis=pathlib.Path(artifact["sass_artifacts"]["disassembly"]).read_text()
    sig["LOP3"]=fam.get("LOP3",0);sig["MEMBAR"]=fam.get("MEMBAR",0);sig["ATOM"]=fam.get("ATOM",0)
    sig["q8_panel1"].update(_panel_constant(dis));return sig
  def binary_gates(anchor,candidate,anchor_ast,candidate_ast):
    gates=original_binary(anchor,candidate,anchor_ast,candidate_ast);gates["ast_signature_exact"]=True
    gates["only_q8_source_qualified"]=all(static["source"]["gates"].values())
    gates["panel_constant_18"]=candidate["q8_panel1"]["panel_constant_loads"]==18
    gates["lop3_anchor"]=candidate["LOP3"]==anchor["LOP3"]
    gates["no_membar_atom"]=candidate["MEMBAR"]==candidate["ATOM"]==0
    return gates
  gate.q6_oracle_broad_cta_kernel=_builder;gate._signature=signature;gate._binary_gates=binary_gates
  gate.CANDIDATE=CANDIDATE;gate.ARMS=(gate.ANCHOR,CANDIDATE)
  old_argv=sys.argv
  sys.argv=[old_argv[0],"--model",str(args.model),"--rounds","31","--warmups","3","--out",str(args.out),
    "--artifacts",str(args.artifacts),"--ledger",str(args.ledger)]
  try: code=gate.main()
  finally:
    sys.argv=old_argv;gate.q6_oracle_broad_cta_kernel=original_builder;gate._signature=original_signature;gate._binary_gates=original_binary
    gate.CANDIDATE=original_candidate;gate.ARMS=original_arms
  result=json.loads(args.out.read_text());result["schema"]="tinygrad.nv_q6_const_restrict_region_panel1_gate13.v1"
  result["static_source_sass"]=static;result["admitted_r31_reference_us"]=BASELINE_US
  result["gate13_verdict"]=("PROMOTE_CONST_RESTRICT_REGION_Q8_PANEL1" if result.get("promotion_eligible") else
    "REJECT_CONST_RESTRICT_REGION_Q8_PANEL1_PERFORMANCE" if result.get("correctness_passed") else
    "FAIL_CLOSED_CONST_RESTRICT_REGION_Q8_PANEL1")
  _write(args.out,result);_ledger(args.ledger,result);return code


def main() -> int:
  ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=("static","full"),default="static")
  ap.add_argument("--out",type=pathlib.Path,required=True);ap.add_argument("--artifacts",type=pathlib.Path,required=True)
  ap.add_argument("--ledger",type=pathlib.Path,required=True);ap.add_argument("--compile-bound",type=int,default=240)
  ap.add_argument("--static-result",type=pathlib.Path);ap.add_argument("--model",type=pathlib.Path,default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  args=ap.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
  if args.phase=="full" and args.static_result is None: raise ValueError("full phase requires --static-result")
  return _static(args) if args.phase=="static" else _full(args)


if __name__=="__main__": raise SystemExit(main())
