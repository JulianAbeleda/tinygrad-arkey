#!/usr/bin/env python3
"""Gate 1: tile-aligned full-K Q6 direct/factored dA qualifier."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, statistics, time
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record, _run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import COLS, ROWS, SHARED_BYTES, q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin

M,N,K=512,4096,12288
TILES_M,TILES_N,TILES,K256=4,32,128,48
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
MAX_LAUNCH_SHARED_BYTES=58_880

P_TREES={
  "left":"(((u0 + u1) + u2) + u3)",
  "inner_left":"((u0 + (u1 + u2)) + u3)",
  "inner_right":"(u0 + ((u1 + u2) + u3))",
  "right":"(u0 + (u1 + (u2 + u3)))",
  "balanced":"((u0 + u1) + (u2 + u3))",
}


def _arm_specs() -> list[dict]:
  specs=[
    {"mode":"direct_da","baseline":True,"factor_dA":False,"scale_grouping":"legacy","p_tree":"legacy",
     "contraction":"implicit","expression":"acc + ((wd * yscale) * dot)"},
    {"mode":"factored_da","baseline":True,"factor_dA":True,"scale_grouping":"legacy","p_tree":"legacy",
     "contraction":"implicit","expression":"acc + (left_fold_zero(yscale_p * dot_p) * wd)"},
  ]
  for grouping,expression in (("wd_yscale_then_dot","acc + ((wd * yscale) * dot)"),
                              ("wd_then_yscale_dot","acc + (wd * (yscale * dot))")):
    for contraction in ("none","final"):
      specs.append({"mode":f"direct_{grouping}_{contraction}","baseline":False,"factor_dA":False,
        "scale_grouping":grouping,"p_tree":"legacy","contraction":contraction,"expression":expression})
  for tree,tree_expression in P_TREES.items():
    for contraction in ("none","tmp_only","final_only","both"):
      specs.append({"mode":f"factored_{tree}_{contraction}","baseline":False,"factor_dA":True,
        "scale_grouping":"legacy","p_tree":tree,"contraction":contraction,
        "expression":f"acc + (({tree_expression}) * wd)"})
  return specs


def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")
def _stats(xs:list[float]): return {"samples_us":xs,"min_us":min(xs),"median_us":statistics.median(xs),"max_us":max(xs)}
def _windows(xs:list[float]): return {"r9":_stats(xs[:9]),"r31":_stats(xs)}


def _normalized_census(summary:dict) -> dict:
  families=summary["families"]; imma=families.get("IMMA",0); resources=summary["resources"] or {}
  feasible=bool(resources and resources.get("registers",256)<=255 and resources.get("stack_bytes",1)==0 and
    resources.get("local_static_bytes",1)==0 and LAUNCH_SHARED_BYTES<=MAX_LAUNCH_SHARED_BYTES)
  return {"instruction_total":summary["instruction_total"],"imma":imma,
    "instructions_per_static_imma":summary["instruction_total"]/imma if imma else None,
    "families":families,
    "families_per_static_imma":({family:count/imma for family,count in families.items()} if imma else None),
    "opcodes":summary["opcodes"],"resources":resources,"spill_regions":summary["spill_regions"],
    "launch_shared_bytes":LAUNCH_SHARED_BYTES,"max_launch_shared_bytes":MAX_LAUNCH_SHARED_BYTES,
    "resource_feasible":feasible}


def _render(spec:dict, artifacts:pathlib.Path):
  mode,factor_dA=spec["mode"],spec["factor_dA"]
  arm_dir=artifacts/mode; arm_dir.mkdir(parents=True,exist_ok=True)
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=q6_oracle_broad_cta_kernel(ph(M*N,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),replicas=TILES,prefetch_second_panel=True,
    combined_initial_publish=False,factor_dA=factor_dA,oracle_publisher=True,depth=K256,tile_grid=(TILES_M,TILES_N),
    fp32_scale_grouping=spec["scale_grouping"],fp32_p_tree=spec["p_tree"],fp32_contraction=spec["contraction"])
  started=time.perf_counter(); program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(x.arg for x in program.src if x.op is Ops.SOURCE); render_ms=(time.perf_counter()-started)*1e3
  controlled=(f"_fp32_{spec['scale_grouping']}_{spec['p_tree']}_{spec['contraction']}"
              if spec["contraction"]!="implicit" else "")
  name=f"nv_q6_oracle_broad_cta_prefetch{'_factor_da' if factor_dA else ''}_oracle_publisher{controlled}_tiles4x32_d48"
  source_path=arm_dir/f"{name}.cu"; source_path.write_text(source)
  started=time.perf_counter(); binary=Device["NV"].compiler.compile(source); compile_ms=(time.perf_counter()-started)*1e3
  cubin=arm_dir/f"{name}.cubin"; cubin.write_bytes(binary)
  census_artifact=analyze_cubin(cubin,arm_dir/"sass",name); census=census_artifact["summary"]
  expected_intrinsics={"FMUL":source.count("__fmul_rn("),"FADD":source.count("__fadd_rn("),"FFMA":source.count("__fmaf_rn(")}
  actual_families={family:census["families"].get(family,0) for family in expected_intrinsics}
  proof_required=spec["contraction"]!="implicit"
  contraction_proof={"required":proof_required,"method":"explicit CUDA round-to-nearest intrinsics plus exact SASS family census",
    "expected_from_source":expected_intrinsics,"actual_sass":actual_families,
    "passed":bool(not proof_required or expected_intrinsics==actual_families)}
  compiler={"render_ms":render_ms,"compile_ms":compile_ms,"source":str(source_path),"source_bytes":len(source),
    "cubin":str(cubin),"cubin_sha256":hashlib.sha256(binary).hexdigest(),
    "sass":_normalized_census(census),"contraction_proof":contraction_proof,
    "sass_artifacts":{k:census_artifact[k] for k in ("sass_json","disassembly","resources")}}
  return name,binary,compiler


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31)
  ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--artifacts",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.rounds<31: raise ValueError("Gate 1 requires R31")
  args.artifacts.mkdir(parents=True,exist_ok=True)

  model=pathlib.Path(args.model); meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K or tuple(reversed(info.dims))!=(N,K): raise RuntimeError(f"illegal fixture {info}")
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K); wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  records=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      records.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
                                  np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(records,axis=0).reshape(-1),device="NV").contiguous().realize()

  wide_dir=args.artifacts/"wide_direct"; wide_dir.mkdir(parents=True,exist_ok=True)
  wide_result=_run("wide_direct",M,N,K,halfs,wide_q8,args.rounds,wide_dir,(128,128,2,4,256))
  wide_source=(wide_dir/"wide_direct.cu").read_text(); compiler=Device["NV"].compiler
  wide_binary=compiler.compile(wide_source)
  match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',wide_source)
  if match is None: raise RuntimeError("trusted wide-direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),wide_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  specs=_arm_specs(); states={}
  for spec in specs:
    mode=spec["mode"]; name,binary,compiler_info=_render(spec,args.artifacts)
    states[mode]={**spec,"program":NVProgram(Device["NV"],name,binary,shared_mem=LAUNCH_SHARED_BYTES),
      "output":Tensor.full((M,N),float("nan"),device="NV").contiguous().realize(),"compiler":compiler_info}

  correctness={}; outputs={}
  for mode,state in states.items():
    state["program"](_buf(state["output"]),_buf(halfs),_buf(q8),global_size=(TILES,1,1),local_size=(256,1,1),
      wait=True,timeout=120000)
    got=state["output"].numpy(); outputs[mode]=got; diff=np.abs(got-expected)
    close=np.isclose(got,expected,rtol=2e-5,atol=2e-3)
    numerical_pass=bool(np.isfinite(got).all() and close.all())
    resource_pass=bool(state["compiler"]["sass"]["resource_feasible"])
    contraction_pass=bool(state["compiler"]["contraction_proof"]["passed"])
    correctness[mode]={"finite":bool(np.isfinite(got).all()),"trusted_reference":"compiler_wide_direct",
      "reference_max_abs":float(diff.max()),"reference_mean_abs":float(diff.mean()),
      "reference_failing_count_rtol2e5_atol2e3":int(np.count_nonzero(~close)),
      "reference_allclose_rtol2e5_atol2e3":bool(close.all()),"numerical_pass":numerical_pass,
      "resource_pass":resource_pass,"contraction_proof_pass":contraction_pass,
      "passed":bool(numerical_pass and resource_pass and contraction_pass)}

  samples={mode:[] for mode in states}
  call_orders=[]; base_order=[spec["mode"] for spec in specs]
  for round_idx in range(args.rounds):
    shift=round_idx%len(base_order); order=base_order[shift:]+base_order[:shift]
    if round_idx%2: order=list(reversed(order))
    call_orders.append(order)
    for mode in order:
      state=states[mode]
      samples[mode].append(state["program"](_buf(state["output"]),_buf(halfs),_buf(q8),global_size=(TILES,1,1),
        local_size=(256,1,1),wait=True,timeout=120000)*1e6)

  total_work=M*N*K; cta_k256_epochs=TILES*K256
  arms={}
  for mode,state in states.items():
    median=statistics.median(samples[mode])
    arms[mode]={"baseline":state["baseline"],"factor_dA":state["factor_dA"],
      "arithmetic":{"scale_grouping":state["scale_grouping"],"p_tree":state["p_tree"],
        "contraction":state["contraction"],"expression":state["expression"]},
      "correctness":correctness[mode],"timing":_windows(samples[mode]),
      "normalization":{"output_elements_times_k":total_work,"cta_k256_epochs":cta_k256_epochs,
        "median_us_per_output_element_k":median/total_work,"median_us_per_cta_k256_epoch":median/cta_k256_epochs,
        "giga_output_element_k_per_s":total_work/(median*1e3)},"compiler":state["compiler"]}
  paired=[d-f for d,f in zip(samples["direct_da"],samples["factored_da"])]
  output_diff=np.abs(outputs["direct_da"]-outputs["factored_da"])
  numerical_modes=[mode for mode in states if correctness[mode]["numerical_pass"]]
  resource_modes=[mode for mode in numerical_modes if correctness[mode]["resource_pass"]]
  passing_modes=[mode for mode in resource_modes if correctness[mode]["contraction_proof_pass"]]
  selected=min(passing_modes,key=lambda mode:statistics.median(samples[mode])) if passing_modes else None
  neither_numerical=not numerical_modes
  verdict=("FAIL_TRUSTED_WIDE_REFERENCE" if not wide_result["passed"] else
    ("FAIL_NUMERICAL_FINITE_SWEEP_EXHAUSTED" if not numerical_modes else
     ("FAIL_RESOURCE_FEASIBILITY" if not resource_modes else
      ("FAIL_CONTRACTION_SASS_PROOF" if not passing_modes else "PASS_SELECT_FASTEST_CORRECT_FEASIBLE_PROVEN"))))
  result={"schema":"tinygrad.nv_q6_oracle_fullk_tiles_gate1.v2","shape":{"M":M,"N":N,"K":K},
    "route":{"grid":[TILES,1,1],"block":[256,1,1],"tile_grid":[TILES_M,TILES_N],
      "k256_epochs_per_cta":K256,"partials":False,"fixup":False,"prefetch_second_panel":True,
      "combined_initial_publish":False,"oracle_publisher":True},
    "fixture":{"model":str(model),"weight":info.name,"format":"Q6_K"},
    "reference":{"kind":"compiler_wide_direct","trusted":bool(wide_result["passed"]),"result":wide_result},
    "arms":arms,"comparison":{"alternated_call_order":True,"rounds":args.rounds,
      "schedule":"round-index rotation with odd-round reversal; every arm appears exactly once per round",
      "call_orders":call_orders,"arm_count":len(states),
      "direct_minus_factored_us":_windows(paired),"factored_wins":sum(x>0 for x in paired),
      "direct_factored_exact":bool(np.array_equal(outputs["direct_da"],outputs["factored_da"])),
      "direct_factored_max_abs":float(output_diff.max()),"direct_factored_mean_abs":float(output_diff.mean())},
    "finite_sweep":{"complete":True,"controlled_arm_count":len(states)-2,"baseline_arm_count":2,
      "scale_groupings":["wd_yscale_then_dot","wd_then_yscale_dot"],"p_reduction_trees":P_TREES,
      "direct_contractions":["none","final"],"factored_contractions":["none","tmp_only","final_only","both"],
      "duplicate_semantic_arms":False},
    "next_branch":{"condition":"no numerical arm in the finite association/contraction sweep passes","active":neither_numerical,
      "implemented":True,"fixed_invariants":["publisher","geometry","operand_order","p_order","real_fixture","trusted_reference"],
      "scale_groupings":[{"id":"wd_yscale_then_dot","expression":"((wd * yscale) * dot)"},
        {"id":"wd_then_yscale_dot","expression":"(wd * (yscale * dot))"}],
      "p_reduction_trees":["(((u0 + u1) + u2) + u3)","((u0 + (u1 + u2)) + u3)",
        "(u0 + ((u1 + u2) + u3))","(u0 + (u1 + (u2 + u3)))","((u0 + u1) + (u2 + u3))"],
      "fma_contraction":{"parentheses_control_contraction":False,"direct":["none","final"],
        "factored":["none","tmp_only","final_only","both"],
        "control":"explicit __fmul_rn/__fadd_rn/__fmaf_rn with exact FMUL/FADD/FFMA SASS census"},
      "signed_zero_note":"Initialize ordered reductions from u0 when signed-zero identity matters."},
    "gpu_lock":{"mode":"outer_flock_required","path":"/tmp/nv-q6-oracle-gpu.lock"},
    "selection":{"numerically_passing":numerical_modes,"numerically_and_resource_passing":resource_modes,
      "fully_passing":passing_modes},"verdict":verdict,"selected_mode":selected,
    "passed":bool(wide_result["passed"] and passing_modes)}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
  print(json.dumps(result,sort_keys=True)); return 0 if result["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
