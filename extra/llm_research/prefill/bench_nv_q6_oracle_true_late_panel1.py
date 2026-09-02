#!/usr/bin/env python3
"""Fail-closed Gate 7 for a genuinely late Q8 panel-1 preload."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, statistics
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import UOp
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_publication_gates import _classify_q8_panel1, _sass_instructions
from extra.llm_research.prefill.bench_nv_q6_oracle_reduction_policy import (
  _balanced_order, _buf, _compile_ast, _paired, _windows)
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record, _run
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import ROWS, COLS, SHARED_BYTES, q6_oracle_broad_cta_kernel
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import (
  M, N, K, OWNERS, K256, TILES_M, TILES, TILE_ELEMS, ast_signature, build_reduction_schedule,
  cpu_all_partials, emit_ordered_fixup)

ANCHOR="early_combined_all_partials"
CANDIDATE="true_late_q8_panel1"
ARMS=(ANCHOR,CANDIDATE)
ANCHOR_MAIN_SHA="6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137"
FIXUP_SHA="483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514"
LAUNCH_SHARED_BYTES=SHARED_BYTES+1024
EXPECTED_FAMILIES={"IMMA":256,"LDSM":32,"LDS":176,"LDG":109,"STS":73,"STG":64,"BAR":4,
  "I2FP":1024,"FMUL":1544,"FADD":1024,"FFMA":0}


def _ast(schedule:str) -> UOp:
  ph=lambda n,dt,i:UOp.placeholder((n,),dt,i)
  return q6_oracle_broad_cta_kernel(
    ph(2*OWNERS*TILE_ELEMS,dtypes.float32,0),ph(N*K256*105,dtypes.uint16,1),
    ph(TILES_M*K256*2*COLS*36,dtypes.uint32,2),prefetch_second_panel=True,
    q8_panel1_schedule=schedule,combined_initial_publish=True,factor_dA=False,oracle_publisher=True,
    weight_scale_contract="trusted_fp16_packed",streamk_owners=OWNERS,streamk_segment=0,
    streamk_segments_in_cta=True)


def _signature(artifact:dict) -> dict:
  sass=artifact["sass"]
  resources=sass.get("resources") or {}
  families=sass.get("families") or {}
  disassembly=pathlib.Path(artifact["sass_artifacts"]["disassembly"]).read_text()
  instructions=_sass_instructions(disassembly)
  barriers=[x for x in instructions if x["opcode"].startswith("BAR.SYNC")]
  panel=_classify_q8_panel1(disassembly)
  panel["barrier_ordinals"]=[x["ordinal"] for x in barriers]
  panel["barrier_pcs"]=[f"0x{x['pc']:x}" for x in barriers]
  return {"instruction_total":sass.get("instruction_total"),"registers":resources.get("registers"),
    "stack_bytes":resources.get("stack_bytes"),"shared_static_bytes":resources.get("shared_static_bytes"),
    "local_static_bytes":resources.get("local_static_bytes"),
    **{key:families.get(key,0) for key in (*EXPECTED_FAMILIES,"LDL","STL")},"q8_panel1":panel}


def _binary_gates(anchor:dict,candidate:dict,anchor_ast:UOp,candidate_ast:UOp) -> dict:
  panel=candidate["q8_panel1"]
  bars=panel["barrier_ordinals"]
  load=panel["panel1_first_load_ordinal"]
  store=panel["panel1_first_store_ordinal"]
  ordering=bool(len(bars)==4 and load is not None and store is not None and bars[0] < load < bars[1] < store < bars[2])
  return {
    "anchor_family_census":all(anchor[k]==v for k,v in EXPECTED_FAMILIES.items()),
    "candidate_family_census":all(candidate[k]==v for k,v in EXPECTED_FAMILIES.items()),
    "ast_signature_exact":ast_signature(candidate_ast)==ast_signature(anchor_ast),
    "panel1_18_loads_18_stores":bool(panel["classified"] and panel["panel1_loads"]==18 and panel["panel1_stores"]==18),
    "panel1_span_le_160":bool(panel["panel1_load_to_store_span_instructions"] is not None and
      panel["panel1_load_to_store_span_instructions"]<=160),
    "panel1_between_initial_and_overwrite_barriers":ordering,
    "instruction_total_le_5144":bool(candidate["instruction_total"] is not None and candidate["instruction_total"]<=5144),
    "registers_le_255":bool(candidate["registers"] is not None and candidate["registers"]<=255),
    "zero_stack":candidate["stack_bytes"]==0,
    "zero_ldl_stl":candidate["LDL"]==0 and candidate["STL"]==0,
    "shared_static_1024":candidate["shared_static_bytes"]==1024,
    "local_static_zero":candidate["local_static_bytes"]==0,
  }


def _sha_array(value:np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _write_ledger(path:pathlib.Path,result:dict) -> None:
  path.parent.mkdir(parents=True,exist_ok=True)
  if not result.get("gpu_work_started",False):
    failed=[name for name,passed in result["binary_gates"].items() if not passed]
    text=("# NV Q6 true-late Q8 panel-1 Gate 7 decision (2026-08-31)\n\n"
      "## Decision\n\n`BLOCKED_BEFORE_GPU`\n\n"
      f"The bounded schedule repair failed these predeclared binary gates: `{', '.join(failed)}`. "
      "No GPU correctness or timing work was started.\n\n"
      f"Evidence: `{result['evidence_path']}`\n")
  else:
    sig=result["signatures"][CANDIDATE]; panel=sig["q8_panel1"]
    timing=result["timing"]; paired=result["paired"]
    text=("# NV Q6 true-late Q8 panel-1 Gate 7 decision (2026-08-31)\n\n"
      f"## Decision\n\n`{result['verdict']}`\n\n"
      "The only candidate change is a dependency-constrained panel-1 preload after the penultimate half-0 p-group. "
      "Packed trusted-FP16 arithmetic, 170-owner one-body ownership, combined initial publication, all-partials output, and the frozen fixup remain unchanged.\n\n"
      "## Binary gate\n\n"
      f"- Candidate cubin: `{result['compiler'][CANDIDATE]['cubin_sha256']}`\n"
      f"- Frozen fixup cubin: `{result['compiler']['fixup']['cubin_sha256']}`\n"
      f"- Panel-1 first load/store: `{panel['panel1_first_load_pc']}` / `{panel['panel1_first_store_pc']}`\n"
      f"- Panel-1 span: `{panel['panel1_load_to_store_span_instructions']}` instructions\n"
      f"- Instructions/registers/stack/LDL-STL: `{sig['instruction_total']}` / `{sig['registers']}` / "
      f"`{sig['stack_bytes']} B` / `{sig['LDL']}/{sig['STL']}`\n\n"
      "## Correctness\n\n"
      f"- Partials/final uint32 exact: `{result['exactness']['partials_bit_exact']}` / `{result['exactness']['final_bit_exact']}`\n"
      f"- Trusted maximum/mean error: `{result['correctness'][CANDIDATE]['reference_max_abs']}` / "
      f"`{result['correctness'][CANDIDATE]['reference_mean_abs']}`\n"
      f"- Trusted failing elements: `{result['correctness'][CANDIDATE]['reference_failing_count']}`\n\n"
      "## Locked R31\n\n"
      "| Arm | Main median | Fixup median | Total median |\n|---|---:|---:|---:|\n"
      f"| Anchor | {timing[ANCHOR]['main']['r31']['median_us']:.3f} us | {timing[ANCHOR]['fixup']['r31']['median_us']:.3f} us | {timing[ANCHOR]['total']['r31']['median_us']:.3f} us |\n"
      f"| True late | {timing[CANDIDATE]['main']['r31']['median_us']:.3f} us | {timing[CANDIDATE]['fixup']['r31']['median_us']:.3f} us | {timing[CANDIDATE]['total']['r31']['median_us']:.3f} us |\n\n"
      f"Main candidate-anchor paired median/wins: `{paired['main']['r31']['median_us']:.3f} us`, "
      f"`{paired['main']['candidate_wins']}/31`. Total: `{paired['total']['r31']['median_us']:.3f} us`, "
      f"`{paired['total']['candidate_wins']}/31`.\n\n"
      f"GPU lock acquired: `{result['gpu_lock']['acquired']}`. Evidence: `{result['evidence_path']}`\n")
  path.write_text(text)


def _blocked(args,compiler,signatures,gates,reason) -> int:
  result={"schema":"tinygrad.nv_q6_true_late_panel1.v1","verdict":"BLOCKED_BEFORE_GPU",
    "reason":reason,"compiler":compiler,"signatures":signatures,"binary_gates":gates,
    "gpu_work_started":False,"gpu_lock":{"path":"/tmp/nv-q6-oracle-gpu.lock",
      "acquired":os.environ.get("NV_Q6_GPU_LOCK_HELD")=="1"},"evidence_path":str(args.out)}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
  _write_ledger(args.ledger,result);print(json.dumps(result,sort_keys=True));return 2


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=31);ap.add_argument("--warmups",type=int,default=3)
  ap.add_argument("--out",type=pathlib.Path,required=True);ap.add_argument("--artifacts",type=pathlib.Path,required=True)
  ap.add_argument("--ledger",type=pathlib.Path,required=True);args=ap.parse_args()
  if args.rounds!=31 or args.warmups!=3:raise ValueError("Gate 7 requires R31 with three warmups")
  args.artifacts.mkdir(parents=True,exist_ok=True)

  asts={ANCHOR:_ast("early"),CANDIDATE:_ast("true_late_tail")}
  compiler={};states={};signatures={}
  for arm in ARMS:
    name,binary,artifact=_compile_ast(asts[arm],arm,args.artifacts)
    compiler[arm]=artifact;signatures[arm]=_signature(artifact)
    states[arm]={"program":NVProgram(Device["NV"],name,binary,shared_mem=LAUNCH_SHARED_BYTES)}
  anchor_frozen=compiler[ANCHOR]["cubin_sha256"]==ANCHOR_MAIN_SHA

  p=lambda n,t,i:UOp.placeholder((n,),t,i)
  fix_ast=emit_ordered_fixup(p(M*N,dtypes.float32,0),p(2*OWNERS*TILE_ELEMS,dtypes.float32,1),
    p(TILES*3,dtypes.int32,2),p(TILES,dtypes.int32,3),direct_final=False)
  fix_name,fix_binary,fix_artifact=_compile_ast(fix_ast,"frozen_all_partials_fixup",args.artifacts)
  compiler["fixup"]=fix_artifact
  fixup_frozen=fix_artifact["cubin_sha256"]==FIXUP_SHA
  gates=_binary_gates(signatures[ANCHOR],signatures[CANDIDATE],asts[ANCHOR],asts[CANDIDATE])
  gates={"anchor_cubin_frozen":anchor_frozen,"fixup_cubin_frozen":fixup_frozen,**gates}
  if not all(gates.values()):return _blocked(args,compiler,signatures,gates,"true-late schedule is not representable within the frozen binary/resource envelope")

  schedule=build_reduction_schedule();slot_map,counts,_=schedule.arrays()
  slot_map_t=Tensor(slot_map.reshape(-1),device="NV").contiguous().realize()
  counts_t=Tensor(counts,device="NV").contiguous().realize()
  fix=NVProgram(Device["NV"],fix_name,fix_binary)

  model=pathlib.Path(args.model);meta=read_metadata(model)
  info=next(x for x in meta.infos if x.name=="blk.0.ffn_down.weight")
  if info.typ!=GGML_Q6_K:raise RuntimeError(info)
  halfs=packed_u16_slice(model,meta,info,device="NV").contiguous().realize()
  wide_host,q,scales=wide_record(M,K);wide_q8=Tensor(wide_host,device="NV").contiguous().realize()
  broad=[]
  for mt in range(TILES_M):
    for epoch in range(K256):
      broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),
        np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
  q8=Tensor(np.concatenate(broad).reshape(-1),device="NV").contiguous().realize()

  wide_artifacts=args.artifacts/"wide";wide_artifacts.mkdir(parents=True,exist_ok=True)
  direct=_run("wide_direct",M,N,K,halfs,wide_q8,args.rounds,wide_artifacts,(128,128,2,4,256))
  direct_source=(wide_artifacts/"wide_direct.cu").read_text();direct_binary=Device["NV"].compiler.compile(direct_source)
  match=re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(',direct_source)
  if match is None:raise RuntimeError("trusted direct symbol missing")
  reference=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
  NVProgram(Device["NV"],match.group(1),direct_binary)(_buf(reference),_buf(wide_q8),_buf(halfs),
    global_size=(32,4,1),local_size=(32,2,4),wait=True)
  expected=reference.numpy()

  def launch_main(state):
    return state["program"](_buf(state["partials"]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),
      local_size=(256,1,1),wait=True,timeout=120000)*1e6
  def launch_fixup(state):
    return fix(_buf(state["output"]),_buf(state["partials"]),_buf(slot_map_t),_buf(counts_t),
      global_size=(TILES,1,1),local_size=(256,1,1),wait=True)*1e6

  correctness={};raws={};outputs={};active=set(schedule.active_slots);unused=sorted(set(range(2*OWNERS))-active)
  for arm in ARMS:
    state=states[arm]
    state["partials"]=Tensor.full((2*OWNERS*TILE_ELEMS),float("nan"),device="NV").contiguous().realize()
    state["output"]=Tensor.full((M,N),float("nan"),device="NV").contiguous().realize()
    launch_main(state);raw=state["partials"].numpy().reshape(2*OWNERS,ROWS,COLS);launch_fixup(state)
    got=state["output"].numpy();cpu=cpu_all_partials(raw,schedule);diff=np.abs(got-expected)
    close=np.isclose(got,expected,rtol=2e-5,atol=2e-3);raws[arm]=raw;outputs[arm]=got
    correctness[arm]={"finite":bool(np.isfinite(got).all()),
      "gpu_fixup_cpu_bit_exact":bool(np.array_equal(got.view(np.uint32),cpu.view(np.uint32))),
      "written_slots_finite":bool(np.isfinite(raw[sorted(active)]).all()),"unused_slots_nan":bool(np.isnan(raw[unused]).all()),
      "reference_max_abs":float(diff.max()),"reference_mean_abs":float(diff.mean()),
      "reference_failing_count":int(np.count_nonzero(~close)),"reference_passed":bool(close.all()),
      "full_partial_sha256":_sha_array(raw),"final_output_sha256":_sha_array(got)}
  exactness={"partials_bit_exact":bool(np.array_equal(raws[CANDIDATE].view(np.uint32),raws[ANCHOR].view(np.uint32))),
    "final_bit_exact":bool(np.array_equal(outputs[CANDIDATE].view(np.uint32),outputs[ANCHOR].view(np.uint32)))}
  correctness_pass=bool(direct["passed"] and exactness["partials_bit_exact"] and exactness["final_bit_exact"] and
    all(all(rec[k] for k in ("finite","gpu_fixup_cpu_bit_exact","written_slots_finite","unused_slots_nan","reference_passed"))
      for rec in correctness.values()) and
    correctness[CANDIDATE]["reference_max_abs"]<=correctness[ANCHOR]["reference_max_abs"] and
    correctness[CANDIDATE]["reference_mean_abs"]<=correctness[ANCHOR]["reference_mean_abs"])

  for warmup in range(args.warmups):
    for arm in _balanced_order(ARMS,warmup):launch_main(states[arm]);launch_fixup(states[arm])
  samples={arm:{"main":[],"fixup":[],"total":[]} for arm in ARMS};orders=[]
  for round_index in range(args.rounds):
    order=_balanced_order(ARMS,round_index);orders.append(list(order))
    for arm in order:
      main_us=launch_main(states[arm]);fixup_us=launch_fixup(states[arm])
      samples[arm]["main"].append(main_us);samples[arm]["fixup"].append(fixup_us);samples[arm]["total"].append(main_us+fixup_us)
  timing={arm:{kind:_windows(values) for kind,values in record.items()} for arm,record in samples.items()}
  paired={kind:_paired(samples[CANDIDATE][kind],samples[ANCHOR][kind]) for kind in ("main","fixup","total")}
  performance={"main":paired["main"]["r31"]["median_us"]<=-3.0 and paired["main"]["candidate_wins"]>=24,
    "total":paired["total"]["r31"]["median_us"]<=-3.0 and paired["total"]["candidate_wins"]>=24}
  promoted=bool(correctness_pass and all(performance.values()))
  verdict="PROMOTE_TRUE_LATE_Q8_PANEL1" if promoted else (
    "REJECT_TRUE_LATE_Q8_PANEL1_PERFORMANCE" if correctness_pass else "FAIL_CLOSED_CORRECTNESS")
  result={"schema":"tinygrad.nv_q6_true_late_panel1.v1","shape":{"M":M,"N":N,"K":K},
    "launch":{"grid":[OWNERS,1,1],"block":[256,1,1],"shared_bytes":LAUNCH_SHARED_BYTES},
    "arms":{"anchor":"early combined all-partials","candidate":"dependency-constrained true-late tail preload"},
    "binary_gates":gates,"signatures":signatures,"compiler":compiler,"ownership":schedule.json_record(),
    "reference":{"kind":"compiler_wide_direct","result":direct},"correctness":correctness,"exactness":exactness,
    "correctness_passed":correctness_pass,"timing":timing,"paired":paired,"performance_gates":performance,
    "launch_orders":orders,"thresholds":{"median_improvement_us":3.0,"minimum_wins":24,"rounds":31,"warmups":3},
    "gpu_work_started":True,"gpu_lock":{"path":"/tmp/nv-q6-oracle-gpu.lock",
      "acquired":os.environ.get("NV_Q6_GPU_LOCK_HELD")=="1"},"promotion_eligible":promoted,
    "verdict":verdict,"evidence_path":str(args.out)}
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+"\n")
  _write_ledger(args.ledger,result);print(json.dumps(result,sort_keys=True))
  return 0 if correctness_pass else 1


if __name__=="__main__":raise SystemExit(main())
