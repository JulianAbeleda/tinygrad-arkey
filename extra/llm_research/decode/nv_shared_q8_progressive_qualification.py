#!/usr/bin/env python3
"""Fresh-process progressive qualification for the closed shared-Q8 hook.

This is an evidence harness, not a route selector.  A child installs explicit
leases after load; normal loads have no lease. Ordinary-provider progression
uses leading blocks, while fused RMSNorm progression uses blocks 1..N and
leaves block 0's embedding/gather provenance untouched. Parents run fresh,
lock-bounded subprocesses for full-logit qualification or included-cost timing.
Nothing in this module selects or promotes a route.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys, time
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission

GROUP_STEPS = (1, 2, 4, 18)
FUSED_GROUP_STEPS = (1, 2, 4, 18, 35)
COOPERATIVE_GROUP_STEPS = (1, 2, 4, 8, 12, 18, 35)
DEFAULT_PRIMITIVE_ORACLE = "docs/task_workflow/output/nv-fused-rmsnorm-q8-provider-gate-20260805.json"
SEMANTIC_CONTRACT = {"top_k":10, "relative_l2_max":1e-3, "perturbation_vs_min_margin_max":1.0,
                     "historical_max_abs_atol":0.01}

def _digest(a:np.ndarray) -> str: return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def _install_leases(model, groups:int) -> list[int]:
  from tinygrad import Tensor, dtypes
  if groups not in (0, *GROUP_STEPS): raise ValueError(f"groups must be 0 or one of {GROUP_STEPS}")
  if groups > len(model.blk): raise ValueError(f"requested {groups} groups, model has {len(model.blk)} blocks")
  # Models are fresh per child. Still explicitly clear the attribute so a
  # caller cannot accidentally reuse a model with a prior qualification lease.
  for block in model.blk:
    if hasattr(block, "_shared_q8_attention_admission"): delattr(block, "_shared_q8_attention_admission")
    if hasattr(block, "_shared_q8_attention_norm_weight"): delattr(block, "_shared_q8_attention_norm_weight")
    if hasattr(block, "_decode_reduce_output_attn_rmsnorm_promoted"): delattr(block, "_decode_reduce_output_attn_rmsnorm_promoted")
  for index in range(groups):
    block=model.blk[index]
    block._shared_q8_attention_admission = SharedQ8AttentionAdmission(index)
    # The model's GGUF norm weight is a lazy packed view. A leased block owns a
    # dedicated fp16 resident copy so the per-token fused provider has only
    # physical x/weight inputs; this is load-time setup, not hidden token work.
    block._shared_q8_attention_norm_weight = Tensor.empty(4096,dtype=dtypes.float16,device=block.attn_norm.weight.device).assign(
      block.attn_norm.weight.cast(dtypes.float16).contiguous()).realize()
  return list(range(groups))

def _install_fused_norm_leases(model, groups:int, cooperative_q4:bool=False, q6_direct_output:bool=False) -> list[int]:
  """Lease blocks 1..groups, excluding block 0's non-REDUCE_OUTPUT provenance."""
  allowed=COOPERATIVE_GROUP_STEPS if cooperative_q4 else FUSED_GROUP_STEPS
  if groups not in allowed or groups >= len(model.blk):
    raise ValueError(f"fused groups must be one of {allowed} and leave block 0 unleased")
  return _install_fused_norm_lease_indices(model,range(1,groups+1),cooperative_q4,q6_direct_output)

def _normalize_fused_indices(spec:str) -> tuple[int,...]:
  if not spec.strip(): return ()
  try: values=tuple(int(x) for x in spec.split(","))
  except ValueError as exc: raise ValueError("--fused-indices must be comma-separated integers") from exc
  if any(x <= 0 for x in values): raise ValueError("fused indices must exclude block 0")
  if len(set(values)) != len(values): raise ValueError("fused indices must be unique")
  if tuple(sorted(values)) != values: raise ValueError("fused indices must be strictly increasing")
  return values

def _install_fused_norm_lease_indices(model, indices, cooperative_q4:bool=False, q6_direct_output:bool=False) -> list[int]:
  """Install an explicit default-off block subset for precision-budget search."""
  from tinygrad import Tensor, dtypes
  indices=list(indices)
  if not indices or any(not isinstance(x,int) or isinstance(x,bool) or x <= 0 or x >= len(model.blk) for x in indices):
    raise ValueError("explicit fused lease indices must be nonempty valid block indices excluding block 0")
  if indices != sorted(set(indices)): raise ValueError("explicit fused lease indices must be unique and increasing")
  for block in model.blk:
    for name in ("_shared_q8_attention_admission", "_shared_q8_attention_norm_weight",
                 "_decode_reduce_output_attn_rmsnorm_promoted"):
      if hasattr(block,name): delattr(block,name)
  for block_index in indices:
    block=model.blk[block_index]
    block._shared_q8_attention_admission=SharedQ8AttentionAdmission(block_index,cooperative_q4=cooperative_q4,
                                                                       q6_direct_output=q6_direct_output)
    block._decode_reduce_output_attn_rmsnorm_promoted=True
    block._shared_q8_attention_norm_weight=Tensor.empty(4096,dtype=dtypes.float16,device=block.attn_norm.weight.device).assign(
      block.attn_norm.weight.cast(dtypes.float16).contiguous()).realize()
  return indices

def child(model_path:str, depth:int, count:int, max_context:int, groups:int, fused_groups:int=0,
          cooperative_q4:bool=False, composed:bool=False, fused_indices:tuple[int,...]=(),
          q6_direct_output:bool=False) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  from tinygrad.helpers import Context
  model = _load(model_path, max_context)
  if sum(bool(x) for x in (groups,fused_groups,fused_indices)) > 1:
    raise ValueError("ordinary, cumulative fused, and explicit fused leases are mutually exclusive")
  leases = _install_fused_norm_lease_indices(model,fused_indices,cooperative_q4,q6_direct_output) if fused_indices else \
    _install_fused_norm_leases(model,fused_groups,cooperative_q4,q6_direct_output) if fused_groups else _install_leases(model,groups)
  model._decode_direct_greedy_promoted=composed
  model._decode_feedback_pingpong_promoted=composed
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try: prelude = int(next(gen))
  finally: gen.close()
  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, max_context-1)
  with Context(JIT=0): _, eager = model.forward_with_logits(token, start_pos.bind(depth), temp)
  eager_np = eager.numpy()
  if not np.isfinite(eager_np).all(): raise RuntimeError("eager full logits non-finite")
  logits,tokens=[],[]
  # Ping-pong owns two distinct JIT captures. A GraphAdmissionCensus binds one
  # call index to one CALL identity, so each slot requires its own observer.
  censuses=[GraphAdmissionCensus() for _ in range(2 if composed else 1)]
  for i in range(count):
    with observe_graph_admissions(censuses[i&1 if composed else 0]):
      sampled, full = model.decode_with_logits(token, start_pos.bind(depth+1+i), temp,
                                                feedback_slot=(i&1) if composed else None)
      sampled_id, full_np = int(sampled.item()), full.numpy()
      # This is the P0 binding gate, not merely a model-output comparison.
      # A retained diagnostic return must be the logits used by *this* sampler
      # on every capture/replay iteration.  Without it a stale returned logits
      # buffer can make a candidate look numerically acceptable while the
      # sampled feedback follows a different tensor.
      logits_argmax = int(full_np.reshape(-1).argmax())
      if sampled_id != logits_argmax:
        raise RuntimeError(f"diagnostic/sample binding mismatch iteration={i} sampled={sampled_id} logits_argmax={logits_argmax}")
    tokens.append(sampled_id); logits.append(full_np); token=sampled
  arr = np.stack(logits)
  if not np.isfinite(arr).all(): raise RuntimeError("decode full logits non-finite")
  programs = [r.program_name for census in censuses for r in census.records if r.program_name]
  fused_provider_count=programs.count("rmsnorm_q8_1_llama_provider_4096")
  ordinary_provider_count=programs.count("q8_1_llama_provider_4096")
  provider_count=fused_provider_count+ordinary_provider_count
  coop_q4_count=sum(p.startswith("q4k_warp_coop_q8_dp4a_partial_") for p in programs)
  q6_direct_count=sum(p.startswith("q6k_q8_warp_direct_") for p in programs)
  legacy_q4_count=sum(p.startswith("q4k_q8_dp4a_") for p in programs)
  capture_factor=2 if composed else 1; expected_providers=len(leases)*capture_factor
  expected_q6_direct=0
  if q6_direct_output:
    from tinygrad.llm.qk_primitives import Q6KPrimitiveLinear
    expected_q6_direct=sum(isinstance(model.blk[index].attn_v,Q6KPrimitiveLinear) for index in leases)*capture_factor
  if (fused_groups or fused_indices) and fused_provider_count != expected_providers:
    raise RuntimeError(f"fused qualification expected {expected_providers} RMSNorm/Q8 providers, observed {fused_provider_count}")
  if cooperative_q4 and (provider_count != expected_providers or coop_q4_count < 2*expected_providers or legacy_q4_count):
    raise RuntimeError(f"cooperative census failed providers={provider_count}/{expected_providers} coop_q4={coop_q4_count} legacy_q4={legacy_q4_count}")
  # Qwen has mixed Q4/Q4/Q4 and Q4/Q4/Q6 attention groups. Count only actual
  # Q6 V leases; a Q4 V is intentionally untouched by this Q6-only route.
  if q6_direct_output and q6_direct_count != expected_q6_direct:
    raise RuntimeError(f"Q6 direct census expected {expected_q6_direct} Q6-V consumers, observed {q6_direct_count}")
  return ({"schema":"tinygrad.nv_shared_q8_progressive_qualification.v1", "groups":groups,
    "fused_groups":fused_groups, "leases":leases,"cooperative_q4":cooperative_q4,"q6_direct_output":q6_direct_output,"composed":composed,
    "fused_indices":list(fused_indices),
    "depth":depth, "count":count, "prelude_token":prelude, "tokens":tokens, "tokens_sha256":hashlib.sha256(
      ",".join(map(str,tokens)).encode()).hexdigest(), "logits_sha256":_digest(arr), "shape":list(arr.shape),
    "finite":True, "program_count":len(programs), "shared_q8_program_count":sum(
      p.startswith(("q4k_q8_dp4a_", "q4k_warp_coop_q8_dp4a_partial_", "q6k_q8_dp4a_", "q6k_q8_warp_direct_")) for p in programs),
    "capture_factor":capture_factor,"q8_provider_count":provider_count,"cooperative_q4_consumer_count":coop_q4_count,
    "q6_direct_consumer_count":q6_direct_count,"q6_direct_expected_count":expected_q6_direct,
    "legacy_q4_shared_consumer_count":legacy_q4_count,
    "fused_rmsnorm_q8_provider_count":fused_provider_count,
    "program_names":programs, "graph_census":[census.to_dict() for census in censuses]}, arr)

def _child_command(args, groups:int, out:pathlib.Path, fused_groups:int=0, mode:str="child") -> list[str]:
  cmd=[sys.executable, str(pathlib.Path(__file__).resolve()), "--mode", mode, "--model", args.model,
          "--depth", str(args.depth), "--count", str(args.count), "--max-context", str(args.max_context),
          "--groups", str(groups), "--fused-groups", str(fused_groups), "--reps",str(args.reps),"--out", str(out)]
  if getattr(args,"cooperative_q4",False): cmd.append("--cooperative-q4")
  if getattr(args,"q6_direct_output",False): cmd.append("--q6-direct-output")
  if getattr(args,"composed",False): cmd.append("--composed")
  if getattr(args,"settled_continuous",False): cmd.append("--settled-continuous")
  if getattr(args,"fused_indices",()): cmd += ["--fused-indices",",".join(map(str,args.fused_indices))]
  return cmd

def _settled_context_required(depth:int,count:int,reps:int) -> int:
  """Prompt + prelude + six decode warmups + every uninterrupted timed token."""
  return depth+1+6+reps*count

def _validate_run_extent(depth:int,count:int,max_context:int,reps:int,settled_continuous:bool) -> None:
  if depth < 1 or count < 1 or max_context <= depth+count: raise ValueError("invalid depth/count/context")
  if reps < 1: raise ValueError("reps must be positive")
  if settled_continuous and max_context < _settled_context_required(depth,count,reps):
    raise ValueError(f"settled continuous run needs max_context >= {_settled_context_required(depth,count,reps)}")

def _settled_high_side_filter(samples:list[float]) -> tuple[float,list[float],list[float]]:
  med=statistics.median(samples); mad=statistics.median(abs(x-med) for x in samples)
  high_limit=med+max(0.25,6*mad)
  accepted=[x for x in samples if x <= high_limit]
  if len(accepted)<3: return high_limit,samples.copy(),[]
  rejected=[x for x in samples if x > high_limit]
  return high_limit,accepted,rejected

def _settled_continuous_windows(gen,dev,count:int,reps:int,clock=None) -> dict:
  """Consume one generator continuously; pure apart from next/synchronize/clock."""
  clock=time.perf_counter_ns if clock is None else clock
  prelude=int(next(gen)); warmup_tokens=[int(next(gen)) for _ in range(6)]; dev.synchronize()
  samples,hashes,timed_tokens=[],[],[]
  for _ in range(reps):
    started=clock(); window=[int(next(gen)) for _ in range(count)]; dev.synchronize()
    samples.append((clock()-started)/count/1e6); timed_tokens.extend(window)
    hashes.append(hashlib.sha256(",".join(map(str,window)).encode()).hexdigest())
  high_limit,accepted,rejected=_settled_high_side_filter(samples)
  return {"prelude_token":prelude,"warmup_tokens":warmup_tokens,"samples_ms_per_token":samples,
    "accepted_samples_ms_per_token":accepted,"external_contention_high_limit_ms":high_limit,
    "rejected_high_samples_ms_per_token":rejected,"median_ms_per_token":statistics.median(accepted),"token_hashes":hashes,
    "token_stream_hash":hashlib.sha256(",".join(map(str,timed_tokens)).encode()).hexdigest(),"timed_token_count":len(timed_tokens)}

def _timing_hash_authority(rows:list[dict],settled_continuous:bool) -> set[str]:
  return {row["token_stream_hash"] for row in rows} if settled_continuous else {h for row in rows for h in row["token_hashes"]}

def qualify(args) -> dict:
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  rows, arrays, comparisons = {}, {}, []
  for groups in (0, *GROUP_STEPS):
    out = root / f"groups-{groups}.json"
    # The outer timeout covers lock acquisition too; this prevents a blocked
    # GPU queue from looking like an unbounded model hang.
    cmd = ["timeout", f"{args.timeout}s", "flock", "-w", str(args.lock_wait), args.lock, *_child_command(args, groups, out)]
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if run.returncode:
      raise RuntimeError(f"groups={groups} child failed rc={run.returncode}: {run.stderr[-4000:]}")
    rows[groups] = json.loads(out.read_text()); arrays[groups] = np.load(out.with_suffix(".npz"))["logits"]
    if groups:
      baseline, candidate = arrays[0], arrays[groups]; delta = np.abs(candidate-baseline)
      denom = np.maximum(np.abs(baseline), 1e-12)
      comparison = {"groups":groups, "finite":bool(np.isfinite(candidate).all()),
        "tokens_equal":rows[groups]["tokens"] == rows[0]["tokens"], "max_abs":float(delta.max()),
        "max_rel":float((delta/denom).max()), "argmax_equal":bool(np.array_equal(candidate.argmax(-1), baseline.argmax(-1))),
        "atol":args.atol, "numeric_pass":bool(np.all(delta <= args.atol)),
        "topology_win":rows[groups]["program_count"] < rows[0]["program_count"]}
      comparison["gate_pass"] = all(comparison[k] for k in ("finite","tokens_equal","argmax_equal","numeric_pass","topology_win"))
      comparisons.append(comparison)
      if not comparison["gate_pass"]: break
  return {"schema":"tinygrad.nv_shared_q8_progressive_qualification.v1", "mode":"qualify", "children":rows,
          "comparisons":comparisons, "note":"qualification metrics only; no timing or promotion credit"}

def _primitive_oracle(path:str) -> dict:
  payload=json.loads(pathlib.Path(path).read_text())
  cases=payload.get("cases",[])
  passed=payload.get("gate") == "PASS" and len(cases) >= 3 and all(
    x.get("bitwise_equal") is True and x.get("mismatch_count") == 0 for x in cases)
  return {"path":path,"schema":payload.get("schema"),"case_count":len(cases),"bitwise_exact":passed}

def _semantic_comparison(baseline:np.ndarray, candidate:np.ndarray, baseline_row:dict, candidate_row:dict) -> dict:
  delta=(candidate-baseline).astype(np.float64)
  flat_base,flat_candidate=baseline.reshape((-1,baseline.shape[-1])),candidate.reshape((-1,candidate.shape[-1]))
  k=SEMANTIC_CONTRACT["top_k"]
  base_topk=np.argsort(flat_base,axis=-1)[:,-k:]
  cand_topk=np.argsort(flat_candidate,axis=-1)[:,-k:]
  topk_sets_equal=all(set(a.tolist()) == set(b.tolist()) for a,b in zip(base_topk,cand_topk))
  ordered_base=np.argsort(flat_base,axis=-1)[:,::-1][:,:k]
  ordered_candidate=np.argsort(flat_candidate,axis=-1)[:,::-1][:,:k]
  top2=np.partition(flat_base,-2,axis=-1)[:,-2:]
  min_margin=float(np.min(np.max(top2,axis=-1)-np.min(top2,axis=-1)))
  max_abs=float(np.max(np.abs(delta)))
  # Any pairwise logit gap can move by at most 2*L-inf perturbation.
  perturbation_vs_min_margin=float((2*max_abs)/min_margin) if min_margin > 0 else float("inf")
  relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(baseline.astype(np.float64).ravel()),1e-30))
  old_atol=SEMANTIC_CONTRACT["historical_max_abs_atol"]
  out={"finite":bool(np.isfinite(candidate).all()),"tokens_equal":candidate_row["tokens"]==baseline_row["tokens"],
    "argmax_equal":bool(np.array_equal(candidate.argmax(-1),baseline.argmax(-1))),
    "top_k":k,"top_k_sets_equal":topk_sets_equal,"top_k_order_equal":bool(np.array_equal(ordered_base,ordered_candidate)),
    "relative_l2":relative_l2,"relative_l2_max":SEMANTIC_CONTRACT["relative_l2_max"],
    "min_baseline_top1_margin":min_margin,"max_abs":max_abs,"perturbation_vs_min_margin":perturbation_vs_min_margin,
    "perturbation_vs_min_margin_max":SEMANTIC_CONTRACT["perturbation_vs_min_margin_max"],
    "historical_max_abs_gate":{"atol":old_atol,"pass":max_abs <= old_atol,"status":"reported_non_authoritative"}}
  out["semantic_pass"]=all((out["finite"],out["tokens_equal"],out["argmax_equal"],out["top_k_sets_equal"],
    relative_l2 <= SEMANTIC_CONTRACT["relative_l2_max"],perturbation_vs_min_margin < SEMANTIC_CONTRACT["perturbation_vs_min_margin_max"]))
  return out

def qualify_fused(args) -> dict:
  if args.fused_indices: raise ValueError("fused-qualify owns cumulative groups; use child for explicit subsets")
  root=pathlib.Path(args.out).with_suffix(""); root.mkdir(parents=True,exist_ok=True)
  oracle=_primitive_oracle(args.primitive_oracle)
  if not oracle["bitwise_exact"]: raise RuntimeError(f"primitive oracle is not bitwise exact: {oracle}")
  rows,arrays,comparisons={}, {}, []
  steps=COOPERATIVE_GROUP_STEPS if args.cooperative_q4 else FUSED_GROUP_STEPS
  for fused_groups in (0,*steps):
    label=f"fused-g{fused_groups}"
    out=root/f"{label}.json"
    cmd=["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,*_child_command(args,0,out,fused_groups)]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"{label} child failed rc={run.returncode}: {run.stderr[-4000:]}")
    rows[label]=json.loads(out.read_text()); arrays[label]=np.load(out.with_suffix(".npz"))["logits"]
    if fused_groups:
      comparison=_semantic_comparison(arrays["fused-g0"],arrays[label],rows["fused-g0"],rows[label])
      comparison.update({"fused_groups":fused_groups,
        "provider_count_exact":rows[label]["fused_rmsnorm_q8_provider_count"]==fused_groups*rows[label]["capture_factor"],
        "cooperative_q4_count":rows[label]["cooperative_q4_consumer_count"],
        "legacy_q4_absent":rows[label]["legacy_q4_shared_consumer_count"]==0,
        "no_duplicate_provider":rows[label]["q8_provider_count"]==fused_groups*rows[label]["capture_factor"],
        "topology_win":rows[label]["program_count"]<rows["fused-g0"]["program_count"]})
      comparison["gate_pass"]=oracle["bitwise_exact"] and comparison["semantic_pass"] and comparison["provider_count_exact"] and \
        comparison["legacy_q4_absent"] and comparison["no_duplicate_provider"] and \
        comparison["cooperative_q4_count"]>=2*fused_groups*rows[label]["capture_factor"]
      comparisons.append(comparison)
      if not comparison["gate_pass"]: break
  return {"schema":"tinygrad.nv_shared_q8_progressive_qualification.v1","mode":"fused-qualify",
          "primitive_oracle":oracle,"semantic_contract":SEMANTIC_CONTRACT,"children":rows,"comparisons":comparisons,
          "note":"intentional llama-Q8 semantic qualification; historical max-abs remains reported but is not the authority gate"}

def timing_child(model_path:str,depth:int,count:int,max_context:int,reps:int,fused_groups:int,
                 cooperative_q4:bool=False,composed:bool=False,settled_continuous:bool=False,
                 fused_indices:tuple[int,...]=(),q6_direct_output:bool=False) -> dict:
  from tinygrad import Device
  model=_load(model_path,max_context)
  if fused_groups and fused_indices: raise ValueError("cumulative and explicit fused leases are mutually exclusive")
  leases=_install_fused_norm_lease_indices(model,fused_indices,cooperative_q4,q6_direct_output) if fused_indices else \
    _install_fused_norm_leases(model,fused_groups,cooperative_q4,q6_direct_output) if fused_groups else []
  model._decode_direct_greedy_promoted=composed; model._decode_feedback_pingpong_promoted=composed
  prompt,dev=_prompt(model_path,depth),Device[Device.DEFAULT]
  samples,hashes=[],[]
  # Included cost: generation runs the fused provider and every selected Q/K/V
  # consumer inside the normal token graph. Load-time resident norm copies are
  # deliberately outside token time in both correctness and timing modes.
  if settled_continuous:
    # TinyJit call 0 is eager, call 1 captures, and call >=2 replays. With two
    # alternating feedback slots, six untimed decode calls put both slots on
    # true replay before any timer starts. One uninterrupted generator then
    # prevents reset/capture/lifecycle work from entering later rep windows.
    gen=model.generate(prompt.copy(),chunk_size=32,temperature=0.0)
    try:
      settled=_settled_continuous_windows(gen,dev,count,reps)
    finally: gen.close()
    return {"schema":"tinygrad.nv_shared_q8_progressive_timing.v1","arm":f"fused-g{fused_groups}",
      "fused_groups":fused_groups,"leases":leases,"included_cost":True,"cooperative_q4":cooperative_q4,"q6_direct_output":q6_direct_output,"composed":composed,
      "fused_indices":list(fused_indices),
      "settled_continuous":True,"warmup_decode_calls":6,"reps":reps,"tokens_per_rep":count,**settled}
  warm=model.generate(prompt.copy(),chunk_size=32,temperature=0.0)
  try:
    for _ in range(3): next(warm)
  finally: warm.close()
  for _ in range(reps):
    model.reset_generation_state(); model._decode_direct_greedy_promoted=composed; model._decode_feedback_pingpong_promoted=composed
    gen=model.generate(prompt.copy(),chunk_size=32,temperature=0.0); tokens=[]
    try:
      next(gen); dev.synchronize(); started=time.perf_counter_ns()
      for _ in range(count): tokens.append(int(next(gen)))
      dev.synchronize(); samples.append((time.perf_counter_ns()-started)/count/1e6)
    finally: gen.close()
    hashes.append(hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest())
  return {"schema":"tinygrad.nv_shared_q8_progressive_timing.v1","arm":f"fused-g{fused_groups}",
    "fused_groups":fused_groups,"leases":leases,"included_cost":True,"cooperative_q4":cooperative_q4,"q6_direct_output":q6_direct_output,"composed":composed,
    "fused_indices":list(fused_indices),
    "reps":reps,"tokens_per_rep":count,
    "samples_ms_per_token":samples,"median_ms_per_token":statistics.median(samples),
    "token_hashes":hashes,"tokens_identical_within_arm":len(set(hashes))==1}

def qualify_fused_timing(args) -> dict:
  if args.fused_indices: raise ValueError("fused-timing owns cumulative groups; use subset-timing")
  allowed=COOPERATIVE_GROUP_STEPS if args.cooperative_q4 else FUSED_GROUP_STEPS
  if args.fused_groups not in allowed: raise ValueError(f"--fused-groups must be one of {allowed}")
  root=pathlib.Path(args.out).with_suffix(""); root.mkdir(parents=True,exist_ok=True)
  rows=[]
  # Reverse bracket guards drift. Every child owns a fresh model/JIT graph.
  for sequence,fused_groups in enumerate((0,args.fused_groups,0)):
    label=f"{'control' if fused_groups == 0 else 'candidate'}-{sequence}"
    out=root/f"{label}.json"
    cmd=["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
         *_child_command(args,0,out,fused_groups,mode="timing-child")]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"{label} child failed rc={run.returncode}: {run.stderr[-4000:]}")
    rows.append(json.loads(out.read_text()))
  hashes=_timing_hash_authority(rows,args.settled_continuous)
  control_median=statistics.median((rows[0]["median_ms_per_token"],rows[2]["median_ms_per_token"]))
  candidate=rows[1]["median_ms_per_token"]
  return {"schema":"tinygrad.nv_shared_q8_progressive_timing.v1","mode":"fused-timing-reverse-bracket",
    "fused_groups":args.fused_groups,"arms":rows,"all_token_hashes_equal":len(hashes)==1,
    "control_bracket_median_ms":control_median,"candidate_ms":candidate,"candidate_minus_control_ms":candidate-control_median,
    "candidate_speedup_pct":(control_median/candidate-1)*100,
    "settled_continuous":args.settled_continuous,
    "note":"included-cost native wall evidence only; qualification/promotion requires a separately passing semantic gate"}

def qualify_subset_timing(args) -> dict:
  if args.fused_groups or not args.fused_indices:
    raise ValueError("subset-timing requires --fused-indices and --fused-groups 0")
  root=pathlib.Path(args.out).with_suffix(""); root.mkdir(parents=True,exist_ok=True)
  rows=[]
  for sequence,indices in enumerate(((),args.fused_indices,())):
    label=f"{'control' if not indices else 'candidate'}-{sequence}"
    out=root/f"{label}.json"
    child_args=argparse.Namespace(**vars(args)); child_args.fused_indices=indices
    cmd=["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
         *_child_command(child_args,0,out,0,mode="timing-child")]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"{label} child failed rc={run.returncode}: {run.stderr[-4000:]}")
    rows.append(json.loads(out.read_text()))
  hashes=_timing_hash_authority(rows,args.settled_continuous)
  control_median=statistics.median((rows[0]["median_ms_per_token"],rows[2]["median_ms_per_token"]))
  candidate=rows[1]["median_ms_per_token"]
  return {"schema":"tinygrad.nv_shared_q8_subset_timing.v1","mode":"subset-timing-reverse-bracket",
    "fused_indices":list(args.fused_indices),"arms":rows,"all_token_hashes_equal":len(hashes)==1,
    "control_bracket_median_ms":control_median,"candidate_ms":candidate,"candidate_minus_control_ms":candidate-control_median,
    "candidate_speedup_pct":(control_median/candidate-1)*100,"settled_continuous":args.settled_continuous,
    "note":"explicit default-off subset; semantic authority is a separate child comparison"}

def qualify_subset_incremental_timing(args) -> dict:
  if args.fused_groups or not args.fused_indices or not args.reference_indices:
    raise ValueError("subset-incremental-timing requires explicit candidate and reference indices")
  if not set(args.reference_indices) < set(args.fused_indices):
    raise ValueError("reference indices must be a strict subset of candidate indices")
  root=pathlib.Path(args.out).with_suffix(""); root.mkdir(parents=True,exist_ok=True)
  rows=[]
  for sequence,indices in enumerate((args.reference_indices,args.fused_indices,args.reference_indices)):
    label=f"{'reference' if sequence != 1 else 'candidate'}-{sequence}"
    out=root/f"{label}.json"
    child_args=argparse.Namespace(**vars(args)); child_args.fused_indices=indices
    cmd=["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,
         *_child_command(child_args,0,out,0,mode="timing-child")]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode: raise RuntimeError(f"{label} child failed rc={run.returncode}: {run.stderr[-4000:]}")
    rows.append(json.loads(out.read_text()))
  hashes=_timing_hash_authority(rows,args.settled_continuous)
  reference=statistics.median((rows[0]["median_ms_per_token"],rows[2]["median_ms_per_token"]))
  candidate=rows[1]["median_ms_per_token"]
  return {"schema":"tinygrad.nv_shared_q8_subset_incremental_timing.v1","mode":"subset-incremental-timing-reverse-bracket",
    "reference_indices":list(args.reference_indices),"candidate_indices":list(args.fused_indices),"arms":rows,
    "all_token_hashes_equal":len(hashes)==1,"reference_bracket_median_ms":reference,"candidate_ms":candidate,
    "candidate_minus_reference_ms":candidate-reference,"candidate_speedup_pct":(reference/candidate-1)*100,
    "settled_continuous":args.settled_continuous,"note":"incremental credit only; do not add cumulative and incremental rows"}

def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("child","qualify","fused-qualify","timing-child","fused-timing","subset-timing","subset-incremental-timing"),required=True)
  ap.add_argument("--model",default=os.environ.get("QK_MODEL","/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")); ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--count",type=int,default=8); ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--groups",type=int,default=0)
  ap.add_argument("--fused-groups",type=int,default=0); ap.add_argument("--reps",type=int,default=3)
  ap.add_argument("--fused-indices",default="",help="research-only explicit increasing block-index subset")
  ap.add_argument("--reference-indices",default="",help="strict-subset reference for incremental settled timing")
  ap.add_argument("--out",required=True); ap.add_argument("--timeout",type=int,default=600); ap.add_argument("--lock-wait",type=int,default=90); ap.add_argument("--lock",default="/tmp/gpu-bench.lock")
  ap.add_argument("--atol",type=float,default=0.01)
  ap.add_argument("--primitive-oracle",default=DEFAULT_PRIMITIVE_ORACLE)
  ap.add_argument("--cooperative-q4",action="store_true"); ap.add_argument("--q6-direct-output",action="store_true"); ap.add_argument("--composed",action="store_true")
  ap.add_argument("--settled-continuous",action="store_true")
  a=ap.parse_args()
  a.fused_indices=_normalize_fused_indices(a.fused_indices)
  a.reference_indices=_normalize_fused_indices(a.reference_indices)
  _validate_run_extent(a.depth,a.count,a.max_context,a.reps,a.settled_continuous)
  if a.mode == "child":
    row, logits = child(a.model,a.depth,a.count,a.max_context,a.groups,a.fused_groups,a.cooperative_q4,a.composed,a.fused_indices,a.q6_direct_output); out=pathlib.Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    # Native NV can retain driver/runtime objects during interpreter teardown.
    # Publish both artifacts atomically, flush the machine-readable stdout row,
    # then bypass teardown. A parent may therefore trust only final paths and
    # never confuse a complete computation with a teardown-only hang.
    npz_out, npz_tmp = out.with_suffix(".npz"), out.with_suffix(".npz.tmp")
    json_tmp = out.with_name(out.name+".tmp")
    with npz_tmp.open("wb") as f:
      np.savez_compressed(f,logits=logits); f.flush(); os.fsync(f.fileno())
    json_tmp.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
    os.replace(npz_tmp,npz_out); os.replace(json_tmp,out)
    print(json.dumps(row,sort_keys=True),flush=True); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
  elif a.mode == "timing-child":
    result=timing_child(a.model,a.depth,a.count,a.max_context,a.reps,a.fused_groups,a.cooperative_q4,a.composed,a.settled_continuous,a.fused_indices,a.q6_direct_output)
    out=pathlib.Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True),flush=True); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
  elif a.mode == "qualify":
    result=qualify(a); pathlib.Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))
  elif a.mode == "fused-qualify":
    result=qualify_fused(a); pathlib.Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))
  elif a.mode == "fused-timing":
    result=qualify_fused_timing(a); pathlib.Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))
  elif a.mode == "subset-timing":
    result=qualify_subset_timing(a); pathlib.Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))
  else:
    result=qualify_subset_incremental_timing(a); pathlib.Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))
  return 0

if __name__ == "__main__": raise SystemExit(main())
