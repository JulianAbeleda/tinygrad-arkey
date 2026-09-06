#!/usr/bin/env python3
"""Closed-default native decode qualification for two-capture token feedback.

Run each arm in a fresh process.  ``logits`` is the semantic gate, ``census``
proves equal in-graph program count plus the alias/lifetime contract, and
``timing`` is only for a GPU-authorized reverse A/B/A after both gates pass.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, pathlib, re, statistics, time
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from tinygrad.llm.feedback_pingpong import pingpong_capture_contract


TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")
GRAPH_TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+batched\s+(\d+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")


def _model(arm:str, model_path:str, max_context:int):
  model = _load(model_path, max_context)
  model._decode_direct_greedy_promoted = arm in ("greedy", "pingpong")
  model._decode_feedback_pingpong_promoted = arm == "pingpong"
  return model


def _redirect() -> int:
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  return int(bool(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT))


def _pingpong_pairs(model, diagnostic:bool=False) -> dict[str, tuple]:
  prefix = "rollout_greedy_logits_pingpong_jits" if diagnostic else "rollout_greedy_pingpong_jits"
  pairs = {prefix:getattr(model, prefix), prefix+"_flash":getattr(model, prefix+"_flash"),
           prefix+"_flash_s6":getattr(model, prefix+"_flash_s6"), prefix+"_flash_s64":getattr(model, prefix+"_flash_s64")}
  pairs.update({f"{prefix}_flash_live[{key}]":pair for key,pair in getattr(model, prefix+"_flash_live").items()})
  return pairs


def _warmed_pair(model, diagnostic:bool=False):
  warm = [pair for pair in _pingpong_pairs(model, diagnostic).values()
          if all(getattr(jit, "captured", None) is not None for jit in pair)]
  if len(warm) != 1: return None
  return warm[0]


def _pair_counts(model, diagnostic:bool=False) -> dict[str, tuple[int, int]]:
  return {name:tuple(jit.cnt for jit in pair) for name,pair in _pingpong_pairs(model, diagnostic).items()}


def _selected_pair(model, before:dict[str, tuple[int, int]], diagnostic:bool=False):
  changed = [(name,pair) for name,pair in _pingpong_pairs(model, diagnostic).items()
             if tuple(jit.cnt for jit in pair) != before.get(name)]
  return changed[0] if len(changed) == 1 else (None, None)


def logits(arm:str, model_path:str, depth:int, count:int, max_context:int, eager:bool=False) -> tuple[dict, np.ndarray]:
  from extra.llm_research.decode.decode_runtime_overhead import _decode_jits
  model = _model(arm, model_path, max_context)
  # diagnostic_full_logits deliberately skips production's automatic prewarm.
  # Mark that completed state so its separate diagnostic JIT selects the same
  # active-horizon split as the steady production route under qualification.
  model._flash_decode_active_horizon_prewarmed = True
  pair_before = _pair_counts(model, diagnostic=True)
  jit_before = {name:jit.cnt for name,jit in _decode_jits(model).items()}
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0, diagnostic_full_logits=True)
  tokens, rows = [], []
  try:
    next(gen)  # prefill return intentionally has no comparable full-logit row
    selected_names = set()
    for _ in range(max(count, 6)):
      if eager:
        for jit in _decode_jits(model).values(): jit.cnt = 0
        step_before = {name:jit.cnt for name,jit in _decode_jits(model).items()}
      token, row = next(gen)
      if eager: selected_names.update(name for name,jit in _decode_jits(model).items() if jit.cnt != step_before.get(name))
      if row is None: continue
      array, sampled = row.numpy(), int(token)
      if not np.isfinite(array).all(): raise RuntimeError("non-finite full logits")
      if len(rows) < count: tokens.append(sampled); rows.append(array)
  finally: gen.close()
  stacked = np.stack(rows)
  pair_name,pair = _selected_pair(model,pair_before,diagnostic=True) if not eager else (None,None)
  used = [(name,jit) for name,jit in _decode_jits(model).items() if jit.cnt != jit_before.get(name)]
  used_names = sorted(selected_names) if eager else [name for name,_ in used]
  contract = pingpong_capture_contract(pair) if arm == "pingpong" and pair is not None else None
  argmax = [int(row.argmax(axis=-1).item()) for row in rows]
  return {"schema":"tinygrad.nv.feedback_pingpong_qualification.v1", "arm":arm, "mode":"logits", "callify_redirect":_redirect(), "tokens":tokens,
          "argmax_tokens":argmax, "sample_argmax_match":tokens == argmax,
          "shape":list(stacked.shape), "logits_sha256":hashlib.sha256(np.ascontiguousarray(stacked).view(np.uint8)).hexdigest(),
          "selector_state":"active_horizon_prewarm_completed", "execution":"eager_no_capture" if eager else "captured",
          "selected_pair":pair_name, "selected_jits":used_names,
          "route_still_promoted":bool(getattr(model, "_decode_feedback_pingpong_promoted", False)), "contract":contract}, stacked


def census(arm:str, model_path:str, depth:int, max_context:int, request_scoped:bool=False) -> dict:
  from extra.llm_research.decode.decode_runtime_overhead import _decode_jits
  model = _model(arm, model_path, max_context)
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0,
                       expected_output_tokens=8 if request_scoped else None)
  try:
    next(gen)
    for _ in range(6): next(gen)
    capture = io.StringIO(); before = _pair_counts(model); jit_before = {name:jit.cnt for name,jit in _decode_jits(model).items()}
    with contextlib.redirect_stdout(capture):
      from tinygrad.helpers import Context
      with Context(DEBUG=2): token = int(next(gen))
  finally: gen.close()
  rows=[]; graph_rows=[]
  for line in capture.getvalue().splitlines():
    if (match:=TM_RE.match(line)):
      rows.append((match.group(1), float(match.group(2))*(1000 if match.group(3)=="ms" else 1)))
    elif (match:=GRAPH_TM_RE.match(line)):
      graph_rows.append((int(match.group(1)), float(match.group(2))*(1000 if match.group(3)=="ms" else 1)))
  pair_name,pair = _selected_pair(model,before)
  contract = pingpong_capture_contract(pair) if arm == "pingpong" and pair is not None else None
  used = [(name,jit) for name,jit in _decode_jits(model).items() if jit.cnt != jit_before.get(name)]
  used_name,used_jit = used[0] if len(used) == 1 else (None,None)
  shadows = len(used_jit.captured._written_input_shadows) if used_jit is not None and used_jit.captured is not None else None
  return {"schema":"tinygrad.nv.feedback_pingpong_qualification.v1", "arm":arm, "mode":"census", "callify_redirect":_redirect(), "token":token,
          "request_scoped_prewarm":request_scoped,
          "program_count":len(rows)+sum(x[0] for x in graph_rows), "graph_group_sizes":[x[0] for x in graph_rows],
          "kernel_us":sum(x[1] for x in rows)+sum(x[1] for x in graph_rows), "selected_pair":pair_name, "contract":contract,
          "selected_jit":used_name, "selected_jit_written_input_shadows":shadows,
          "route_still_promoted":bool(getattr(model, "_decode_feedback_pingpong_promoted", False)),
          "program_names":[x[0] for x in rows], "raw_debug":capture.getvalue().splitlines()[-20:]}


def timing(arm:str, model_path:str, depth:int, count:int, reps:int, max_context:int, gpu_state:bool=False, request_scoped:bool=False) -> dict:
  from tinygrad import Device
  model, dev, samples, hashes, states, selected_pairs = _model(arm, model_path, max_context), Device[Device.DEFAULT], [], [], [], []
  for _ in range(reps):
    model.reset_generation_state(); model._decode_direct_greedy_promoted = arm in ("greedy", "pingpong")
    model._decode_feedback_pingpong_promoted = arm == "pingpong"
    gen, output = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0,
                                 expected_output_tokens=count+7 if request_scoped else None), []
    try:
      next(gen)
      for _ in range(6): next(gen)  # both arms captured before included timing
      dev.synchronize()
      before = None
      if gpu_state:
        from extra.llm_research.decode.decode_runtime_overhead import _nv_gpu_state
        before = _nv_gpu_state()
      pair_before = _pair_counts(model); started=time.perf_counter_ns()
      for _ in range(count): output.append(int(next(gen)))
      dev.synchronize(); samples.append((time.perf_counter_ns()-started)/count/1e6)
      after = _nv_gpu_state() if gpu_state else None
      states.append({"before":before, "after":after})
      selected_pairs.append(_selected_pair(model,pair_before)[0])
    finally: gen.close()
    hashes.append(hashlib.sha256(",".join(map(str, output)).encode()).hexdigest())
  selected_name = selected_pairs[0] if selected_pairs and len(set(selected_pairs)) == 1 else None
  pair = _pingpong_pairs(model).get(selected_name) if selected_name is not None else None
  return {"schema":"tinygrad.nv.feedback_pingpong_qualification.v1", "arm":arm, "mode":"timing", "callify_redirect":_redirect(), "samples_ms":samples,
          "request_scoped_prewarm":request_scoped,
          "median_ms":statistics.median(samples), "token_hashes":hashes, "tokens_identical":len(set(hashes))==1,
          "gpu_state_scope":"immediately_before_and_after_timed_decode_window" if gpu_state else None, "gpu_states":states,
          "selected_pairs":selected_pairs, "contract":pingpong_capture_contract(pair) if arm == "pingpong" and pair is not None else None}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm", choices=("legacy","greedy","pingpong"), required=True)
  ap.add_argument("--mode", choices=("logits","census","timing"), required=True); ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--gpu-state", action="store_true", help="record NV state immediately around each timed decode window")
  ap.add_argument("--eager-logits", action="store_true", help="keep diagnostic decode eager to avoid duplicate-graph memory")
  ap.add_argument("--request-scoped-prewarm", action="store_true", help="prewarm only the request's measured output horizon")
  ap.add_argument("--depth", type=int, default=512); ap.add_argument("--count", type=int, default=8); ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--max-context", type=int, default=1024); ap.add_argument("--out", type=pathlib.Path, required=True); args=ap.parse_args()
  if args.mode == "logits": result,array=logits(args.arm,args.model,args.depth,args.count,args.max_context,args.eager_logits); np.savez_compressed(args.out.with_suffix(".npz"),logits=array)
  elif args.mode == "census": result=census(args.arm,args.model,args.depth,args.max_context,args.request_scoped_prewarm)
  else: result=timing(args.arm,args.model,args.depth,args.count,args.reps,args.max_context,args.gpu_state,args.request_scoped_prewarm)
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
