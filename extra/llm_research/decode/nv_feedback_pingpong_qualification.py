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


def _warmed_pair(model, diagnostic:bool=False):
  candidates = ((model.rollout_greedy_logits_pingpong_jits_flash, model.rollout_greedy_logits_pingpong_jits)
                if diagnostic else (model.rollout_greedy_pingpong_jits_flash, model.rollout_greedy_pingpong_jits))
  warm = [pair for pair in candidates if all(getattr(jit, "captured", None) is not None for jit in pair)]
  if len(warm) != 1: return None
  return warm[0]


def logits(arm:str, model_path:str, depth:int, count:int, max_context:int) -> tuple[dict, np.ndarray]:
  model = _model(arm, model_path, max_context)
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0, diagnostic_full_logits=True)
  tokens, rows = [], []
  try:
    next(gen)  # prefill return intentionally has no comparable full-logit row
    for _ in range(max(count, 6)):
      token, row = next(gen)
      if row is None: continue
      array, sampled = row.numpy(), int(token)
      if not np.isfinite(array).all(): raise RuntimeError("non-finite full logits")
      if len(rows) < count: tokens.append(sampled); rows.append(array)
  finally: gen.close()
  stacked = np.stack(rows)
  pair = _warmed_pair(model, diagnostic=True)
  contract = pingpong_capture_contract(pair) if arm == "pingpong" and pair is not None else None
  argmax = [int(row.argmax(axis=-1).item()) for row in rows]
  return {"schema":"tinygrad.nv.feedback_pingpong_qualification.v1", "arm":arm, "mode":"logits", "callify_redirect":_redirect(), "tokens":tokens,
          "argmax_tokens":argmax, "sample_argmax_match":tokens == argmax,
          "shape":list(stacked.shape), "logits_sha256":hashlib.sha256(np.ascontiguousarray(stacked).view(np.uint8)).hexdigest(),
          "route_still_promoted":bool(getattr(model, "_decode_feedback_pingpong_promoted", False)), "contract":contract}, stacked


def census(arm:str, model_path:str, depth:int, max_context:int) -> dict:
  model = _model(arm, model_path, max_context)
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try:
    next(gen)
    for _ in range(6): next(gen)
    capture = io.StringIO()
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
  pair = _warmed_pair(model)
  contract = pingpong_capture_contract(pair) if arm == "pingpong" and pair is not None else None
  return {"schema":"tinygrad.nv.feedback_pingpong_qualification.v1", "arm":arm, "mode":"census", "callify_redirect":_redirect(), "token":token,
          "program_count":len(rows)+sum(x[0] for x in graph_rows), "graph_group_sizes":[x[0] for x in graph_rows],
          "kernel_us":sum(x[1] for x in rows)+sum(x[1] for x in graph_rows), "contract":contract,
          "route_still_promoted":bool(getattr(model, "_decode_feedback_pingpong_promoted", False)),
          "program_names":[x[0] for x in rows], "raw_debug":capture.getvalue().splitlines()[-20:]}


def timing(arm:str, model_path:str, depth:int, count:int, reps:int, max_context:int) -> dict:
  from tinygrad import Device
  model, dev, samples, hashes = _model(arm, model_path, max_context), Device[Device.DEFAULT], [], []
  for _ in range(reps):
    model.reset_generation_state(); model._decode_direct_greedy_promoted = arm in ("greedy", "pingpong")
    model._decode_feedback_pingpong_promoted = arm == "pingpong"
    gen, output = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0), []
    try:
      next(gen)
      for _ in range(6): next(gen)  # both arms captured before included timing
      dev.synchronize(); started=time.perf_counter_ns()
      for _ in range(count): output.append(int(next(gen)))
      dev.synchronize(); samples.append((time.perf_counter_ns()-started)/count/1e6)
    finally: gen.close()
    hashes.append(hashlib.sha256(",".join(map(str, output)).encode()).hexdigest())
  pair = _warmed_pair(model)
  return {"schema":"tinygrad.nv.feedback_pingpong_qualification.v1", "arm":arm, "mode":"timing", "callify_redirect":_redirect(), "samples_ms":samples,
          "median_ms":statistics.median(samples), "token_hashes":hashes, "tokens_identical":len(set(hashes))==1,
          "contract":pingpong_capture_contract(pair) if arm == "pingpong" and pair is not None else None}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm", choices=("legacy","greedy","pingpong"), required=True)
  ap.add_argument("--mode", choices=("logits","census","timing"), required=True); ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512); ap.add_argument("--count", type=int, default=8); ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--max-context", type=int, default=1024); ap.add_argument("--out", type=pathlib.Path, required=True); args=ap.parse_args()
  if args.mode == "logits": result,array=logits(args.arm,args.model,args.depth,args.count,args.max_context); np.savez_compressed(args.out.with_suffix(".npz"),logits=array)
  elif args.mode == "census": result=census(args.arm,args.model,args.depth,args.max_context)
  else: result=timing(args.arm,args.model,args.depth,args.count,args.reps,args.max_context)
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
