#!/usr/bin/env python3
"""Memory-bounded fixed-depth decode coverage in one continuous request."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, sys, time

from extra.llm_research.decode.decode_harness import DEFAULT_MODEL
from extra.llm_research.decode.decode_runtime_overhead import (
  _atomic_json, _captured_program_count, _captured_program_evidence, _decode_jits, _make_prompt,
  _model_identity, _nv_gpu_state, _prefill, _route, _token_evidence, _used_decode_jits)


def _census_payload(census, selected:dict, count:int, depth:int) -> dict:
  payload = census.to_dict()
  programs = {name:_captured_program_evidence(jit) for name,jit in selected.items()}
  sources = {}
  for rows in programs.values():
    for row in rows:
      source = row.pop("source_text")
      if source is None: continue
      source_sha = row["source_sha256"]
      if source_sha in sources and sources[source_sha] != source: raise RuntimeError(f"source SHA collision: {source_sha}")
      sources[source_sha] = source
  payload["capture"] = {"phase":"decode", "fixed_depth":depth, "selected_jits":list(selected),
    "warmup_decode":count, "programs_by_jit":{name:_captured_program_count(jit) for name,jit in selected.items()},
    "program_evidence_by_jit":programs, "source_text_by_sha256":sources, "captured":True}
  return payload


def main(argv:list[str]|None=None) -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL)); ap.add_argument("--depth", type=int, required=True)
  ap.add_argument("--max-context", type=int, required=True); ap.add_argument("--nmeas", type=int, default=10)
  ap.add_argument("--reps", type=int, default=3); ap.add_argument("--warmup-decode", type=int, default=3)
  ap.add_argument("--chunk-size", type=int, default=32); ap.add_argument("--gpu-state", action="store_true")
  ap.add_argument("--request-scoped-prewarm", action="store_true"); ap.add_argument("--census-out", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True); args=ap.parse_args(argv)
  if min(args.depth,args.max_context,args.nmeas,args.reps,args.warmup_decode,args.chunk_size) < 1: raise ValueError("all sizes must be positive")
  if args.depth + args.warmup_decode + args.nmeas*args.reps >= args.max_context: raise ValueError("request exceeds max context")

  from tinygrad import Device, UOp
  from tinygrad.engine.jit import observe_graph_admissions
  from tinygrad.helpers import Context
  from tinygrad.llm.generate import load_model_and_tokenizer

  started_process = time.perf_counter()
  dev=Device[Device.DEFAULT]; model,tokenizer=load_model_and_tokenizer(args.model,args.max_context,seed=20260617)
  base=(tokenizer.prefix() if hasattr(tokenizer,"prefix") else [])+tokenizer.encode("the quick brown fox jumps. "*800)
  prompt=_make_prompt(base,args.depth); model.reset_generation_state()
  horizon=args.warmup_decode+args.nmeas*args.reps+1 if args.request_scoped_prewarm else None
  gen,prelude=_prefill(model,prompt,args.chunk_size,horizon)
  before={name:jit.cnt for name,jit in _decode_jits(model).items()}
  try:
    with Context(TRACEMETA=1), observe_graph_admissions() as census:
      warm_tokens=[int(next(gen)) for _ in range(args.warmup_decode)]
    selected=_used_decode_jits(model,before)
    if not selected or any(jit.captured is None for jit in selected.values()):
      raise RuntimeError(f"warmup did not capture selected JITs: {list(selected)}")
    _atomic_json(args.census_out.resolve(),_census_payload(census,selected,args.warmup_decode,args.depth))
    reps=[]
    for rep in range(args.reps):
      dev.synchronize(); gpu_before=_nv_gpu_state() if args.gpu_state and Device.DEFAULT == "NV" else None
      latencies=[]; tokens=[]
      for _ in range(args.nmeas):
        started=time.perf_counter(); tokens.append(int(next(gen))); latencies.append(time.perf_counter()-started)
      gpu_after=_nv_gpu_state() if args.gpu_state and Device.DEFAULT == "NV" else None
      reps.append({"rep":rep,"elapsed_s":sum(latencies),"tok_s":args.nmeas/sum(latencies),
                   "per_token_ms":[x*1e3 for x in latencies],"token_evidence":_token_evidence(tokens,include_token_ids=True),
                   "gpu_state_before":gpu_before,"gpu_state_after":gpu_after})
  finally: gen.close()
  route_sp=UOp.variable("reported_start_pos",0,args.max_context-1).bind(args.depth)
  artifact={"schema":"tinygrad.decode.continuous_context_coverage.v1","created_unix_ns":time.time_ns(),
    "model":_model_identity(args.model),"device":{"tinygrad_device":Device.DEFAULT,"runtime_type":type(dev).__name__},
    "workload":{"depth":args.depth,"max_context":args.max_context,"nmeas":args.nmeas,"reps":args.reps,
      "warmup_decode":args.warmup_decode,"chunk_size":args.chunk_size,"request_scoped_prewarm":args.request_scoped_prewarm,
      "measurement_scope":"three sequential windows after excluded capture/warmup in one continuous request"},
    "route":"flash" if _route(model,route_sp,1) else "sdpa","selected_jits":list(selected),
    "programs_per_token_by_selected_jit":{name:_captured_program_count(jit) for name,jit in selected.items()},
    "prefill_token_evidence":_token_evidence([prelude],include_token_ids=True),
    "warmup_token_evidence":_token_evidence(warm_tokens,include_token_ids=True),
    "reps":reps,"median_ms_per_token":1e3/statistics.median(x["tok_s"] for x in reps),
    "median_tok_s":statistics.median(x["tok_s"] for x in reps),"process_elapsed_s":time.perf_counter()-started_process,
    "tool":{"path":str(pathlib.Path(__file__).resolve()),"argv":list(sys.argv if argv is None else argv)}}
  _atomic_json(args.out.resolve(),artifact); print(json.dumps({"out":str(args.out),"median_tok_s":artifact["median_tok_s"]})); return 0


if __name__ == "__main__": raise SystemExit(main())
