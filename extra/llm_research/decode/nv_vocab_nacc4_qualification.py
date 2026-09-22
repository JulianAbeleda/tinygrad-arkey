#!/usr/bin/env python3
"""Fresh-process semantic and wall gate for the closed-default vocab nacc4 lease."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt


def _model(arm: str, model_path: str, max_context: int):
  model = _load(model_path, max_context)
  if arm == "candidate": model.output._decode_vocab_accumulators_lease = 4
  return model


def logits(arm: str, model_path: str, depth: int, count: int, max_context: int, out: pathlib.Path) -> dict:
  model = _model(arm, model_path, max_context)
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0, diagnostic_full_logits=True)
  tokens, rows = [], []
  try:
    next(gen)
    while len(rows) < count:
      token, row = next(gen)
      if row is None: continue
      array = row.numpy().astype(np.float32)
      if not np.isfinite(array).all(): raise RuntimeError("non-finite logits")
      tokens.append(int(token)); rows.append(array)
  finally: gen.close()
  stacked = np.stack(rows)
  np.savez_compressed(out.with_suffix(".npz"), logits=stacked)
  return {"schema":"tinygrad.nv_vocab_nacc4_qualification.v1", "mode":"logits", "arm":arm,
    "tokens":tokens, "argmax_tokens":[int(x.argmax(axis=-1).item()) for x in rows],
    "shape":list(stacked.shape), "sha256":hashlib.sha256(np.ascontiguousarray(stacked).view(np.uint8)).hexdigest()}


def timing(arm: str, model_path: str, depth: int, count: int, reps: int, max_context: int) -> dict:
  from tinygrad import Device
  model, dev, samples, hashes = _model(arm, model_path, max_context), Device[Device.DEFAULT], [], []
  for _ in range(reps):
    model.reset_generation_state()
    if arm == "candidate": model.output._decode_vocab_accumulators_lease = 4
    gen, output = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0), []
    try:
      next(gen)
      for _ in range(6): next(gen)
      dev.synchronize(); started=time.perf_counter_ns()
      for _ in range(count): output.append(int(next(gen)))
      dev.synchronize(); samples.append((time.perf_counter_ns()-started)/count/1e6)
    finally: gen.close()
    hashes.append(hashlib.sha256(",".join(map(str,output)).encode()).hexdigest())
  return {"schema":"tinygrad.nv_vocab_nacc4_qualification.v1", "mode":"timing", "arm":arm,
    "samples_ms":samples, "median_ms":statistics.median(samples), "token_hashes":hashes,
    "tokens_identical":len(set(hashes)) == 1}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=("control","candidate"),required=True)
  ap.add_argument("--mode",choices=("logits","timing"),required=True); ap.add_argument("--model",default=DEFAULT_MODEL)
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=4)
  ap.add_argument("--reps",type=int,default=7); ap.add_argument("--max-context",type=int,default=768)
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  args.out.parent.mkdir(parents=True,exist_ok=True)
  result = logits(args.arm,args.model,args.depth,args.count,args.max_context,args.out) if args.mode == "logits" else \
    timing(args.arm,args.model,args.depth,args.count,args.reps,args.max_context)
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
