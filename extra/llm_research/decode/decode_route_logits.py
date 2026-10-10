#!/usr/bin/env python3
"""Full-logit check for one decode route change, control against candidate.

One arm per process: the route's rollback switch set (--switch NAME, e.g. TINYGRAD_Q6K_ROLE_AGNOSTIC_DISABLE) is the
control, unset is the candidate. Both arms prefill the same prompt, then decode COUNT steps teacher-forced on the same continuation
tokens, so a logit difference can only come from the decode GEMV. `--compare A.json B.json` reports the gate."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

def run(a) -> dict:
  from tinygrad import Tensor, UOp, getenv
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  ids = _prompt(a.model, a.depth + a.count)
  model = _load(a.model, a.max_context)
  installed = sorted({type(x).__name__ for x in model._q4k_linears.linears})
  q8 = [x for x in model._q4k_linears.linears if hasattr(x, "block_quant_storage")]
  by_kind = {k: sorted({(x.out_features, x.in_features) for x in model._q4k_linears.linears if type(x).__name__ == k}) for k in installed}
  gen = model.generate(ids[:a.depth], chunk_size=32, temperature=0.0)
  try: next(gen)
  finally: gen.close()
  temp, sp, logits = Tensor([0.0]), UOp.variable("start_pos", 0, a.max_context - 1), []
  for i in range(a.count):
    token = Tensor([[ids[a.depth + i]]], dtype="int32").contiguous()
    _sample, full = model.decode_with_logits(token, sp.bind(a.depth + i), temp)
    logits.append(full.numpy().astype(np.float32).reshape(-1))
  stack = np.stack(logits); out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out.with_suffix(".npz"), logits=stack)
  ret = {"schema": "tinygrad.decode_route_logits.v1", "switch": a.switch, "switch_set": bool(getenv(a.switch, 0)) if a.switch else None,
         "shapes_by_kind": {k: [list(x) for x in v] for k, v in by_kind.items()}, "model": a.model, "depth": a.depth, "count": a.count,
         "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
         "q8_0_gemv_disabled": bool(getenv("TINYGRAD_Q8_0_GEMV_DISABLE", 0)), "primitive_kinds": installed,
         "q8_0_linears": len(q8), "q8_0_shapes": sorted({(x.out_features, x.in_features) for x in q8}),
         "argmax": [int(x) for x in stack.argmax(-1)], "finite": bool(np.isfinite(stack).all()),
         "logits_sha256": hashlib.sha256(stack.tobytes()).hexdigest()}
  out.write_text(json.dumps(ret, indent=2) + "\n"); print(json.dumps(ret)); return ret

def compare(a_path:str, b_path:str, rtol:float) -> dict:
  A, B = (json.loads(pathlib.Path(p).read_text()) for p in (a_path, b_path))
  la, lb = (np.load(pathlib.Path(p).with_suffix(".npz"))["logits"].astype(np.float64) for p in (a_path, b_path))
  scale = np.abs(la).max(axis=-1, keepdims=True)
  rel = np.abs(la - lb) / scale
  top5 = [len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) for x, y in zip(la, lb)]
  ret = {"control": a_path, "candidate": b_path, "bitwise_equal": A["logits_sha256"] == B["logits_sha256"],
         "argmax_equal": A["argmax"] == B["argmax"], "max_abs": float(np.abs(la - lb).max()),
         "max_rel_to_step_max": float(rel.max()), "top5_overlap_min": int(min(top5)), "rtol": rtol}
  ret["pass"] = bool(ret["argmax_equal"] and ret["max_rel_to_step_max"] <= rtol and A["finite"] and B["finite"])
  print(json.dumps(ret)); return ret

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model"); ap.add_argument("--depth", type=int, default=128); ap.add_argument("--count", type=int, default=16)
  ap.add_argument("--max-context", type=int, default=512); ap.add_argument("--out")
  ap.add_argument("--compare", nargs=2); ap.add_argument("--rtol", type=float, default=1e-3)
  ap.add_argument("--switch", default="TINYGRAD_Q8_0_GEMV_DISABLE")
  a = ap.parse_args()
  if a.compare: return 0 if compare(*a.compare, a.rtol)["pass"] else 1
  run(a); return 0

if __name__ == "__main__": raise SystemExit(main())
