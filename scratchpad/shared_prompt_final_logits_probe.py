#!/usr/bin/env python3
"""Diagnostic-only P1-B shared-token prompt-final logits gate.

It obtains the deterministic 512-token corpus from the existing decode authority
prompt builder, runs an explicitly pinned llama.cpp CUDA dumper on exactly those
integer IDs, then runs tinygrad native NV (or CUDA) with the same IDs and prompt
position zero.  It compares a compact distributed logit sample plus each side's
argmax.  No generated token is fed back, so both engines start from identical KV.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, subprocess, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LLAMA = "/home/ubuntu/env/llama.cpp/build-cuda"
SELECTED = (0,1,2,3,10,42,151,256,512,1024,2048,4096,8192,16384,32768,49152,65536,81920,98304,114688,131072,147456,151935)

def sha(path:pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()

def prompt_ids(model_path:str, depth:int) -> list[int]:
  from extra.llm_research.decode.decode_runtime_overhead import _make_prompt
  from tinygrad.llm.gguf import gguf_load_metadata
  from tinygrad.llm.runtime_state import SimpleTokenizer
  # Metadata-only load: the corpus is the exact authority tokenizer construction
  # without reserving model weights before the llama.cpp subprocess runs.
  kv, _ = gguf_load_metadata(model_path)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  return _make_prompt(base, depth)

def run_tinygrad(model_path:str, tokens:list[int], dev:str, prefill_sample:bool=False):
  from tinygrad import Device, Tensor
  from tinygrad.llm.generate import load_model_and_tokenizer
  # Match the established native-NV prefill diagnostic setup. This affects only
  # this process and never changes a default route.  It must precede model
  # construction because the route is stored in TransformerConfig.
  if dev == "NV":
    import tinygrad.llm.model as tgm
    tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  model, _ = load_model_and_tokenizer(model_path, len(tokens)+1, seed=20260617)
  logits_t = model.logits(Tensor([tokens], dtype="int32"), 0)[:, -1, :]
  # This is the production-relevant reduction: keep it on the backend before
  # materializing the identical row for host-side numerical comparison.
  gpu_argmax = int(logits_t.argmax(-1).item())
  logits = logits_t.realize().numpy()[0]
  # Directly exercise the production prefill sampling expression on the same
  # token IDs. This distinguishes bad logits from a post-logit/sample fault.
  sampled = None
  if prefill_sample: sampled = int(model(Tensor([tokens], dtype="int32"), 0, Tensor([0.0]), use_flash=False).item())
  Device[dev].synchronize()
  argmax = int(logits.argmax())
  # Keep the runner-up explicit: an argmax match with a tiny margin would not be
  # a useful semantic gate under quantized cross-engine numerical variation.
  runner_up = int(logits.argsort()[-2])
  selected = {str(i): float(logits[i]) for i in (*SELECTED, argmax) if i < len(logits)}
  return {"engine": f"tinygrad {dev}", "prompt_tokens": len(tokens), "vocab": len(logits), "argmax": argmax, "gpu_argmax":gpu_argmax,
          "prefill_temperature_zero_sample":sampled,
          "runner_up": runner_up, "argmax_logit":float(logits[argmax]), "runner_up_logit":float(logits[runner_up]), "selected": selected}, logits

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=MODEL); ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--dev", choices=("NV", "CUDA"), default="NV")
  ap.add_argument("--production-prefill-sample", action="store_true", help="also run model(..., temp=0) after the logits check")
  ap.add_argument("--llama-dumper", default="/tmp/llama_prompt_final_logits_dump")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if args.depth < 1: raise ValueError("depth must be positive")
  tokens = prompt_ids(args.model, args.depth)
  with tempfile.TemporaryDirectory(prefix="shared_logits_") as td:
    tdir = pathlib.Path(td); token_file = tdir / "tokens.csv"; llama_out = tdir / "llama.json"
    token_file.write_text(",".join(map(str, tokens)) + "\n")
    subprocess.run([args.llama_dumper, args.model, str(token_file), str(llama_out)], check=True)
    llama = json.loads(llama_out.read_text())
    import numpy as np
    llama_logits = np.fromfile(str(llama_out)+".f32", dtype=np.float32)
  tiny, tiny_logits = run_tinygrad(args.model, tokens, args.dev, args.production_prefill_sample)
  if llama["vocab"] != tiny["vocab"]: raise RuntimeError(f"vocab mismatch {llama['vocab']} vs {tiny['vocab']}")
  common = sorted(set(llama["selected"]) | set(tiny["selected"]), key=int)
  rows = [{"id": int(i), "llama": llama["selected"].get(i), "tinygrad": tiny["selected"].get(i)} for i in common]
  diffs = [abs(r["llama"]-r["tinygrad"]) for r in rows if r["llama"] is not None and r["tinygrad"] is not None]
  import numpy as np
  if llama_logits.shape != tiny_logits.shape: raise RuntimeError(f"full logits shape mismatch {llama_logits.shape} vs {tiny_logits.shape}")
  delta = llama_logits.astype(np.float64) - tiny_logits.astype(np.float64)
  def top_ids(x, k): return set(map(int, np.argpartition(x, -k)[-k:]))
  top10_l, top10_t, top100_l, top100_t = top_ids(llama_logits, 10), top_ids(tiny_logits, 10), top_ids(llama_logits, 100), top_ids(tiny_logits, 100)
  cosine = float(np.dot(llama_logits.astype(np.float64), tiny_logits.astype(np.float64)) /
                 (np.linalg.norm(llama_logits.astype(np.float64))*np.linalg.norm(tiny_logits.astype(np.float64))))
  pearson = float(np.corrcoef(llama_logits, tiny_logits)[0, 1])
  margin = min(llama["argmax_logit"]-llama["runner_up_logit"], tiny["argmax_logit"]-tiny["runner_up_logit"])
  payload = {"schema":"nv.decode.shared_prompt_final_logits.v1", "model":str(pathlib.Path(args.model).resolve()),
    "model_sha256":sha(pathlib.Path(args.model)), "depth":args.depth,
    "token_ids_sha256":hashlib.sha256(",".join(map(str,tokens)).encode()).hexdigest(), "token_ids":tokens,
    "llama":llama, "tinygrad":tiny, "selected":rows,
    "comparison":{"argmax_equal":llama["argmax"]==tiny["argmax"], "gpu_argmax_equal":llama["argmax"]==tiny["gpu_argmax"],
                  "tinygrad_cpu_gpu_argmax_equal":tiny["argmax"]==tiny["gpu_argmax"], "runner_up_equal":llama["runner_up"]==tiny["runner_up"],
                  "min_top1_top2_margin":margin, "full_vocab":{"count":int(llama_logits.size), "mae":float(np.mean(np.abs(delta))),
                    "rmse":float(np.sqrt(np.mean(delta*delta))), "max_abs":float(np.max(np.abs(delta))), "cosine":cosine, "pearson":pearson,
                    "top10_intersection":len(top10_l&top10_t), "top100_intersection":len(top100_l&top100_t),
                    "llama_top10_tinygrad_abs_errors":{str(i):float(abs(delta[i])) for i in top10_l}},
                  "min_top1_top2_margin":margin, "max_abs":max(diffs), "mean_abs":sum(diffs)/len(diffs),
                  "declared_tolerance":{"atol":1e-2,"argmax":"exact"},
                  "passes":max(diffs)<=1e-2 and llama["argmax"]==tiny["argmax"]}}
  out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  tiny_raw = pathlib.Path(str(out)+".tinygrad.f32"); tiny_logits.astype(np.float32).tofile(tiny_raw)
  llama_raw = pathlib.Path(str(out)+".llama.f32"); llama_logits.astype(np.float32).tofile(llama_raw)
  payload["raw_logits"] = {"llama": {"path":str(llama_raw), "sha256":sha(llama_raw)}, "tinygrad":{"path":str(tiny_raw), "sha256":sha(tiny_raw)}}
  out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload["comparison"], sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
