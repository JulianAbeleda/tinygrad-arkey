"""Static contract harness for the provenance-clean many-row Q6 vocabulary arm.

This is intentionally default-off and GPU-free.  It records the exact ABI that
the existing generated primitive exposes; model integration is a later packet.
"""
from __future__ import annotations
import argparse, json
import time
from pathlib import Path

from tinygrad.llm.q6k_vocab_manyrow import K, Q8_GROUPS, Q8_WORDS, ROWS

CANDIDATE = "nv_vocab_manyrow_q8_q6"

def contract() -> dict:
  return {
    "schema": "nv_q6k_vocab_manyrow_native_candidate/v1",
    "candidate_id": CANDIDATE,
    "default_enabled": False,
    "device": "NV",
    "origin": "tinygrad_generated",
    "binary_provenance": "tinygrad UOp KernelProgram emitters",
    "producer": {"identity": "nv_vocab_manyrow/q8_provider", "input": [1, 1, K],
                 "output": [Q8_WORDS], "dtype": "uint32", "groups": Q8_GROUPS},
    "consumer": {"identity": "nv_vocab_manyrow/q6_mmvq", "weights": {"dtype": "uint16", "shape": [ROWS, K]},
                 "input_packet": [Q8_WORDS], "output": [ROWS], "dtype": "float32"},
    "output": {"shape": [1, 1, ROWS], "logits": ROWS},
    "census": {"producer_programs": 1, "consumer_programs": 1, "physical_main_invocations": 1,
               "one_owner": True, "llama_programs": 0, "evidence_cubins": 0},
    "violations": [],
    "integration": "not_model_integrated",
  }

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path)
  ap.add_argument("--fixture", action="store_true")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  a = ap.parse_args()
  report = contract()
  if a.fixture:
    from tinygrad import Tensor, dtypes
    from tinygrad.llm.generate import load_model_and_tokenizer
    from tinygrad.llm.decode_routes import generated_q6k_vocab_logits
    from tinygrad.helpers import getenv
    try:
      model, _ = load_model_and_tokenizer(a.model, 512, seed=20260829)
    except Exception as e:
      report["fixture"] = {"status": "blocked", "trace": f"{type(e).__name__}: {e}",
                            "gpu_serialization": "flock -w 600 /tmp/gpu-bench.lock", "model": a.model}
      payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
      if a.out: a.out.write_text(payload)
      else: print(payload, end="")
      return 0
    linear = model.output
    if getattr(linear, "q6k_storage", None) is None:
      raise RuntimeError("fixture trace: model.output has no canonical q6k_storage")
    if getattr(linear, "route_role", "") != "lm_head": linear.route_role = "lm_head"
    x = (Tensor.arange(K, dtype=dtypes.float32).to("NV").reshape(1, 1, K) / K).contiguous()
    from tinygrad.llm.q6k_vocab_manyrow import Q6KVocabManyRowAdmission, q6k_vocab_manyrow_call
    admission = Q6KVocabManyRowAdmission()
    control = generated_q6k_vocab_logits(linear, x)
    candidate = q6k_vocab_manyrow_call(admission, linear, x)
    if control is None or candidate is None: raise RuntimeError("fixture trace: generated or candidate call returned None")
    t0=time.perf_counter(); cn=control.realize(); c_ms=(time.perf_counter()-t0)*1000
    t0=time.perf_counter(); nn=candidate.realize(); n_ms=(time.perf_counter()-t0)*1000
    c=cn.numpy(); n=nn.numpy(); delta=abs(c-n)
    report.update({"fixture": {"model": a.model, "input": {"shape":[1,1,K],"seed":"arange(K)/K"},
      "correctness": {"shape_equal": list(c.shape)==[1,1,ROWS], "finite_control": bool(__import__('numpy').isfinite(c).all()),
                       "finite_candidate": bool(__import__('numpy').isfinite(n).all()), "max_abs_error": float(delta.max()),
                       "argmax_control": int(c.reshape(-1).argmax()), "argmax_candidate": int(n.reshape(-1).argmax()),
                       "allclose": bool(__import__('numpy').allclose(c,n,rtol=1e-3,atol=1e-2))},
      "r9": {"control_ms": c_ms, "candidate_ms": n_ms, "samples": 1},
      "census": {"control_programs": 1, "candidate_programs": 2, "candidate_one_owner": True},
      "provenance": {"control":"tinygrad_generated", "candidate":"tinygrad_generated", "llama_programs":0, "evidence_cubins":0}}})
  payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if a.out: a.out.write_text(payload)
  else: print(payload, end="")
  return 0

if __name__ == "__main__": raise SystemExit(main())
