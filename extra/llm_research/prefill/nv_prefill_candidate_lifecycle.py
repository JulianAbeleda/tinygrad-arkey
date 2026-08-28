#!/usr/bin/env python3
"""Fresh-process gates for the promoted NV sm_120 fp16 prefill candidates."""
from __future__ import annotations

import argparse, hashlib, json, pathlib
import numpy as np


ROLES = {
  "attn_qo": "attn_q",
  "attn_kv": "attn_k",
  "ffn_gate_up": "ffn_gate",
  "ffn_down": "ffn_down",
}


def _stats(value:np.ndarray) -> dict:
  finite = np.isfinite(value)
  return {
    "shape":list(value.shape), "dtype":str(value.dtype), "finite":bool(finite.all()),
    "nan":int(np.isnan(value).sum()), "inf":int(np.isinf(value).sum()),
    "min":float(value[finite].min()) if finite.any() else None,
    "max":float(value[finite].max()) if finite.any() else None,
    "sha256":hashlib.sha256(value.tobytes()).hexdigest(),
  }


def projection(args) -> dict:
  from tinygrad import Tensor
  from tinygrad.llm.generate import load_model_and_tokenizer
  from tinygrad.llm.prefill_graph_gemm import (_attached_candidate_admission, _candidate_warmstart_opts,
                                                _contiguous_candidate_operand)
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
  from tinygrad.llm.prefill_candidate_runtime import KernelCandidateContext

  model,_ = load_model_and_tokenizer(args.model, 512, seed=20260617)
  block = model.blk[0]
  linear = getattr(block, ROLES[args.role])
  tokens = Tensor([[(i*7)%1000 for i in range(512)]], dtype="int32").contiguous()
  base = block.attn_norm(model.token_embd(tokens).float()).realize()
  if args.role == "ffn_down":
    # Non-zero deterministic carrier: an all-zero down-projection input cannot
    # distinguish a correct kernel from one that drops every contribution.
    vals=((np.arange(512*12288, dtype=np.uint32)%257).astype(np.float32)-128.0).astype(np.float16)/256
    base=Tensor(vals.reshape(1,512,12288)).contiguous().realize()
  admission = _attached_candidate_admission(linear, args.role,
    (512, int(linear.out_features), int(linear.in_features)))
  if admission is None: raise RuntimeError(f"no exact admission for {args.role}")
  key = pr.warmstart_key({512, int(linear.out_features)}, int(linear.in_features))
  target = admission.normalized_payload["workload"]["target"]
  opts = _candidate_warmstart_opts(target["backend"], target["arch"], target["wave_size"])
  selected_opts = opts[:1] if args.arm == "candidate_tconly" else opts
  state_opts = None if args.arm == "generic" else {key:selected_opts}
  context = admission.context
  if args.arm == "candidate_single":
    identity = hashlib.sha256((admission.canonical_identity+":single-buffer-diagnostic").encode()).hexdigest()
    context = KernelCandidateContext(context.schema_version, identity, context.geometry,
      KernelStage1PipelinePlan(1, context.geometry.lds_bytes, 1))
  if args.arm == "candidate_nopipe":
    identity = hashlib.sha256((admission.canonical_identity+":no-pipeline-diagnostic").encode()).hexdigest()
    context = KernelCandidateContext(context.schema_version, identity, context.geometry, None)
  state_contexts = {key:context} if args.arm in {"candidate", "candidate_single", "candidate_nopipe", "candidate_tconly"} else None
  with pr.warmstart_candidate_state(state_opts, state_contexts):
    if args.fixture == "ones":
      a = Tensor.ones(512, int(linear.in_features), dtype="float16").contiguous()
      bt = Tensor.ones(int(linear.out_features), int(linear.in_features), dtype="float16").contiguous()
    elif args.fixture == "kmap":
      av=np.zeros((512, int(linear.in_features)), dtype=np.float16); av[:,::32]=1
      bv=np.zeros((int(linear.out_features), int(linear.in_features)), dtype=np.float16)
      for row in range(int(linear.out_features)): bv[row,row%32::32]=1
      a,bt=Tensor(av).contiguous(),Tensor(bv).contiguous()
    elif args.fixture == "mnmap":
      av=np.zeros((512, int(linear.in_features)), dtype=np.float16)
      for row in range(512): av[row,row%32::32]=1
      bv=np.zeros((int(linear.out_features), int(linear.in_features)), dtype=np.float16)
      for row in range(int(linear.out_features)): bv[row,row%32::32]=1
      a,bt=Tensor(av).contiguous(),Tensor(bv).contiguous()
    elif args.fixture == "tilemap":
      av=np.zeros((512, int(linear.in_features)), dtype=np.float16)
      for row in range(512): av[row,(row%128)*32]=1
      bv=np.zeros((int(linear.out_features), int(linear.in_features)), dtype=np.float16)
      for row in range(int(linear.out_features)): bv[row,(row%128)*32]=1
      a,bt=Tensor(av).contiguous(),Tensor(bv).contiguous()
    else:
      a = base.reshape(512, int(linear.in_features)).cast("float16").contiguous()
      bt = _contiguous_candidate_operand(linear._pf16_w.cast("float16"))
    output = (a @ bt.transpose()).reshape(1, 512, int(linear.out_features)).realize().numpy()
  if args.npy: np.save(args.npy, output)
  return {
    "schema":"tinygrad.nv_prefill_candidate_projection.v1", "arm":args.arm, "role":args.role,
    "candidate_identity":context.canonical_identity, "warmstart_key":repr(key),
    "opts":[repr(x) for x in selected_opts], "context_installed":args.arm in {"candidate", "candidate_single", "candidate_nopipe", "candidate_tconly"}, "output":_stats(output),
  }


def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("command", choices=("projection",))
  parser.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  parser.add_argument("--role", choices=tuple(ROLES), default="attn_qo")
  parser.add_argument("--arm", choices=("generic", "opts", "candidate", "candidate_single", "candidate_nopipe", "candidate_tconly"), default="generic")
  parser.add_argument("--out", required=True)
  parser.add_argument("--npy")
  parser.add_argument("--fixture", choices=("model", "ones", "kmap", "mnmap", "tilemap"), default="model")
  args=parser.parse_args()
  result=projection(args)
  path=pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2)+"\n")
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
