#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os

from extra.qk_nll_eval import CALIB_TEXT

def install_q8_ffn_proxy() -> None:
  from tinygrad import Tensor, dtypes
  from tinygrad.llm import model as model_mod
  from extra.qk_layout import q8_1_dequantize, q8_1_quantize

  orig_feed_forward = model_mod.FFNBlock._feed_forward

  def q8_feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, "ffn_gate_exps") or hasattr(self, "ffn_gateup") or getattr(self, "_prefill_v2", False):
      return orig_feed_forward(self, x)
    if len(x.shape) != 3 or x.shape[-1] != self.config.dim:
      return orig_feed_forward(self, x)
    q, scales = q8_1_quantize(x.reshape(-1, self.config.dim).cast(dtypes.float32))
    xq = q8_1_dequantize(q, scales).reshape(*x.shape)
    return self.ffn_down(self.ffn_gate(xq).silu().contiguous() * self.ffn_up(xq))

  model_mod.FFNBlock._feed_forward = q8_feed_forward

def eval_nll(model_path:str, max_context:int, n_tokens:int, seed:int, q8_proxy:bool) -> dict:
  from tinygrad import Tensor, TinyJit, UOp
  from extra.llm_generate import load_model_and_tokenizer

  if q8_proxy: install_q8_ffn_proxy()
  model, tok = load_model_and_tokenizer(model_path, max_context, seed=seed)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  ids = ids[: n_tokens + 1]
  if len(ids) < 8: raise ValueError(f"calibration too short ({len(ids)} tokens)")

  v_sp = UOp.variable("start_pos", 0, max_context - 1)
  step = TinyJit(lambda t, sp: model.logits(t, sp).realize())

  total_nll, counted = 0.0, 0
  for i in range(len(ids) - 1):
    lg = step(Tensor([[ids[i]]], dtype="int32").contiguous(), v_sp.bind(i))
    total_nll += -float(lg[0, 0].log_softmax()[ids[i + 1]].item())
    counted += 1
  return {
    "nll": total_nll / counted,
    "tokens": counted,
    "model": model_path,
    "q8_ffn_quality_proxy": q8_proxy,
  }

def main() -> None:
  ap = argparse.ArgumentParser(description="Q8H Track A quality proxy: q8-dequantize FFN norm output before gate/up")
  ap.add_argument("--model", required=True)
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--tokens", type=int, default=160)
  ap.add_argument("--seed", type=int, default=20260616)
  ap.add_argument("--proxy", action="store_true")
  ap.add_argument("--out")
  args = ap.parse_args()
  result = eval_nll(args.model, args.max_context, args.tokens, args.seed, args.proxy)
  result["env"] = {k: os.environ.get(k, "") for k in ("DEV", "JIT", "Q4K_PRIMITIVE", "Q6K_PRIMITIVE", "QK_PRIMITIVE_STORAGE")}
  text = json.dumps(result, indent=2) + "\n"
  if args.out:
    with open(args.out, "w") as f: f.write(text)
  print(text, end="")

if __name__ == "__main__":
  main()
