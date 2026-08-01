#!/usr/bin/env python3
"""P5 census probe: prove the fused prefill attention route fires on NV and report the trace row."""
import os
os.environ.setdefault("DEV", "NV")

from tinygrad.llm.model import Transformer
from tinygrad.llm.runtime_state import SimpleTokenizer
from tinygrad.llm.fused_attention import reset_custom_kernel_attention_trace, custom_kernel_attention_trace_snapshot

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main():
  model, kv = Transformer.from_gguf(MODEL, 2048)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  print("prefill_custom_kernel_attn:", model.config.prefill_custom_kernel_attn)
  reset_custom_kernel_attention_trace()
  prompt = [(i % 1000) + 1 for i in range(512)]
  gen = model.generate(prompt)
  first = [int(next(gen)) for _ in range(16)]
  print("first_tokens:", first)
  print("trace:", custom_kernel_attention_trace_snapshot())

if __name__ == "__main__":
  main()
