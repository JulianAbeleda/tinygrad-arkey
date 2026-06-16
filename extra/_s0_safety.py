#!/usr/bin/env python3
"""S0 safety probe for the path-aware Q4K_PRIMITIVE default. Loads MODEL, runs a few decodes,
reports allocated VRAM (after load + peak) and whether it ran. storage_bytes (sidecar delta) comes
from the QK_PRIMITIVE_STORAGE_DEBUG line when Q4K_PRIMITIVE_DEBUG=1."""
import os
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.model import Transformer

PATH = os.environ["MODEL"]; prim = os.environ.get("Q4K_PRIMITIVE", "0")
try:
  model, kv = Transformer.from_gguf(PATH, 4096)
  after_load = GlobalCounters.mem_used
  toks = []
  for i, t in enumerate(model.generate(list(range(10, 18)), temperature=0.0)):
    toks.append(int(t))
    if i >= 4: break
  peak = GlobalCounters.mem_used
  print(f"RESULT model={os.path.basename(PATH)} prim={prim} ran=YES "
        f"mem_after_load_GB={after_load/1e9:.2f} peak_GB={peak/1e9:.2f} toks={toks}")
except Exception as e:
  print(f"RESULT model={os.path.basename(PATH)} prim={prim} ran=NO error={type(e).__name__}: {str(e)[:120]}")
