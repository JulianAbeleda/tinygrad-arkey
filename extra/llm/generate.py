#!/usr/bin/env python3
"""Shared LLM model-load core for the QK eval/bench tooling.

The AMD env-ordering invariant is sacred: `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` plus
the Q4K/Q6K primitive flags must be set *before* `from tinygrad import ...`. This
module therefore imports tinygrad lazily, *inside* `load_model_and_tokenizer`, so
importing it never pulls in tinygrad and never freezes the env.
"""
from __future__ import annotations

import pathlib
from typing import Any


def load_model_and_tokenizer(model:Any, max_context:int, *, seed:int, adapter:Any=None) -> tuple[Any, Any]:
  """Seed, load the gguf Transformer (+ optional adapter), return (model, tokenizer).

  tinygrad is imported here, lazily, so the env-ordering invariant holds: callers
  must have configured the environment before calling this.
  """
  from tinygrad import Tensor
  from extra.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(seed)
  model_obj, kv = Transformer.from_gguf(pathlib.Path(model).expanduser(), max_context)
  if adapter is not None:
    from extra.llm.adapter import load_adapter
    load_adapter(model_obj, pathlib.Path(adapter).expanduser())
  return model_obj, SimpleTokenizer.from_gguf_kv(kv)
