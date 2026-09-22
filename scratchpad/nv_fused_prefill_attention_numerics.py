#!/usr/bin/env python3
"""P5 (nv-fused-prefill-attention-port-scope-20260801.md): NVCC compile + correctness on three axes.

Drives the REAL production seam (custom_kernel_attention -> FlashPrefillAttentionSpec.emit ->
Tensor.uop_program -> NVCC compile/run) on the RTX 5090 for BOTH admitted grids, and reports
max_abs_error vs the SDPA reference, write coverage of the output, and determinism across >= 3 runs.
"""
import os
os.environ.setdefault("DEV", "NV")

import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.fused_attention import custom_kernel_attention, _attention_spec_target
from tinygrad.uop.ops import SharedAttentionCandidateContext

SENTINEL = -777.0
SCALE = 0.125


def make_qkv(hq: int, hkv: int, q_tokens: int, kv_tokens: int, seed: int):
  rng = np.random.default_rng(seed)
  q = rng.normal(0, .04, (1, hq, q_tokens, 128)).astype(np.float16)
  k = rng.normal(0, .04, (1, hkv, kv_tokens, 128)).astype(np.float16)
  v = rng.normal(0, .04, (1, hkv, kv_tokens, 128)).astype(np.float16)
  return tuple(Tensor(x, device="NV") for x in (q, k, v)), (q, k, v)


def ctx_for(hq: int, hkv: int, q_tokens: int, kv_tokens: int):
  return SharedAttentionCandidateContext("qwen3_8b_q4k_m_gfx1100", "FULL_RESIDENT_OVERLAY",
    q_tokens, kv_tokens, kv_tokens - q_tokens, hq, hkv, 128, True)


def reference(hq: int, hkv: int, q_tokens: int, kv_tokens: int, q: Tensor, k: Tensor, v: Tensor):
  g = hq // hkv
  mask = Tensor.full((1, 1, q_tokens, kv_tokens), float("-inf"), dtype=dtypes.float16, buffer=False).triu(kv_tokens - q_tokens + 1)
  return q.scaled_dot_product_attention(k.repeat_interleave(g, dim=-3), v.repeat_interleave(g, dim=-3), attn_mask=mask)


def run_grid(hq: int, hkv: int, q_tokens: int, kv_tokens: int, seed: int) -> dict:
  (q, k, v), _ = make_qkv(hq, hkv, q_tokens, kv_tokens, seed)
  ctx = ctx_for(hq, hkv, q_tokens, kv_tokens)
  out = custom_kernel_attention(q, k, v, scale=SCALE, causal=True, ctx=ctx)
  got = out.numpy().astype(np.float32)
  ref = reference(hq, hkv, q_tokens, kv_tokens, q, k, v).numpy().astype(np.float32)
  max_abs = float(np.max(np.abs(got - ref)))
  allclose = bool(np.allclose(got, ref, rtol=.03, atol=.006))

  # Write coverage: re-run the same program with a sentinel-filled output buffer; any unwritten
  # element keeps the sentinel.
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
  from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
  spec = FlashPrefillAttentionSpec(Hq=hq, Hkv=hkv, Hd=128, q_tokens=q_tokens, kv_tokens=kv_tokens,
    causal=True, scale=SCALE, target=_attention_spec_target(q.device))
  fxn = spec.emit()
  program = KernelProgram("prefill_flash_attention_generated", "prefill_flash_attention.p5_coverage",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, fxn,
    output_spec=OutputSpec((hq * q_tokens * 128,), dtypes.float16))
  sentinel = Tensor.full((hq * q_tokens * 128,), SENTINEL, dtype=dtypes.float16, device="NV")
  covered = execute_promoted_program(sentinel, q.reshape(hq * q_tokens * 128), k.reshape(hkv * kv_tokens * 128),
    v.reshape(hkv * kv_tokens * 128), program=program).numpy()
  coverage = float((covered != SENTINEL).mean())

  # Determinism: >= 3 fresh executions, exact-equality comparison.
  outs = [custom_kernel_attention(q, k, v, scale=SCALE, causal=True, ctx=ctx).numpy() for _ in range(3)]
  deterministic = bool(all(np.array_equal(outs[0], o) for o in outs[1:]))
  return {"grid": (hq, hkv, q_tokens), "spec_target": _attention_spec_target(q.device),
          "compiled": True, "max_abs_error": max_abs, "allclose": allclose,
          "write_coverage": coverage, "deterministic": deterministic, "runs": len(outs)}


if __name__ == "__main__":
  Device["NV"].synchronize()
  for (hq, hkv), seed in [((32, 8), 20260801), ((40, 8), 20260802)]:
    print(run_grid(hq, hkv, 512, 512, seed))
