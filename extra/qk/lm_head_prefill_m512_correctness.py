#!/usr/bin/env python3
"""UNRUN / GPU-GATED. Do not execute in this pass -- no GPU kernel execution was performed while writing this
script (per the LM-head prefill-route wiring task: host-side only, GPU correctness deferred to a later pass).

Standalone M=512 LM-head correctness check for the prefill-route wiring landed in:
  - tinygrad/llm/prefill_routes.py (_direct_packed_role / _direct_packed_module_role now resolve "output"/
    "output.weight" to role "lm_head")
  - tinygrad/llm/model.py (Transformer.logits routes self.output through _pf16 when
    self.blk[0]._prefill_v2 is True and self.output is an installed Q4_K/Q6_K direct-packed primitive)
  - tinygrad/llm/model.py (Transformer._prefill_v2_covered now also yields self.output, tagged
    _prefill_graph_role="lm_head", so the VRAM budget preflight in realize_prefill_v2_weights accounts for it)

This proves NUMERIC CORRECTNESS of the generated `q6k_gen_prefill_direct_out_151936_4096_512` kernel that the
above wiring now actually reaches (previously dead code -- nothing ever called route_direct_packed_prefill for
the LM head; see docs referenced in the task). It does NOT prove performance, resource capture, or run the
whole-model A/B (see extra/qk/prefill_whole_synced.py invocation notes at the bottom of this file for that).

Two comparators over the SAME Q6_K-packed weight bytes (self-consistent synthetic weights, same construction
style as extra/qk/prefill_mmq_parity_gate.py's Q4_K parity helper -- the reference dequants the identical raw
bytes the packed kernel reads, so byte-level GGML validity doesn't matter, only self-consistency):
  1. out_packed: extra.qk.q6k_prefill_route_spec.describe_q6k_packed_prefill / emit_q6k_packed_prefill_kernel,
     output_layout="direct_out", role="lm_head" -- the exact call shape route_direct_packed_prefill now uses
     for self.output once this wiring is live (tinygrad/llm/prefill_routes.py:Q6KDirectPackedPrefillCandidate.run).
  2. out_ref: extra.qk.layout.q6_k_reference dequant of the identical raw bytes, matmul'd against the same
     fp16 activation in fp32.

Run (GPU/AMD gfx1100 required -- NOT run by this task):
  cd /home/ubuntu/tinygrad-arkey && PYTHONPATH=. DEV=AMD python3 extra/qk/lm_head_prefill_m512_correctness.py
  # smaller smoke shape (still direct_out, still lm_head role):
  PYTHONPATH=. DEV=AMD python3 extra/qk/lm_head_prefill_m512_correctness.py --rows 4096 --k 4096 --m 32
"""
from __future__ import annotations

import argparse

import numpy as np

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, q6_k_reference
from extra.qk.q6k_prefill_route_spec import describe_q6k_packed_prefill, emit_q6k_packed_prefill_kernel

# Real 8B-class LM-head prefill authority shape (matches
# test/unit/test_q6k_prefill_route_spec.py::test_describe_q6k_packed_prefill_lm_head_authority_shape and the
# dead kernel name cited in the task: q6k_gen_prefill_direct_out_151936_4096_512).
DEFAULT_ROWS, DEFAULT_K, DEFAULT_M = 151936, 4096, 512
RTOL, ATOL = 3e-2, 3e-2  # Q6_K is a lossy quant; this mirrors the tolerance style of the Q4_K wmma parity gate


def _make_q6k_words(rows:int, k:int, seed:int) -> tuple[Tensor, Tensor]:
  """Self-consistent random Q6_K-shaped weight: builds raw per-block bytes with a real fp16 `d` scale in the
  header (bytes [208:210), matching the GGML Q6_K block tail), dequants those SAME bytes for the reference, and
  bitcasts them to uint16 "halfs" for the packed kernel -- exactly the storage format
  Q6KPrimitiveLinear.q6k_storage.halfs uses (see tinygrad/llm/qk_primitives.py)."""
  assert k % Q6_K_BLOCK_ELEMS == 0
  nblocks = (rows * k) // Q6_K_BLOCK_ELEMS
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=nblocks * Q6_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q6_K_BLOCK_BYTES)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 0.02).astype(np.float16)
  raw[:, 208:210] = d.view(np.uint8).reshape(nblocks, 2)  # ql[128]+qh[64]+scales[16]=208 bytes, then d (fp16)
  byte_t = Tensor(raw.reshape(-1).copy()).realize()
  ref_w = q6_k_reference(byte_t, rows * k).reshape(rows, k).cast(dtypes.float32).contiguous().realize()
  halfs = byte_t.bitcast(dtypes.uint16).contiguous().realize()
  assert halfs.shape[0] == nblocks * Q6K_HALFWORDS_PER_BLOCK
  return halfs, ref_w


def run(rows:int=DEFAULT_ROWS, k:int=DEFAULT_K, m:int=DEFAULT_M, seed:int=1337) -> None:
  halfs, ref_w = _make_q6k_words(rows, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).cast(dtypes.float16).contiguous().realize()

  # comparator 1: the packed direct_out kernel, role="lm_head" -- the exact spec/emit pair
  # route_direct_packed_prefill -> Q6KDirectPackedPrefillCandidate.run now reaches for self.output once
  # self.blk[0]._prefill_v2 and is_direct_packed_prefill_linear(self.output) are both true (see
  # tinygrad/llm/model.py:_lm_head_wants_pf16 / logits).
  spec = describe_q6k_packed_prefill(rows, k, m, role="lm_head", parts=1, output_layout="direct_out",
                                     opts=("LOCAL:0:64",))
  assert spec.kernel_name == f"q6k_gen_prefill_direct_out_{rows}_{k}_{m}"
  out_packed = Tensor.empty(m, rows, dtype=dtypes.float32, device=x.device).custom_kernel(
    halfs.to(x.device), x.reshape(m * k), fxn=emit_q6k_packed_prefill_kernel(spec))[0]

  # comparator 2: dense dequant-and-matmul reference over the SAME raw Q6_K bytes.
  out_ref = (x.cast(dtypes.float32) @ ref_w.T)

  got, ref = out_packed.numpy(), out_ref.numpy()
  abs_diff = np.abs(got - ref)
  max_abs = float(abs_diff.max())
  denom = np.maximum(np.abs(ref), 1e-6)
  max_rel = float((abs_diff / denom).max())
  ok = np.allclose(got, ref, rtol=RTOL, atol=ATOL)
  print(f"lm_head_prefill_m512_correctness rows={rows} k={k} m={m} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
       f"{'PASS' if ok else 'FAIL'} (rtol={RTOL}, atol={ATOL})")
  if not ok:
    raise SystemExit("lm_head_prefill_m512_correctness FAILED")


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="vocab_size (N)")
  ap.add_argument("--k", type=int, default=DEFAULT_K, help="hidden dim (K)")
  ap.add_argument("--m", type=int, default=DEFAULT_M, help="prefill token batch (M), must be the PREFILL_UBATCH the route requires")
  ap.add_argument("--seed", type=int, default=1337)
  args = ap.parse_args()
  run(args.rows, args.k, args.m, args.seed)

# ---------------------------------------------------------------------------------------------------------------
# Whole-model A/B (NOT run by this task -- GPU-gated, pinned-clock timing). After the correctness run above
# passes, the A/B compares the routed set WITHOUT output.weight (current shipped coverage) vs WITH output.weight
# (this wiring), on the same authority command extra/qk/prefill_whole_synced.py already uses:
#
#   # baseline: lm_head NOT in the routed/direct-packed set (pre-existing coverage; PREFILL_DIRECT_SKIP_TENSORS
#   # excludes output.weight from direct-packed routing so it keeps taking the dense fp16 fallback)
#   cd /home/ubuntu/tinygrad-arkey && PYTHONPATH=. DEV=AMD PREFILL_DIRECT_SKIP_TENSORS=output.weight \
#     python3 extra/qk/prefill_whole_synced.py --mode authority -K 8 --warmups 4 --rounds 3 \
#     --whole-lengths 512 --pin-clock
#
#   # candidate: lm_head included (this task's wiring; output.weight now resolves role "lm_head" and routes
#   # through _pf16 -> route_direct_packed_prefill for the T=512 prefill-v2 batch)
#   cd /home/ubuntu/tinygrad-arkey && PYTHONPATH=. DEV=AMD \
#     python3 extra/qk/prefill_whole_synced.py --mode authority -K 8 --warmups 4 --rounds 3 \
#     --whole-lengths 512 --pin-clock
#
# Compare the two JSON artifacts' pp512 tok/s and the whole-model logits parity (--logits-only / --quality-gate
# per prefill_whole_synced.py's existing flags) before promoting. This pass did not run either command.
# ---------------------------------------------------------------------------------------------------------------
