#!/usr/bin/env python3
"""M1 feasibility probe (compile-only). Reuses the exact production machinery that
`tinygrad/llm/packed_wmma_prefill.py::warmstart_entry` / `route_packed_wmma_prefill` drive --
`packed_half_carrier`, `PackedWeightTransform`, `Opt(OptOps.TC, 0, (-1, 2, 1))`, and the
`warmstart_candidate_state` context manager that installs `_WARMSTART_OPTS`/`_WARMSTART_CANDIDATE_CONTEXTS`
in `tinygrad/codegen/opt/postrange.py` -- WITHOUT adding a row to `PACKED_WMMA_ROUTES` and without
running any GPU workload. A `PackedWmmaRoute` is constructed locally, in-process, and never touches
the frozen production table.

Target shape: Qwen3-8B gate/up GEMM, m=512, k=4096, n=12288, Q4_K (per
docs/task_workflow/input/metal-prefill-M-theories-scope-20260730.md, Theory M1).

Two configurations are tried, per the task's need to separate the TC-selection mechanism (generic,
device-keyed via `self.ren.tensor_cores`) from the packed-weight precontract mechanism (the actual
M1 lever, `PrecontractCandidateContract` in `tinygrad/codegen/opt/kernel_lds.py`):

  A) "packed"      -- candidate_context IS attached (the real packed-WMMA path: in-register/LDS
                       dequant fused into the WMMA operand via PrecontractCandidateContract).
  B) "carrier_only" -- candidate_context is NOT attached; only the bare
                       `Opt(OptOps.TC, 0, (-1, 2, 1))` is forced onto the movement-only
                       `packed_half_carrier` view chain (bitcast/reshape/pad/expand/reshape/bitcast),
                       exercising the SAME generic TC-selection code the clean fp16 GEMM already
                       reaches on Metal (schedule-search scope 2.2's 17 __WMMA reference).

No production behaviour is changed: PACKED_WMMA_ROUTES, PACKED_WMMA_ROUTE_BY_KEY, and
PACKED_WMMA_GEOM are read but never written; `set_packed_wmma_canary_verifier` is never called;
`warmstart_candidate_state` is entered as a `with`-scoped context and unwound automatically on the
way out (mirrors the module's own contextmanager contract), so global compiler state reverts.
"""
from __future__ import annotations
import sys, traceback
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.uop.ops import Ops
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import warmstart_key, warmstart_candidate_state
from tinygrad.llm.packed_wmma_prefill import (
  PackedWmmaRoute, PACKED_WMMA_ROUTES, PACKED_WMMA_ROUTE_BY_KEY, packed_half_carrier, _candidate_context,
)

M, K, N, QUANT = 512, 4096, 12288, "Q4_K"

# The AMD-tuned geometry closest in role to the shape under test (PACKED_WMMA_ROUTES' one and only
# "ffn_gate_up" row). Reused VERBATIM -- not re-tuned for Metal, not re-tuned for the 8B shape.
# This is the geometry tuple `_candidate_context` will turn into an LDS/threads layout below; question
# 3 (lds_bytes vs. Metal's 32768-byte threadgroup limit) is answered against exactly this tuple.
_AMD_FFN_GATE_UP_GEOMETRY = PACKED_WMMA_ROUTE_BY_KEY[("Q4_K", "ffn_gate_up", (512, 17408, 5120))].geometry
assert _AMD_FFN_GATE_UP_GEOMETRY == (256, 64, 32, 8, 1, 1), _AMD_FFN_GATE_UP_GEOMETRY

# A LOCAL route row for the shape under test. This is never inserted into PACKED_WMMA_ROUTES /
# PACKED_WMMA_ROUTE_BY_KEY -- it exists only in this probe's local variable.
_LOCAL_ROUTE = PackedWmmaRoute(QUANT, "ffn_gate_up", (M, N, K), _AMD_FFN_GATE_UP_GEOMETRY,
                                canonical_identity="m1-probe-local-only-not-a-production-row")

assert ("Q4_K", "ffn_gate_up", (M, N, K)) not in PACKED_WMMA_ROUTE_BY_KEY, \
  "refusing to run: this shape is unexpectedly already a production row"
assert all(r.canonical_identity != _LOCAL_ROUTE.canonical_identity for r in PACKED_WMMA_ROUTES), \
  "local probe route identity leaked into the production table"


def build_ast():
  """Reproduce PackedWmmaPrefillCandidate.run's graph shape exactly, with uninitialized backing
  tensors (shapes/dtypes alone determine generated code; no checkpoint, no realize, no dispatch)."""
  context, transform = _candidate_context(_LOCAL_ROUTE)
  raw_words = Tensor.empty(transform.packed_bytes // transform.storage_width, dtype=transform.storage_dtype, device="METAL")
  b = packed_half_carrier(raw_words, transform, N, K)              # (N, K) fp16, movement-only view chain
  x_batch = Tensor.empty(M, K, dtype=dtypes.float16, device="METAL")
  out = (x_batch @ b.transpose()).contiguous().reshape(1, M, N)
  calls = [c for c in out.schedule_linear().src if c.op is Ops.CALL]
  if len(calls) != 1:
    raise RuntimeError(f"expected exactly one CALL in the schedule, got {len(calls)}")
  return calls[0].src[0], context, transform


def try_lower(label: str, attach_candidate_context: bool):
  ast, context, transform = build_ast()
  key = warmstart_key({M, N}, K, transform.storage_dtype)
  opts = {key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}
  candidate_contexts = {key: context} if attach_candidate_context else None
  ren = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  print(f"\n=== configuration: {label} (candidate_context attached: {attach_candidate_context}) ===")
  print(f"geometry tuple (tm,tn,tk,wm,wn,bc) = {_LOCAL_ROUTE.geometry}  <- verbatim from "
        f"PACKED_WMMA_ROUTES['Q4_K','ffn_gate_up',(512,17408,5120)] (AMD gfx1100, 14B ffn_gate_up row)")
  print(f"context.geometry.lds_bytes = {context.geometry.lds_bytes}  (Metal threadgroup limit = 32768)")
  try:
    with warmstart_candidate_state(opts, candidate_contexts):
      prog = to_program(ast, ren)
    src = next(u.arg for u in prog.src if u.op is Ops.SOURCE)
    wmma_calls = src.count("__WMMA")
    sg_mma = src.count("simdgroup_multiply_accumulate")
    print(f"LOWERED OK. rendered source length={len(src)} chars")
    print(f"__WMMA call count: {wmma_calls}")
    print(f"simdgroup_multiply_accumulate count: {sg_mma}")
    return {"label": label, "ok": True, "wmma": wmma_calls, "sg_mma": sg_mma,
            "lds_bytes": context.geometry.lds_bytes, "src_len": len(src)}
  except Exception as e:
    tb = traceback.format_exc()
    print(f"RAISED: {type(e).__name__}: {e}")
    print("--- full traceback ---")
    print(tb)
    return {"label": label, "ok": False, "exc_type": type(e).__name__, "exc_msg": str(e),
            "lds_bytes": context.geometry.lds_bytes}


if __name__ == "__main__":
  results = []
  results.append(try_lower("packed (true packed-WMMA precontract path, candidate_context attached)", True))
  results.append(try_lower("carrier_only (bare TC opt on the packed_half_carrier view chain, no candidate_context)", False))
  print("\n=== summary ===")
  for r in results:
    print(r)
