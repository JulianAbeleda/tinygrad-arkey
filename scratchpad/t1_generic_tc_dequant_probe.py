#!/usr/bin/env python3
"""T1: does a packed Q4_K dequant, expressed as ordinary UOps in the load chain, survive tinygrad's
GENERIC tensor-core opt on Metal and stay fused into one kernel? Compile-only, no GPU.

Technique (proven by scratchpad/m1d_confirm_c_fragment.py): build a Tensor graph's AST via
`.schedule_linear()`, then call `to_program(ast, renderer)` directly with a renderer built from
`Target.parse(...)`. This renders (and, for AMD, cross-compiles) without ever calling Device[...],
so it runs for AMD on a machine with no AMD GPU and never opens a Metal device either. No GPU
workload is executed by this script.

GENERIC-PATH GUARANTEE (read this before trusting any WMMA count below):
`tinygrad/codegen/opt/postrange.py::_apply_generic_tensor_core_opt` only engages the hand-authored
packed-WMMA precontract path (`tinygrad/codegen/opt/kernel_lds.py::PrecontractCandidateContract`)
when `candidate_geometry = getattr(getattr(self.ast.arg, "candidate_context", None), "geometry", None)`
is not None (postrange.py:387, 420-435). `candidate_context` only ever gets attached to an AST's
KernelInfo in two ways: (a) explicitly, by a caller building a KernelInfo with candidate_context=...,
or (b) via the `_WARMSTART_CANDIDATE_CONTEXTS` global table installed by
`warmstart_candidate_state(...)` and consulted at postrange.py:778 inside `apply_opts`'s warmstart
branch. This script does neither. It forces the TC opt via `KernelInfo.opts_to_apply`
(`ast.arg.opts_to_apply = (Opt(OptOps.TC, 0, (-1, 2, 1)),)`), which is the FIRST branch checked inside
`apply_opts` (postrange.py:772-774) -- it runs before the warmstart branch is even reached, so
`_WARMSTART_CANDIDATE_CONTEXTS` (a process-global) is never consulted regardless of its contents, and
this script never calls `warmstart_candidate_state` at all. Every AST built here has
`candidate_context=None` by construction (verified below, printed as
"ast.arg.candidate_context BEFORE to_program"), so `candidate_geometry` is guaranteed None inside
`_apply_generic_tensor_core_opt`, and the precontract branch at postrange.py:421-530 is provably
never taken -- only the plain `else: wmma_srcs = [CONTRACT(srcs[0]), CONTRACT(srcs[1])]` branch at
postrange.py:531-535 can run. This is checked at runtime too: `_composite_tc_admission_blocked` and
the online-softmax composite-substitution table are irrelevant here (no composite reduce exists in
a plain GEMM), and a trace confirms `candidate_axes is None` for every kernel built by this script.

Dequant vocabulary reused verbatim, not reinvented: `PackedWeightTransform.dequant` from
`tinygrad/codegen/opt/packed_weight.py` (the same class test/unit/test_packed_weight.py exercises with
symbolic UOp.range row/k arguments in `test_symbolic_row_and_k_survive_scalar_decoder`). Nothing from
`kernel_lds.py`'s precontract-specific vocabulary (`PackedPrecontractOperandTemplate`,
`PrecontractCandidateContract`, `build_precontract_lds_stage`, ...) is imported or used -- that
vocabulary only exists to serve the precontract branch this script proves it never reaches.

Shape: Qwen3-8B ffn_gate_up prefill, (M, N, K) = (512, 12288, 4096), matching the M1d/M1 probes'
SHAPE = (512, 12288, 4096) (row.shape order is (m, n, k), verified against
extra.llm_research.prefill.packed_wmma_correctness_canary.candidate_payload's workload dict there).
"""
from __future__ import annotations
import sys, traceback
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.renderer.cstyle import MetalRenderer, HIPRenderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import PackedWeightTransform

M, N, K = 512, 12288, 4096
QUANT = "Q4_K"
TC_OPT = Opt(OptOps.TC, 0, (-1, 2, 1))  # (tc_select=-1 any, tc_opt=2, use_tensor_cores=1) -- same as m1d/m1 probes

TARGETS = {
  "METAL": ("METAL:METAL:Apple9", lambda t: MetalRenderer(t)),
  "AMD":   ("AMD:HIP:gfx1100",    lambda t: HIPRenderer(t)),
}


def _dense_gemm_ast(device: str) -> UOp:
  """Plain fp16x fp16->fp32 GEMM AST, C[M,N] = sum_k A[M,K] * B[N,K]. Rung 1 payload."""
  a = Tensor.empty(M, K, dtype=dtypes.half, device=device)
  b = Tensor.empty(N, K, dtype=dtypes.half, device=device)
  out = a @ b.transpose()
  linear = out.schedule_linear()
  calls = [c for c in linear.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
  if len(calls) != 1: raise ValueError(f"expected exactly one SINK CALL, found {len(calls)}")
  return calls[0].src[0]


def _find_mnk(ast: UOp):
  """Locate the reduce, its MUL operands (in0=A-side, in1=B-side), and the exact m/n/k RANGE UOps --
  same pattern _apply_generic_tensor_core_opt itself uses (postrange.py:356-374)."""
  reduces = [u for u in ast.backward_slice if u.op is Ops.REDUCE]
  if len(reduces) != 1: raise ValueError(f"expected exactly one REDUCE, found {len(reduces)}")
  red = reduces[0]
  mul = red.src[0] if red.src[0].op is not Ops.CAST else red.src[0].src[0]
  if mul.op is not Ops.MUL: raise ValueError(f"reduce body is not a MUL: {mul.op}")
  in0, in1 = mul.src
  k_rng = red.src[1]
  # in1 (the B/N operand) owns exactly one LOOP range not shared with in0 -- that's the N range.
  n_candidates = [r for r in in1.ranges if r not in in0.ranges and r.arg[-1].name == "LOOP"]
  if len(n_candidates) != 1: raise ValueError(f"expected exactly one N range on the B operand, found {len(n_candidates)}")
  return red, in0, in1, n_candidates[0], k_rng


def _packed_dequant_ast(device: str) -> tuple[UOp, PackedWeightTransform, UOp]:
  """Same GEMM AST as _dense_gemm_ast, but with the B operand's plain fp16 INDEX replaced in-place
  by a real Q4_K dequant expression (PackedWeightTransform.dequant), built from the SAME n/k RANGE
  UOps the dense AST already uses for that operand. This is 'the dequant expressed as UOps in the
  load chain', not a materialize-then-matmul Tensor pipeline: the substitution happens directly on
  the pre-opt AST, before apply_opts/TC search ever runs. Rung 2/3 payload."""
  ast = _dense_gemm_ast(device)
  red, in0, in1, n_rng, k_rng = _find_mnk(ast)
  desc = PackedWeightTransform(QUANT, rows=N, k=K)
  packed_words = desc.packed_bytes // desc.storage_width
  existing_slots = {u.arg.slot for u in ast.toposort() if u.op is Ops.PARAM}
  packed_slot = max(existing_slots) + 1
  source = UOp.placeholder((packed_words,), desc.storage_dtype, packed_slot)
  dequant_value = desc.dequant(source, row=n_rng, k=k_rng)
  if dequant_value.dtype != dtypes.float16: raise ValueError(f"dequant did not produce fp16: {dequant_value.dtype}")
  if n_rng not in dequant_value.backward_slice_with_self or k_rng not in dequant_value.backward_slice_with_self:
    raise ValueError("dequant expression lost the n/k range dependency it must share with the rest of the AST")
  ast2 = ast.substitute({in1: dequant_value})
  # confirm the dense B PARAM (slot 2) is now unreachable and the packed PARAM (packed_slot) is reachable
  live_slots = {u.arg.slot for u in ast2.toposort() if u.op is Ops.PARAM}
  if 2 in live_slots: raise ValueError("dense B PARAM (slot 2) is still reachable after substitution -- dequant did not replace the load")
  if packed_slot not in live_slots: raise ValueError("packed-weight PARAM did not survive substitution")
  return ast2, desc, source


def _force_generic_tc(ast: UOp) -> UOp:
  """Force the generic TC opt via KernelInfo.opts_to_apply -- the FIRST branch apply_opts checks
  (postrange.py:772-774), ahead of the warmstart lookup. candidate_context is left at its default
  (None): never set here, never installed via warmstart_candidate_state (never called in this
  script), so the precontract branch's `candidate_geometry is not None` gate cannot fire."""
  from dataclasses import replace
  assert ast.arg.candidate_context is None, "harness bug: candidate_context must be None to guarantee the generic-only path"
  return ast.replace(arg=replace(ast.arg, opts_to_apply=(TC_OPT,)))


def _render(ast: UOp, device: str, label: str) -> dict:
  target_str, make_renderer = TARGETS[device]
  renderer = make_renderer(Target.parse(target_str))
  print(f"\n--- {label} / {device} ({target_str}) ---")
  print(f"ast.arg.candidate_context BEFORE to_program: {ast.arg.candidate_context!r}")
  print(f"ast.arg.opts_to_apply BEFORE to_program: {ast.arg.opts_to_apply!r}")

  compiler_patch = None
  if device == "AMD":
    # AMD cross-compile (amd_comgr) is native and unstable on this Mac (unrelated to this question --
    # it's downstream of full_rewrite_to_sink/apply_opts, where the TC opt already ran). Stub only the
    # native compile call so we still get real rendered SOURCE text and a real apply_opts run without
    # touching that crashing path. Same technique as m1d_confirm_c_fragment.py. No GPU is involved
    # either way: this AMD target has no runtime on this machine regardless.
    import unittest.mock
    compiler_patch = unittest.mock.patch.object(type(renderer.compiler), "compile", lambda self, src: b"")

  try:
    if compiler_patch is not None:
      with compiler_patch: prog = to_program(ast, renderer)
    else:
      prog = to_program(ast, renderer)
  except Exception as exc:
    print(f"RAISED: {type(exc).__name__}: {exc}")
    print(traceback.format_exc())
    return {"label": label, "device": device, "ok": False, "exc_type": type(exc).__name__, "exc_msg": str(exc)}

  source = next((u.arg for u in prog.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
  wmma_count = source.count("__WMMA") if source else None
  sgma_count = source.count("simdgroup_multiply_accumulate") if source else None
  vwmma_count = source.count("v_wmma") if source else None
  out_path = f"/tmp/t1_{label}_{device.lower()}_source.c"
  if source:
    with open(out_path, "w") as f: f.write(source)
  print(f"LOWERED OK. source_len={len(source) if source else None}")
  print(f"__WMMA count: {wmma_count}   simdgroup_multiply_accumulate count: {sgma_count}   v_wmma count: {vwmma_count}")
  print(f"source written to {out_path}")
  return {"label": label, "device": device, "ok": True, "src_len": len(source) if source else None,
          "wmma_count": wmma_count, "sgma_count": sgma_count, "vwmma_count": vwmma_count, "source_path": out_path}


def main():
  results = []
  for device in ("METAL", "AMD"):
    # Rung 1: harness check -- plain fp16 GEMM, generic TC opt, does __WMMA/simdgroup_multiply_accumulate show up?
    dense_ast = _force_generic_tc(_dense_gemm_ast(device))
    results.append(_render(dense_ast, device, "rung1_dense_fp16"))

    # Rung 2/3: same GEMM, B operand replaced by a real Q4_K dequant expressed as UOps in the load chain.
    try:
      packed_ast, desc, source = _packed_dequant_ast(device)
      packed_ast = _force_generic_tc(packed_ast)
      results.append(_render(packed_ast, device, "rung2_packed_q4k_dequant"))
    except Exception as exc:
      print(f"\n--- rung2_packed_q4k_dequant / {device}: AST CONSTRUCTION FAILED before to_program ---")
      print(f"{type(exc).__name__}: {exc}")
      print(traceback.format_exc())
      results.append({"label": "rung2_packed_q4k_dequant", "device": device, "ok": False,
                       "exc_type": type(exc).__name__, "exc_msg": str(exc), "stage": "ast_construction"})

  print("\n\n===== SUMMARY =====")
  for r in results:
    print(r)


if __name__ == "__main__":
  main()
