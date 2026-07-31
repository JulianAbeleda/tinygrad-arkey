#!/usr/bin/env python3
"""T1b: does the VECTORIZED dequant form (PackedWeightTransform.dequant_tile) reach the generic
tensor-core opt on Metal and AMD without triggering pm_split_ranges' RANGE-%-CONST rewrite?
Compile-only, no GPU. Reuses T1's harness verbatim (imported, not reimplemented): _dense_gemm_ast,
_find_mnk, _force_generic_tc, _render, TARGETS, M/N/K/QUANT/TC_OPT all come from
scratchpad/t1_generic_tc_dequant_probe.py.

BACKGROUND (established by T1): PackedWeightTransform.dequant(source, row, k=k_rng) passes the BARE
reduce-owning RANGE UOp directly into `k % self.block_elems` inside _dequant_q4/_dequant_q6. tinygrad's
codegen/simplify.py::pm_split_ranges matches the literal pattern `UPat(Ops.RANGE, name="r") % UPat.cvar`
-- syntactic, not semantic -- and when it fires it substitutes the split expression for r EVERYWHERE r is
referenced in the sink graph, including inside reduceop.src[1:] (the reduce's own range list), turning a
bare RANGE into an ADD node there. That pass runs unconditionally in full_rewrite_to_sink, BEFORE
apply_opts / the TC-opt search ever gets a look, so _apply_generic_tensor_core_opt's
`red_ranges = sorted(reduceop.src[1:], key=lambda x: x.arg[0], ...)` (postrange.py:375) then crashes on
an ADD node whose .arg is None.

The precontract path's dequant_tile (kernel_lds.py:568,649) never exposes a bare RANGE to `%` because
the k value it hands to dequant()/dequant_tile() is always a *compound* tile-coordinate expression
(`epoch*geometry.tile[2]+logical_k`, an ADD/MUL tree) -- never the naked reduce RANGE.

THIS SCRIPT tests two ways of getting a compound (non-bare) k onto the GENERIC path (no
candidate_context, forced via opts_to_apply, exactly like T1):

  Experiment A ("naive dodge"): call dequant_tile(source, row=n_rng, k_base=k_rng, width=W).value,
  then GEP lane 0. dequant_tile's own internal `k_base+i` (i a *Python int* constant, not a UOp range)
  already wraps k_rng in an ADD before any %-block-elems happens, for every lane including i=0 (ADD
  does not eagerly fold with x+0 in this codebase -- confirmed: UOp.alu just builds UOp(op,...) with no
  identity folding, and .cast() is the only short-circuit found, and it's not exercised here). This is
  mathematically identical to a single dequant(k_rng) call (GEP lane 0 of a k_rng+0..7 window) but
  computes 7 redundant unused lanes per single reduce step -- included as a cheap sanity check of
  whether pm_split_ranges' bug is purely syntactic, not a meaningful fused-tile construction.

  Experiment B ("real tile coordinate", the one the task asks for): manually split the K reduce axis
  itself, BEFORE calling _force_generic_tc, into two genuine UOp RANGEs -- outer_tile (AxisType.REDUCE,
  size K//W) and inner_lane (AxisType.REDUCE, size W) -- with k_expr = outer_tile*W + inner_lane. A's
  operand (in0) has its k_rng reference substituted by k_expr. B's operand becomes
  dequant_tile(source, row=n_rng, k_base=outer_tile*W, width=W).value (a STACK of W independently-built
  scalar dequant expressions, each referencing outer_tile only via multiplication -- never bare), with a
  WHERE-chain selecting the inner_lane'th lane (GEP requires a compile-time-constant index; inner_lane is
  a symbolic RANGE, so lane selection must go through CMPEQ/WHERE, not GEP-by-variable). The reduce is
  rebuilt with src[1:] = (outer_tile, inner_lane) -- two real RANGE nodes, neither ever appearing bare
  under a `%`. W is chosen per backend to exactly match tc.dims[2] (8 for Metal, 16 for AMD) so the
  generic TC opt's own axis pick (TC_OPT axis=0, i.e. axis_choices[0] -- the highest-id red_range first,
  which is inner_lane since it's constructed after outer_tile) lands exactly on inner_lane with no
  padding required.

Every claim below is checked directly against reduceop.src[1:] types (via a standalone
graph_rewrite(ast, pm_split_ranges+pm_flatten_range, ...) call -- the exact rewrite full_rewrite_to_sink
runs at codegen/__init__.py:130, applied here in isolation for direct inspection) and against the
rendered source / to_program result, never inferred.
"""
from __future__ import annotations
import sys, traceback
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp, AxisType, graph_rewrite
from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
from tinygrad.codegen.simplify import pm_split_ranges, pm_flatten_range

sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp/scratchpad")
import t1_generic_tc_dequant_probe as T1

M, N, K, QUANT, TC_OPT, TARGETS = T1.M, T1.N, T1.K, T1.QUANT, T1.TC_OPT, T1.TARGETS

# tc.dims[2] per backend, read directly off the real renderer tensor_cores lists (checked below in main(),
# not hardcoded on faith).
WIDTH_FOR_BACKEND = {"METAL": 8, "AMD": 16}


def _select_lane(vec: UOp, axis: UOp, width: int) -> UOp:
  """Select lane `axis` (a symbolic UOp, not a compile-time int) out of a width-`width` vector via a
  WHERE-chain (CMPEQ + WHERE), since Ops.GEP requires a compile-time-constant index and axis is not one."""
  result = vec.gep(width - 1)
  for i in reversed(range(width - 1)):
    result = (axis.eq(i)).where(vec.gep(i), result)
  return result


def _inspect_split_ranges(ast: UOp, label: str) -> dict:
  """Apply EXACTLY the rewrite full_rewrite_to_sink runs at codegen/__init__.py:130
  (`graph_rewrite(sink, pm_split_ranges+pm_flatten_range, ctx={}, name="split ranges")`), standalone,
  and report reduceop.src[1:] node op types before and after -- the decisive observable."""
  reduces_before = [u for u in ast.backward_slice if u.op is Ops.REDUCE]
  before = [[x.op.name for x in r.src[1:]] for r in reduces_before]
  try:
    after_ast = graph_rewrite(ast, pm_split_ranges + pm_flatten_range, ctx={}, name="t1b split ranges probe")
    reduces_after = [u for u in after_ast.backward_slice if u.op is Ops.REDUCE]
    after = [[x.op.name for x in r.src[1:]] for r in reduces_after]
    ok = True
  except Exception as exc:
    after, ok = f"{type(exc).__name__}: {exc}", False
  print(f"  [{label}] reduceop.src[1:] BEFORE pm_split_ranges: {before}")
  print(f"  [{label}] reduceop.src[1:] AFTER  pm_split_ranges: {after}")
  return {"before": before, "after": after, "ok": ok}


def _experiment_a_naive_dodge(device: str, width: int) -> UOp:
  """dequant_tile(k_base=k_rng), GEP lane 0. Mathematically == dequant(k_rng); no tile restructuring."""
  ast = T1._dense_gemm_ast(device)
  red, in0, in1, n_rng, k_rng = T1._find_mnk(ast)
  desc = PackedWeightTransform(QUANT, rows=N, k=K)
  packed_words = desc.packed_bytes // desc.storage_width
  existing_slots = {u.arg.slot for u in ast.toposort() if u.op is Ops.PARAM}
  packed_slot = max(existing_slots) + 1
  source = UOp.placeholder((packed_words,), desc.storage_dtype, packed_slot)
  tile = desc.dequant_tile(source, row=n_rng, k_base=k_rng, width=width)
  b_value = tile.value.gep(0)
  if b_value.dtype != dtypes.float16: raise ValueError(f"lane-0 value is not fp16: {b_value.dtype}")
  if k_rng not in b_value.backward_slice_with_self: raise ValueError("lane-0 value lost its k_rng dependency")
  ast2 = ast.substitute({in1: b_value})
  live_slots = {u.arg.slot for u in ast2.toposort() if u.op is Ops.PARAM}
  if 2 in live_slots: raise ValueError("dense B PARAM still reachable -- substitution failed")
  if packed_slot not in live_slots: raise ValueError("packed PARAM did not survive substitution")
  return ast2


def _experiment_b_tile_coordinate(device: str, width: int) -> UOp:
  """Genuine two-range K split: outer_tile (REDUCE, size K//W) x inner_lane (REDUCE, size W),
  k_expr = outer_tile*W + inner_lane. B operand = dequant_tile(k_base=outer_tile*W).value, lane-selected
  by inner_lane through a WHERE-chain (not GEP -- inner_lane is symbolic). A operand gets its k_rng
  reference replaced by k_expr so both operands agree on the same logical k."""
  ast = T1._dense_gemm_ast(device)
  red, in0, in1, n_rng, k_rng = T1._find_mnk(ast)
  if K % width: raise ValueError(f"K={K} not divisible by width={width}")

  outer_tile = UOp.range(K // width, 9001, AxisType.REDUCE)
  inner_lane = UOp.range(width, 9002, AxisType.REDUCE)
  k_expr = outer_tile * width + inner_lane

  desc = PackedWeightTransform(QUANT, rows=N, k=K)
  packed_words = desc.packed_bytes // desc.storage_width
  existing_slots = {u.arg.slot for u in ast.toposort() if u.op is Ops.PARAM}
  packed_slot = max(existing_slots) + 1
  source = UOp.placeholder((packed_words,), desc.storage_dtype, packed_slot)

  tile = desc.dequant_tile(source, row=n_rng, k_base=outer_tile * width, width=width)
  if k_rng in tile.value.backward_slice_with_self: raise ValueError("tile value unexpectedly depends on bare k_rng")
  b_value = _select_lane(tile.value, inner_lane, width)
  if b_value.dtype != dtypes.float16: raise ValueError(f"selected lane is not fp16: {b_value.dtype}")
  if outer_tile not in b_value.backward_slice_with_self or inner_lane not in b_value.backward_slice_with_self:
    raise ValueError("selected lane lost its outer_tile/inner_lane dependency")

  in0_new = in0.substitute({k_rng: k_expr})
  if k_rng in in0_new.backward_slice_with_self: raise ValueError("A operand still references bare k_rng after substitution")

  mul = red.src[0] if red.src[0].op is not Ops.CAST else red.src[0].src[0]
  mul_new = mul.replace(src=(in0_new, b_value))
  body_new = red.src[0].substitute({mul: mul_new}) if red.src[0].op is Ops.CAST else mul_new
  red_new = red.replace(src=(body_new, outer_tile, inner_lane))

  ast2 = ast.substitute({red: red_new})
  if k_rng in ast2.backward_slice_with_self: raise ValueError("bare k_rng still reachable after full substitution -- surgery failed")
  live_slots = {u.arg.slot for u in ast2.toposort() if u.op is Ops.PARAM}
  if 2 in live_slots: raise ValueError("dense B PARAM still reachable -- substitution failed")
  if packed_slot not in live_slots: raise ValueError("packed PARAM did not survive substitution")
  reduces = [u for u in ast2.backward_slice if u.op is Ops.REDUCE]
  if len(reduces) != 1: raise ValueError(f"expected exactly one REDUCE post-surgery, found {len(reduces)}")
  if set(reduces[0].src[1:]) != {outer_tile, inner_lane}:
    raise ValueError(f"reduce ranges are not exactly {{outer_tile, inner_lane}}: {reduces[0].src[1:]}")
  return ast2


def main():
  results = []
  for device in ("METAL", "AMD"):
    width = WIDTH_FOR_BACKEND[device]
    target_str, make_renderer = TARGETS[device]
    from tinygrad.helpers import Target
    renderer = make_renderer(Target.parse(target_str))
    tc_dims2 = {tc.dims[2] for tc in renderer.tensor_cores}
    print(f"\n=== {device}: tc.dims[2] observed = {sorted(tc_dims2)} (width chosen = {width}) ===")
    if width not in tc_dims2:
      print(f"  WARNING: chosen width {width} not among observed tc.dims[2] {tc_dims2}")

    # ---- Experiment A: naive dodge ----
    print(f"\n--- experiment A (naive dodge, dequant_tile k_base=k_rng, GEP lane 0) / {device} ---")
    try:
      ast_a = _experiment_a_naive_dodge(device, width)
      split_a = _inspect_split_ranges(ast_a, f"A/{device}")
      ast_a_forced = T1._force_generic_tc(ast_a)
      r = T1._render(ast_a_forced, device, "t1b_expA_naive_dodge")
      r["split_ranges"] = split_a
      results.append(r)
    except Exception as exc:
      print(f"AST CONSTRUCTION / RENDER FAILED: {type(exc).__name__}: {exc}")
      print(traceback.format_exc())
      results.append({"label": "t1b_expA_naive_dodge", "device": device, "ok": False,
                       "exc_type": type(exc).__name__, "exc_msg": str(exc)})

    # ---- Experiment B: real tile coordinate ----
    print(f"\n--- experiment B (tile-coordinate split, outer_tile x inner_lane) / {device} ---")
    try:
      ast_b = _experiment_b_tile_coordinate(device, width)
      split_b = _inspect_split_ranges(ast_b, f"B/{device}")
      ast_b_forced = T1._force_generic_tc(ast_b)
      r = T1._render(ast_b_forced, device, "t1b_expB_tile_coordinate")
      r["split_ranges"] = split_b
      results.append(r)
    except Exception as exc:
      print(f"AST CONSTRUCTION / RENDER FAILED: {type(exc).__name__}: {exc}")
      print(traceback.format_exc())
      results.append({"label": "t1b_expB_tile_coordinate", "device": device, "ok": False,
                       "exc_type": type(exc).__name__, "exc_msg": str(exc)})

  print("\n\n===== SUMMARY =====")
  for r in results:
    print(r)


if __name__ == "__main__":
  main()
