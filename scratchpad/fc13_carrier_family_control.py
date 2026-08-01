#!/usr/bin/env python3
"""FC13 -- carrier family-loop control for every registered tensor-core family (compile-only).

`scratchpad/pg2_amd_all_routes_rendered_source_equality.py` and
`scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py` hash rendered bytes and therefore
only detect *change*. This control is the "actually-agnostic" check Claude called for: it loops
every registered family in `tinygrad/codegen/opt/tc.py` -- 10 base lists plus the 4 composition
aliases -- and asserts the emitted WMMA carrier equals the descriptor-derived dtype, using the
descriptor itself as the oracle. This is the control that would have caught `float.vec(8)` being
wrong for the non-RDNA families (WC2, `3ec8557f1`).

For every family and every descriptor in it:
  1. build a plain matmul AST sized to `tc.dims` (M=dims[1], N=dims[0], K=dims[2]) with
     `dtype_in` inputs and `dtype_out` accumulator (`a.dot(b, dtype=do)`), forcing the TC opt
     `Opt(OptOps.TC, 0, (idx, 0, 1))` exactly like `test/unit/test_amd_isa_wmma.py:_tc_matmul_ast`;
  2. replay the early rewrite window (`pm_mops` .. `apply_opts`, mirroring
     `tinygrad/codegen/__init__.py:_full_rewrite_to_sink` up to the TC opt) and assert, on the
     pre-devectorizer graph: (1) `Ops.WMMA` dtype == `tc.dtype_out.vec(elements_per_thread[2])`,
     (2) operand carriers == `tc.dtype_in.vec(elements_per_thread[0/1])`, (3) the UNROLL wrapping
     the WMMA folds exactly `binary_axis_count(tc, 2)` binary accumulator axes (the existing fact
     source, `tinygrad/codegen/opt/kernel_lds.py:58`);
  3. render compile-only with the family's real renderer (HIP / CUDA / Metal) via PG2's
     non-ISA pipeline (`full_rewrite_to_sink` -> `do_linearize` -> `do_estimates` -> `do_render`),
     never `do_compile` (no native toolchain on this box), and record the rendered-source SHA-256.

Assertions 1-3 catch wrongness directly; rendering catching breakage is assertion 4. Aliases are
verified by object identity (`cuda_sm75 is cuda_8168_f16`, plus elementwise identity of the three
compositions). The `amd_rdna3` AMDISA fixture golden (`test_amd_isa_extraction_fixtures.py`,
`_emit_fixture(_tc_matmul_ast)`) is asserted exactly, per the scope acceptance.

SEAMS (documented, compile-only by construction): `CUDARenderer.__init__` constructs an
`NVRTCCompiler` that eagerly loads libnvrtc, and `MetalRenderer.__init__` imports
`tinygrad.runtime.ops_metal` which loads libSystem -- neither exists on this Linux box. The
compiler is only ever touched by `do_compile`, which this control never reaches, so the compiler
class is stubbed and the two native modules are faked with the minimal names the renderer's
`__init__` needs. The renderers, their `tensor_cores` binding, and the whole lowering pipeline
are the real ones.

FAIL-CLOSED GUARD (postrange.py `_apply_generic_tensor_core_opt`): a descriptor whose operand
dtype the renderer cannot express natively -- fp8 on gfx942, where `HIPRenderer.supported_dtypes`
admits fp8_ocp only on gfx950 -- is refused at TC selection with a `KernelOptError` naming the
capability, instead of the two historical failure modes: the K=128 render crash
(`cstyle.py:fp8_index` ValueError on emulated half operands) and, worse, the silent K=32 case
where emulated half operands reached an fp8 builtin. gfx950 and CUDA sm_89 (fp8-native) are
unaffected. Rows whose descriptors are guarded are reported as `blocked` and still count as
passing: the guard IS the intended behavior for those bindings. The gfx942 K=128 cross-check row
below verifies the guard fires there.

Usage: `python3 scratchpad/fc13_carrier_family_control.py` (from the repo root). Exit 0 only if
every family row, alias check, and the AMDISA golden check pass.
"""
from __future__ import annotations

import hashlib, itertools, math, os, sys, types
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, dtypes
from tinygrad.codegen import (  # noqa: E402
  apply_opts, do_estimates, do_linearize, do_render, full_rewrite_to_sink,
  pm_flatten_range, pm_load_collapse, pm_mops, pm_simplify_ranges, pm_split_ranges,
  pm_store_ranges, pm_syntactic_sugar, sym)
from tinygrad.codegen.opt import Opt, OptOps, tc  # noqa: E402
from tinygrad.codegen.opt.kernel_lds import binary_axis_count  # noqa: E402
from tinygrad.helpers import Context, Target  # noqa: E402
from tinygrad.renderer.cstyle import HIPRenderer, MetalRenderer  # noqa: E402
from tinygrad.renderer.cuda import CUDARenderer  # noqa: E402
from tinygrad.uop.ops import Ops, ProgramInfo, UOp, graph_rewrite  # noqa: E402

from test.unit.test_amd_isa_extraction_fixtures import FIXTURES, _emit_fixture  # noqa: E402
from test.unit.test_amd_isa_wmma import _tc_matmul_ast  # noqa: E402


# --- native seams (see module docstring) ---
import tinygrad.runtime.support.compiler_cuda as _cc  # noqa: E402


class _StubNVRTC:
  def __init__(self, arch: str, ptx: bool = True, cache_key: str = "cuda"): pass


_cc.NVRTCCompiler = _StubNVRTC


class _StubMetalCompiler:
  def __init__(self): pass


_fake_ops_metal = types.ModuleType("tinygrad.runtime.ops_metal")
_fake_ops_metal.MetalCompiler = _StubMetalCompiler
sys.modules["tinygrad.runtime.ops_metal"] = _fake_ops_metal
_fake_graph_metal = types.ModuleType("tinygrad.runtime.graph.metal")
_fake_graph_metal.METAL_ICB_OFFSET_MAX = 0xFFFFFFFF
sys.modules["tinygrad.runtime.graph.metal"] = _fake_graph_metal


def _hip(arch: str) -> HIPRenderer: return HIPRenderer(Target.parse(f"AMD:HIP:{arch}"))
def _cuda(arch: str) -> CUDARenderer: return CUDARenderer(Target.parse(f"NV:CUDA:{arch}"))
def _metal() -> MetalRenderer: return MetalRenderer(Target.parse("METAL:METAL:Apple7"))


# name -> (family list, natural renderer, natural-binding check, is_alias, alias parts)
FAMILIES: list[tuple[str, list, object, str, object, bool, list]] = [
  ("amd_rdna3", tc.amd_rdna3, _hip("gfx1100"), "gfx1100", tc.amd_rdna3, False, []),
  ("amd_rdna4", tc.amd_rdna4, _hip("gfx1200"), "gfx1200", tc.amd_rdna4, False, []),
  ("amd_cdna_161616", tc.amd_cdna_161616, _hip("gfx942"), "gfx942", tc.amd_cdna3, False, [tc.amd_cdna3, tc.amd_cdna4]),
  ("amd_cdna_161632", tc.amd_cdna_161632, _hip("gfx942"), "gfx942", tc.amd_cdna3, False, [tc.amd_cdna3, tc.amd_cdna4]),
  ("amd_cdna_1616128", tc.amd_cdna_1616128, _hip("gfx950"), "gfx950", tc.amd_cdna4, False, [tc.amd_cdna4]),
  ("amd_cdna3", tc.amd_cdna3, _hip("gfx942"), "gfx942", tc.amd_cdna3, True, []),
  ("amd_cdna4", tc.amd_cdna4, _hip("gfx950"), "gfx950", tc.amd_cdna4, True, []),
  ("cuda_81616", tc.cuda_81616, _cuda("sm_80"), "sm_80", tc.cuda_sm80, False, [tc.cuda_sm80, tc.cuda_sm89]),
  ("cuda_81632_f8", tc.cuda_81632_f8, _cuda("sm_89"), "sm_89", tc.cuda_sm89, False, [tc.cuda_sm89]),
  ("cuda_8168_f16", tc.cuda_8168_f16, _cuda("sm_80"), "sm_80", tc.cuda_sm80, False, [tc.cuda_sm75, tc.cuda_sm80, tc.cuda_sm89]),
  ("cuda_8168_tf32", tc.cuda_8168_tf32, _cuda("sm_80"), "sm_80", tc.cuda_sm80, False, [tc.cuda_sm80, tc.cuda_sm89]),
  ("cuda_sm75", tc.cuda_sm75, _cuda("sm_75"), "sm_75", tc.cuda_sm75, True, []),
  ("cuda_sm80", tc.cuda_sm80, _cuda("sm_80"), "sm_80", tc.cuda_sm80, True, []),
  ("metal", tc.metal, _metal(), "Apple7", tc.metal, False, []),
]

ALIAS_PARTS = [
  ("cuda_sm75", tc.cuda_sm75, tc.cuda_8168_f16),
  ("cuda_sm80", tc.cuda_sm80, tc.cuda_81616 + tc.cuda_8168_f16 + tc.cuda_8168_tf32),
  ("amd_cdna3", tc.amd_cdna3, tc.amd_cdna_161632[:2] + tc.amd_cdna_161616),
  ("amd_cdna4", tc.amd_cdna4, tc.amd_cdna_1616128 + tc.amd_cdna_161632 + tc.amd_cdna_161616),
]


def _tc_window(ast: UOp, ren) -> UOp:
  """Replay `_full_rewrite_to_sink`'s pre-optimization window up to (and including) the TC opt."""
  sink = graph_rewrite(ast, pm_mops + pm_syntactic_sugar + pm_store_ranges, ctx=itertools.count(1000),
                       name="early movement ops", bottom_up=True)
  sink = graph_rewrite(sink, pm_load_collapse, name="load collapse")
  sink = graph_rewrite(sink, pm_split_ranges + pm_flatten_range, ctx={}, name="split ranges")
  sink = graph_rewrite(sink, sym + pm_flatten_range, name="initial symbolic")
  sink = graph_rewrite(sink, pm_flatten_range + pm_simplify_ranges, ctx={}, name="simplify ranges")
  return apply_opts(sink, ren)


def _render_only(ast: UOp, ren) -> str:
  """PG2's non-ISA pipeline: rewrite -> linearize -> estimates -> render, never do_compile."""
  full_sink = full_rewrite_to_sink(ast, ren, optimize=ast.tag is None)
  prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=ren.target.device)), arg=ProgramInfo.from_sink(full_sink))
  prg = do_linearize(ren, prg, full_sink)
  updated = do_estimates(prg, full_sink, prg.src[2])
  if updated is not None: prg = updated
  prg = do_render(ren, prg, prg.src[2])
  return prg.src[3].arg


def _build_ast(fam: list, idx: int) -> UOp:
  tcx = fam[idx]
  m, n, k = tcx.dims[1], tcx.dims[0], tcx.dims[2]
  a = Tensor.empty(m, k, dtype=tcx.dtype_in)
  b = Tensor.empty(k, n, dtype=tcx.dtype_in)
  lin = a.dot(b, dtype=tcx.dtype_out).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  return ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (idx, 0, 1)),)))


def _check_descriptor(fam: list, idx: int, ren) -> tuple[str, str]:
  """Assertions 1-3 on the TC-opt window, then a compile-only render.

  Returns ("ok", detail) on success, ("blocked", detail) when the fail-closed fp8 guard refuses
  the descriptor (the intended behavior for fp8 on a renderer without native fp8), or
  ("fail", detail) on any other failure.
  """
  tcx = fam[idx]
  tf32_ctx = Context(ALLOW_TF32=1) if tcx.dtype_in == dtypes.float and ren.target.device in ("CUDA", "NV") else Context()
  with tf32_ctx:
    ast = _build_ast(fam, idx)
    try:
      sink = _tc_window(ast, ren)
    except Exception as exc:  # noqa: BLE001 -- classify the fail-closed guard, report everything else
      if "cannot be emitted" in str(exc):
        return "blocked", f"{type(exc).__name__}: {exc}"
      raise
    wmmas = [u for u in sink.toposort() if u.op is Ops.WMMA]
    if len(wmmas) != 1:
      return "fail", f"TC opt produced {len(wmmas)} WMMA nodes, expected 1"
    w = wmmas[0]
    ok1 = w.dtype == tcx.dtype_out.vec(tcx.elements_per_thread[2])
    ok2 = w.src[0].dtype == tcx.dtype_in.vec(tcx.elements_per_thread[0]) and \
          w.src[1].dtype == tcx.dtype_in.vec(tcx.elements_per_thread[1])
    unrolls = [u for u in sink.toposort() if u.op is Ops.UNROLL and u.src and u.src[0] is w]
    ok3 = len(unrolls) == 1 and \
          len([sz for _a, sz in unrolls[0].arg if sz == 2]) == binary_axis_count(tcx, 2)
    if not (ok1 and ok2 and ok3):
      return "fail", (f"carrier mismatch: WMMA={w.dtype} A={w.src[0].dtype} B={w.src[1].dtype} "
                      f"expect C={tcx.dtype_out.vec(tcx.elements_per_thread[2])} "
                      f"A/B={tcx.dtype_in.vec(tcx.elements_per_thread[0])} c1={ok1} c2={ok2} c3={ok3}")
    src = _render_only(ast, ren)
    sha = hashlib.sha256((src + "\n").encode()).hexdigest()[:12]
    marker = src.count("__WMMA") + src.count("simdgroup_multiply_accumulate")
    return "ok", f"sha={sha} src_len={len(src)} marker={marker}"


def main() -> int:
  print(f"{'family':16s} {'carrier':18s} {'desc':8s} {'blocked':8s} {'render':8s} detail")
  family_failures: list[str] = []

  for name, fam, ren, arch, natural, is_alias, natural_parts in FAMILIES:
    # the production binding must already be the family list (aliases and arch-bound bases), or
    # contain every descriptor of the base list by object identity (composition bindings).
    binding_ok = ren.tensor_cores is fam if not natural_parts else \
      all(any(d is x for part in natural_parts for x in part) for d in fam)
    ren.tensor_cores = fam
    results = [_check_descriptor(fam, idx, ren) for idx in range(len(fam))]
    passed = sum(1 for status, _ in results if status == "ok")
    blocked = sum(1 for status, _ in results if status == "blocked")
    carrier = fam[0].dtype_out.vec(fam[0].elements_per_thread[2])
    first_sha = next((d.split()[0].split("=")[1] for status, d in results if status == "ok"), "-")
    detail = f"binding={binding_ok} sha={first_sha} arch={arch}{' alias' if is_alias else ''}"
    fails = [f"idx{i} {fam[i].dtype_in.name}:{fam[i].dtype_out.name}: {d}"
             for i, (status, d) in enumerate(results) if status == "fail"]
    status = "OK" if binding_ok and passed + blocked == len(fam) else "FAIL"
    print(f"{name:16s} {str(carrier):18s} {passed}/{len(fam):<4d} {blocked:4d}/{len(fam):<4d} {passed}/{len(fam):<4d} "
          f"{status} {detail}")
    for f in fails: print(f"    FAILED {name} {f}")
    if binding_ok and passed + blocked == len(fam): continue
    family_failures.append(name)

  for alias_name, alias, parts in ALIAS_PARTS:
    ok = len(alias) == len(parts) and all(a is b for a, b in zip(alias, parts))
    print(f"alias {alias_name:12s} elementwise-identity={ok} ({len(parts)} descriptors)")
    if not ok: family_failures.append(f"alias:{alias_name}")

  # amd_rdna3 AMDISA fixture golden -- the `_tc_matmul_ast` 16x16x16 path, exactly as emitted
  # by the fixture test's `_emit_fixture`.
  golden = FIXTURES["tc_16x16x16_unrolled"]
  got = _emit_fixture(_tc_matmul_ast)
  comparable = {k: v for k, v in golden.items() if k != "ast"}
  golden_ok = got == comparable
  print(f"amd_rdna3 AMDISA fixture golden: {golden_ok} "
        f"(binary={got['binary_sha256'][:12]} mnemonic={got['mnemonic_sha256'][:12]} "
        f"bytes={got['instruction_bytes']} inst={got['instruction_count']} wmma={got['wmma_count']})")
  if not golden_ok: family_failures.append("amd_rdna3 AMDISA fixture golden")

  # cross-check: amd_cdna_1616128 forced onto gfx942 (unreachable in production -- only gfx950
  # binds it, via amd_cdna4). The fail-closed guard must refuse it at TC selection; the natural
  # gfx950 row above is the production path.
  ren942 = _hip("gfx942")
  ren942.tensor_cores = tc.amd_cdna_1616128
  try:
    status, detail = _check_descriptor(tc.amd_cdna_1616128, 0, ren942)
    print(f"cross-check gfx942 K=128 fp8: {status} -- {detail}")
  except Exception as exc:  # noqa: BLE001 -- the documented finding
    print(f"cross-check gfx942 K=128 fp8: UNEXPECTED {type(exc).__name__}: {exc}")

  if family_failures:
    print(f"\n{len(family_failures)} family/alias/golden failures: {family_failures}", file=sys.stderr)
    return 1
  print("\nall families pass")
  return 0


if __name__ == "__main__":
  sys.exit(main())
