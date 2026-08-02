#!/usr/bin/env python3
"""PG3 -- decode render-equality control (HIP arm; the Metal arm is macOS-only).

Pin rendered-source SHA-256s for the production decode emitters exactly as PG2
(scratchpad/pg2_amd_all_routes_rendered_source_equality.py) does for the prefill
routes: replay `to_program`'s non-ISA pipeline (`full_rewrite_to_sink` ->
`do_linearize` -> `do_estimates` -> `do_render`), never `do_compile` -- no ROCm
compiler and no GPU needed. The ASTs are built by calling the real emitters
(`q4k_g3_lanemap_gemv_kernel`, `emit_q6k_gemv_kernel` via `q6k_spec_for_role`,
`emit_q6k_vocab_scalar_reduce_kernel`, `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`
via `FlashDecodeTileSpec.emit`, `flash_fused_gmax_combine_kernel` via `FlashCombineSpec.emit`)
with `UOp.placeholder` buffers, mirroring the shapes `decode_routes.py` /
`flash_decode_attention.py` bind, and verified byte-identical to the Tensor
`uop_program` path. The flash tile is rendered with the production symbolic
`start_pos` variable (Tc = start_pos + 1) at the campaign's measured
max_context=4608 (`/tmp/qwen3-8b-nv-p4-decode.json`), which shapes the cache
buffer strides in the rendered index arithmetic.

Hashes follow the house convention: `sha256((src + "\n").encode())`, first 12 hex
digits, plus src_len, plus per-instruction marker counts (`__WMMA`, `fdot2`,
`simd_shuffle_xor`, `ds_bpermute` occurrences in the rendered source).

Usage: `python3 scratchpad/pg3_decode_rendered_source_equality.py [--renderer hip|metal]`
`hip` (default) instantiates HIPRenderer anywhere. `metal` instantiates
MetalRenderer, which imports the macOS Metal runtime and can therefore only run
on the macOS box -- it must NOT be run on this Linux NV box.
"""
from __future__ import annotations
import hashlib, sys

from tinygrad import dtypes
from tinygrad.codegen import do_estimates, do_linearize, do_render, full_rewrite_to_sink
from tinygrad.helpers import Target, getenv
from tinygrad.llm.decode_kernels import (emit_q6k_gemv_kernel, emit_q6k_vocab_scalar_reduce_kernel,
  q4k_g3_lanemap_gemv_kernel, q6k_spec_for_role)
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import HIPRenderer, MetalRenderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp

MARKERS = ("__WMMA", "fdot2", "simd_shuffle_xor", "ds_bpermute")
# Campaign pin: qwen3-8b-nv-p4-decode.json max_context=4608. It shapes the flash cache
# buffer strides, so the flash hashes are tied to it; the tile is rendered at Tc=start_pos+1.
MAXC = 4608


def render_only(ast: UOp, ren: Renderer) -> str:
  """Replay to_program's non-ISA pipeline up to (and including) do_render, never do_compile."""
  full_sink = full_rewrite_to_sink(ast, ren, optimize=ast.tag is None)
  prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=ren.target.device)), arg=ProgramInfo.from_sink(full_sink))
  prg = do_linearize(ren, prg, full_sink)
  updated = do_estimates(prg, full_sink, prg.src[2])
  if updated is not None: prg = updated
  prg = do_render(ren, prg, prg.src[2])
  return prg.src[3].arg


def _q4k_ast(rows: int, k: int) -> UOp:
  out = UOp.placeholder((rows,), dtypes.float32, 0)
  words = UOp.placeholder((rows * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1)
  x = UOp.placeholder((k,), dtypes.float16, 2)
  return q4k_g3_lanemap_gemv_kernel(rows, k)(out, words, x)


def _q6k_gemv_ast(rows: int, k: int, *, parts: int = 1, use_coop: bool = True,
                  reduction: str = "external_sum", row_tile: int = 4) -> UOp:
  spec = q6k_spec_for_role(rows, k, parts=parts, row_tile=row_tile, use_coop=use_coop, reduction=reduction)
  extent = 1 if reduction == "in_kernel" else spec.partial_axis_extent
  shape = (rows,) if extent == 1 else (rows, extent)
  partials = UOp.placeholder(shape, dtypes.float32, 0)
  halfs = UOp.placeholder((rows * (k // Q6_K_BLOCK_ELEMS) * Q6K_HALFWORDS_PER_BLOCK,), dtypes.uint16, 1)
  x = UOp.placeholder((k,), dtypes.float16, 2)
  return emit_q6k_gemv_kernel(spec)(partials, halfs, x)


def _q6k_vocab_reduce_ast() -> UOp:
  spec = q6k_spec_for_role(151936, 4096, parts=1, row_tile=4, use_coop=True)
  out = UOp.placeholder((spec.rows,), dtypes.float32, 0)
  partials = UOp.placeholder((spec.rows, spec.partial_axis_extent), dtypes.float32, 1)
  return emit_q6k_vocab_scalar_reduce_kernel(spec)(out, partials)


def _flash_tile_ast() -> UOp:
  spec = describe_flash_decode_attention(32, 128, 8, MAXC, 48, fused_combine=True,
                                          query_group_size=None, stage_width=1)
  tc = UOp.variable("start_pos", 0, MAXC - 1) + 1
  pout = UOp.placeholder((32 * 48 * (128 + 2),), dtypes.float32, 0)
  q = UOp.placeholder((32 * 128,), dtypes.float16, 1)
  cache = UOp.placeholder((2, 1, 8, MAXC, 128), dtypes.float16, 2)
  return spec.emit_tile(tc)(pout, q, cache)


def _flash_combine_ast() -> UOp:
  spec = describe_flash_decode_attention(32, 128, 8, MAXC, 48, fused_combine=True,
                                          query_group_size=None, stage_width=1)
  out = UOp.placeholder((32 * 128,), dtypes.float32, 0)
  pout = UOp.placeholder((32 * 48 * (128 + 2),), dtypes.float32, 1)
  return spec.emit_combine()(out, pout)


# Campaign shapes/roles (nv-performance-campaign-scope-20260801.md section 14.1):
# gate/up 12288x4096, q/o 4096x4096, down 4096x12288, k/v 1024x4096; Q6_K vocab head 151936.
KERNELS = [
  ("q4k_g3_lanemap_gemv_12288_4096", "gate/up", lambda: _q4k_ast(12288, 4096)),
  ("q4k_g3_lanemap_gemv_4096_4096", "q/o", lambda: _q4k_ast(4096, 4096)),
  ("q4k_g3_lanemap_gemv_4096_12288", "down", lambda: _q4k_ast(4096, 12288)),
  ("q4k_g3_lanemap_gemv_1024_4096", "k/v", lambda: _q4k_ast(1024, 4096)),
  ("q6k_gen_coop_4096_12288", "down", lambda: _q6k_gemv_ast(4096, 12288)),
  ("q6k_gen_coop_151936_4096", "vocab", lambda: _q6k_gemv_ast(151936, 4096)),
  ("q6k_gen_partial_1024_4096_4", "k/v", lambda: _q6k_gemv_ast(1024, 4096, parts=4, use_coop=False)),
  ("q6k_gen_coop_4096_12288_inkernel", "down", lambda: _q6k_gemv_ast(4096, 12288, reduction="in_kernel", row_tile=2)),
  ("q6k_vocab_scalar_reduce_151936_4096", "vocab", _q6k_vocab_reduce_ast),
  ("flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", "G4", _flash_tile_ast),
  ("flash_fused_gmax_combine_32_128", "G4", _flash_combine_ast),
]


def make_renderer(name: str) -> Renderer:
  if name == "hip": return HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  if name == "metal":
    if sys.platform != "darwin":
      raise SystemExit("MetalRenderer is macOS-only (it imports the macOS Metal runtime); "
                       "run the metal arm on the macOS box, or use --renderer hip here.")
    return MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  raise SystemExit(f"unknown renderer {name!r}; use --renderer hip (default) or --renderer metal (macOS only)")


def main() -> None:
  renderer = "hip"
  args = sys.argv[1:]
  while args:
    arg = args.pop(0)
    if arg == "--renderer": renderer = args.pop(0)
    elif arg.startswith("--renderer="): renderer = arg.split("=", 1)[1]
    else: raise SystemExit(f"unknown argument {arg!r}; usage: --renderer hip|metal")
  ren = make_renderer(renderer)
  # These gates change rendered source; record their values so a drift is attributable.
  print(f"renderer={type(ren).__name__} target={ren.target} "
        f"PREFILL_SOFTMAX_REDUCE_FUSE={getenv('PREFILL_SOFTMAX_REDUCE_FUSE', 1)} "
        f"DECODE_STAGE_COALESCE={getenv('DECODE_STAGE_COALESCE')} "
        f"DECODE_FAST_EXP2={getenv('DECODE_FAST_EXP2', 0)} "
        f"DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE={getenv('DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE', 0)}")
  results = []
  for name, role, build in KERNELS:
    try:
      src = render_only(build(), ren)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep over a single bad kernel
      print(f"{name:58s} {role:8s} FAILED: {exc!r}")
      results.append((name, None))
      continue
    digest = hashlib.sha256((src + "\n").encode()).hexdigest()
    counts = " ".join(f"{m}={src.count(m)}" for m in MARKERS)
    print(f"{name:58s} {role:8s} sha256={digest[:12]} src_len={len(src):7d} {counts}")
    results.append((name, digest))
  if any(digest is None for _, digest in results):
    print(f"\n{sum(digest is None for _, digest in results)} of {len(results)} kernels FAILED to render", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
