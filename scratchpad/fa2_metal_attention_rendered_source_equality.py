#!/usr/bin/env python3
"""FA2 Phase B -- Metal render of the fused prefill-attention kernel (compile-only, no GPU dispatch).

The Metal analogue of `scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py`. Same technique
(`FlashPrefillAttentionSpec.emit()` -> `Target.parse` + a render call, PARAM placeholders in production's
exact slot convention, causal=True, kv_tokens=q_tokens/start_pos=0, both ADMITTED_GRIDS), one material
difference: this machine has a REAL Metal toolchain (Darwin), so `tinygrad.codegen.to_program`'s normal
pipeline does not stay compile-only for `MetalRenderer` the way it does for `AMDISARenderer` (a pure-Python
ELF encoder, no native compiler) -- `to_program` unconditionally chains render -> `do_compile`, and
`do_compile` invokes the real Metal Shading Language compiler. This script reproduces
`tinygrad.codegen.do_to_program`'s own pipeline verbatim (codegen/__init__.py:459) but stops right after
`do_render`, before `do_compile` -- the same "compile-only" posture the task's own header specifies
("Compile-only; no GPU dispatch required"), and the same place `AMDISARenderer.asm()` stops (never
invoking comgr). No GPU is dispatched and no native Metal compiler is invoked by this script.

WHAT PHASE B ADDED (tinygrad/renderer/cstyle.py, MetalRenderer.__init__, gated on `self.tensor_cores`,
i.e. an Apple7+ simdgroup-matmul capability fact -- never `if backend == "METAL"`):
  - the 4 Ops FA0 found renderer-neutral, registered unmodified (same objects HIPRenderer registers):
    AMD_PACKED_FRAGMENT_LOAD, AMD_ROW_SOFTMAX_SLOT, AMD_PV_C_LANE, StateHandle phase publication.
  - Ops.AMD_ROW_SOFTMAX_REPACK routed through native_state=False (HIP precedent, cstyle.py:139-141).
  - the AMD-ds_bpermute-shaped "bpermute" CUSTOMI marker rewritten into Metal's own simd_shuffle_xor via
    the existing CStyleLanguage.warp_shfl_xor hook (cstyle.py:520), which MetalRenderer already implements.
  - Ops.AMD_ATTENTION_LOOP_STATE given a Metal buffer-indexed expansion, the identical shape to
    _hip_expand_attention_loop_state (HIP's own 8-line design; not a third one).
  - Ops.AMD_ATTENTION_OUTPUT_DRAIN (not one of FA0's six -- but structurally required to render this
    kernel's final store at all) reuses _hip_expand_attention_output_drain unmodified: it is ordinary
    UOp.special/index/store arithmetic with zero AMD-ISA content, exactly the same portability shape FA0
    established for the other four neutral Ops.
  - `_PREFILL_EMITTERS["metal"]` added to tinygrad/llm/fused_attention.py (the target-keyed seam already
    built for this -- a new dict entry, not a rewrite).

KNOWN, OUT-OF-SCOPE LIMITATION (reported, not fixed here): the rendered source uses AMD's fragment shape
verbatim (half16/float8 STACK operands, dims=(16,16,16) baked into the WMMA macro's NAME only) --
MetalRenderer's own __WMMA_16_16_16_half_float macro body (cstyle.py render_kernel) is hardware-correct on
its own terms (it always declares simdgroup_half8x8/simdgroup_float8x8 mat_a/mat_b/mat_c and calls the
real 8x8 simdgroup_multiply_accumulate, ignoring the macro's dims-derived NAME), but the macro's own
PARAMETER TYPES are `half2 a, half2 b` (Metal's true elements_per_thread=(2,2,2) fragment width) while this
render passes it `half16`/`float8` operands. A real `xcrun`/Metal compile of this exact source fails with
"invalid use of incomplete type 'half16'" -- confirmed directly (this script's own earlier iteration ran
the full `to_program` including `do_compile` and reproduced that exact real-compiler error). Metal genuinely
has no vector type wider than 4; AMD's 16-wide WMMA fragment convention does not fit Metal's simdgroup-
matrix convention by substitution alone. This is exactly the class of gap FA3 (numeric/hardware-correctness
measurement) exists to close, not FA2 (this port): FA2's own scope boundary is explicit that this is "a
correctness port [of the six/eight renderer-specific Ops], not a performance result," and the render/hash
below is the deliverable, not a claim that this text compiles on real Metal hardware today.

Usage: `python3 scratchpad/fa2_metal_attention_rendered_source_equality.py`
"""
from __future__ import annotations
import hashlib, sys
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from dataclasses import replace  # noqa: F401  (kept for parity with codegen/__init__.py's do_to_program shape)
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, graph_rewrite, KernelInfo, ProgramInfo
from tinygrad.codegen import full_rewrite_to_sink, do_linearize, do_estimates, do_render
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
from tinygrad.llm.fused_attention import ADMITTED_GRIDS

HEAD_DIM = 128  # matches FA-CTRL: the only head_dim ADMITTED_GRIDS' rows are proven at

# render only: do_compile deliberately excluded, matching AMDISARenderer.asm()'s no-native-compiler posture.
pm_render_only = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE)), name="prg"), do_linearize),
  (UPat(Ops.PROGRAM, src=(UPat(Ops.SINK, name="sink"), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_estimates),
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(Ops.DEVICE), UPat(Ops.LINEAR, name="lin")), name="prg"), do_render),
])


def render_only(ast: UOp, renderer) -> str:
  """Reproduces do_to_program (codegen/__init__.py:459) up to and including do_render; never reaches
  do_compile. No Device[...] opened, no GPU touched -- same posture as FA-CTRL's to_program call."""
  assert ast.op is Ops.SINK and isinstance(ast.arg, KernelInfo)
  full_sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
  prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=renderer.target.device)), arg=ProgramInfo.from_sink(full_sink))
  prg = graph_rewrite(prg, pm_render_only, ctx=renderer, name="linearize/render (no compile)")
  return next(u.arg for u in prg.src if u.op is Ops.SOURCE)


def render_grid(q_heads: int, kv_heads: int, q_tokens: int) -> tuple[str, int]:
  spec = FlashPrefillAttentionSpec(Hq=q_heads, Hkv=kv_heads, q_tokens=q_tokens, kv_tokens=q_tokens,
    causal=True, scale=1.0 / (HEAD_DIM ** 0.5), Hd=HEAD_DIM, valid_kv=q_tokens, query_start=0, target="metal")
  spec.validate()
  fxn = spec.emit()
  out_ph = UOp.placeholder((q_heads * q_tokens * HEAD_DIM,), dtypes.half, 0)
  q_ph = UOp.placeholder((q_heads * q_tokens * HEAD_DIM,), dtypes.half, 1)
  k_ph = UOp.placeholder((kv_heads * q_tokens * HEAD_DIM,), dtypes.half, 2)
  v_ph = UOp.placeholder((kv_heads * q_tokens * HEAD_DIM,), dtypes.half, 3)
  ast = fxn(out_ph, q_ph, k_ph, v_ph)
  renderer = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  source = render_only(ast, renderer)
  return source, source.count("simdgroup_multiply_accumulate")


def main() -> None:
  grids = sorted(ADMITTED_GRIDS)  # (32,8,512) then (40,8,512)
  results = []
  for q_heads, kv_heads, q_tokens in grids:
    label = f"Hq={q_heads:<3d} Hkv={kv_heads:<3d} q_tok={q_tokens:<5d} Hd={HEAD_DIM}"
    try:
      src, wmma_calls = render_grid(q_heads, kv_heads, q_tokens)
    except Exception as exc:  # noqa: BLE001
      print(f"{label} FAILED: {type(exc).__name__}: {exc}")
      results.append(((q_heads, kv_heads, q_tokens), None, None))
      continue
    digest = hashlib.sha256((src + "\n").encode()).hexdigest()
    print(f"{label} sha256={digest[:12]} simdgroup_multiply_accumulate={wmma_calls} "
          f"instructions={len(src.splitlines())} src_len={len(src)}")
    results.append(((q_heads, kv_heads, q_tokens), digest, wmma_calls))

  failed = [grid for grid, digest, _ in results if digest is None]
  if failed:
    print(f"\n{len(failed)} of {len(results)} admitted grids FAILED to render: {failed}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
