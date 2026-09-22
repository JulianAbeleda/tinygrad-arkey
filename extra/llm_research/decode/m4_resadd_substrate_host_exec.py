"""M4 resadd rangeify substrate S3: host-execution proof of the folded epi_resadd subgraph on CPU.

Scope: docs/task_workflow/input/m4-resadd-rangeify-substrate-scope-20260806.md section S3 (fallback
arm): "execute a single-block folded epi_resadd subgraph on CPU with the residual as the block-output
AFTER and assert numeric equality against the copy-ABI variant."

The folded kernel is the production epi_resadd emitter shape: out = Q4K GEMV contrib + residual, where
the residual arg is the block-output producer's CONTIGUOUS(1,1,N) base (the fold fires inside
execute_promoted_program via _fold_residual_input_views). This is also the first RENDER proof of the
substrate: `UOp.placeholder_like` reshapes a non-flat arg, so the opaque residual reaches the
kernel body as RESHAPE(PARAM, shape-STACK); the S3 codegen fold in `pm_index_is_shrink`
(`tinygrad/codegen/__init__.py`) folds shaped GLOBAL-ptr PARAM views to the flat PARAM and loads
the scalar pointer-typed INDEX value read, making the folded kernel renderable. The body reads
the arg with a flat row index using the production `.cast(dtypes.float32)` spelling.

CPU only. Never realize on NV.
"""
import os
import sys
import hashlib

if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tinygrad.helpers import DEV, Context

import numpy as np
from tinygrad import Tensor, dtypes, UOp
from tinygrad.function import function
from tinygrad.llm.memory_semantics import runtime_activation
from tinygrad.llm.decode_kernels import _q4k_block_dot_packed_load, Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
                                         ResidualViewRequest, execute_promoted_program, _validated_residual_view)
from tinygrad.uop.ops import Ops, KernelInfo

N, K = 256, 1024
K_BLOCKS = K // Q4_K_BLOCK_ELEMS


def block_producer(n: int) -> Tensor:
  @function(precompile=True, allow_implicit=True)
  def block(x: Tensor, ffn_out: Tensor) -> Tensor:
    h = runtime_activation(x + ffn_out)
    return runtime_activation((h + ffn_out).contiguous())
  x = Tensor(np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(1, 1, n))
  ffn_out = Tensor((np.arange(n, dtype=np.float32) * 0.25).reshape(1, 1, n))
  return runtime_activation(block(x, ffn_out).contiguous())


def layer0_producer(n: int) -> Tensor:
  tokens = Tensor(np.zeros((1, 1, n), dtype=np.int32))
  emb = Tensor(np.ones((n, n), dtype=np.float16))
  return runtime_activation(tokens @ emb).float()


def residual_chain(producer: Tensor, n: int) -> Tensor:
  return producer[:, 0, :].reshape(n).cast(dtypes.float32)


def epi_resadd_host_kernel(rows: int, k: int):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  def kernel(out: UOp, words: UOp, x: UOp, *extra: UOp) -> UOp:
    row = UOp.range(rows, 0)
    contrib = UOp.const(dtypes.float32, 0.0)
    for blk in range(k_blocks):
      base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
      for lane4 in range(4):
        contrib = contrib + _q4k_block_dot_packed_load(words, x, base,
          UOp.const(dtypes.weakint, blk), UOp.const(dtypes.int32, lane4))
    return out[row].store(contrib + extra[0][row].cast(dtypes.float32)).end(row).sink(
      arg=KernelInfo(name=f"epi_resadd_host_{rows}_{k}", opts_to_apply=()))
  return kernel


def build(opt_in: bool, n: int, k: int, words: np.ndarray, xv: np.ndarray, resid: Tensor) -> Tensor:
  program = KernelProgram('decode_q4k_g3_generated', 'quant_linear_decode.q4k_generated_g3.gemv',
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, epi_resadd_host_kernel(n, k),
    output_spec=OutputSpec((n,), dtypes.float32),
    residual_input_views=(ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(n,),
                                              route_role='attn_qo', kind='residual_add'),) if opt_in else ())
  wt = Tensor(words).contiguous()
  xt = Tensor(xv.astype(np.float16)).contiguous()
  return execute_promoted_program(Tensor.zeros(n, dtype=dtypes.float32), wt, xt, resid, program=program)


def run_proof(n: int = N, k: int = K, seed: int = 42) -> dict:
  with Context(DEV="CPU"):
    rng = np.random.default_rng(seed)
    k_blocks = k // Q4_K_BLOCK_ELEMS
    words = rng.integers(0, 2**31, size=(n, k_blocks, Q4K_WORDS_PER_BLOCK), dtype=np.uint32)
    # Keep every block's d/dmin finite (fp16 1.0 / 0.0) so the packed dequant is
    # real-valued. Random scale words otherwise decode to NaN/Inf and make the
    # bitwise fold-vs-copy comparison depend on NaN payload propagation.
    words[:, :, 0] = 0x00003C00
    words = words.reshape(-1)
    xv = rng.standard_normal(k).astype(np.float16)
    request = ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(n,),
                                  route_role='attn_qo', kind='residual_add')
    prog = KernelProgram('decode_q4k_g3_generated', 'quant_linear_decode.q4k_generated_g3.gemv',
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED, epi_resadd_host_kernel(n, k),
      output_spec=OutputSpec((n,), dtypes.float32), residual_input_views=(request,))
    resid = residual_chain(block_producer(n), n)
    base, reason = _validated_residual_view(resid.uop, request, prog)
    result = {"fold": base is not None, "fold_base": None if base is None else str(base.op), "fold_reason": reason}
    base0, reason0 = _validated_residual_view(residual_chain(layer0_producer(n), n).uop, request, prog)
    result["layer0_reject"] = base0 is None
    result["layer0_reason"] = reason0

    # The copy ABI materializes the residual before the kernel; realize it once so
    # both arms consume the same concrete buffer instead of a lazy precompiled GETTUPLE.
    materialized = resid.realize()
    fold = build(True, n, k, words, xv, materialized).realize()
    copy = build(False, n, k, words, xv, materialized).realize()
    a, b = fold.numpy(), copy.numpy()
    result["fold_eq_copy_bitwise"] = a.tobytes() == b.tobytes()
    result["sha_fold"] = hashlib.sha256(a.tobytes()).hexdigest()[:16]
    result["sha_copy"] = hashlib.sha256(b.tobytes()).hexdigest()[:16]

    zwords = np.zeros(n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK, dtype=np.uint32)
    zresid = residual_chain(block_producer(n), n).realize()
    zfold = build(True, n, k, zwords, xv, zresid).realize()
    zcopy = build(False, n, k, zwords, xv, zresid).realize()
    za, zb = zfold.numpy(), zcopy.numpy()
    zproducer = residual_chain(block_producer(n), n).realize().numpy()
    result["zero_dot_fold_eq_copy"] = za.tobytes() == zb.tobytes()
    result["zero_dot_fold_eq_producer"] = za.tobytes() == zproducer.tobytes()
    return result


def main():
  with Context(DEV="CPU"):
    if os.environ.get("M4_RESADD_BOUNDARY_DEBUG"):
      print("fold validation:", flush=True)
    result = run_proof()
    for key, value in result.items():
      print(f"{key}: {value}", flush=True)
    ok = result["fold"] and result["layer0_reject"] and result["fold_eq_copy_bitwise"] and \
         result["zero_dot_fold_eq_copy"] and result["zero_dot_fold_eq_producer"]
    print("PROTO DONE" if ok else "PROTO FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
