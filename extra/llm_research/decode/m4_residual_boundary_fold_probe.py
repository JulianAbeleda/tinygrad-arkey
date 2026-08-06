"""M4 variant-reopen P0 probe 1 - ordinary-producer residual typed-view fold (CPU-only, hermetic).

Scope: `docs/task_workflow/input/m4-variant-reopen-boundary-p0-scope-20260806.md` section 4,
probe 1. Question: can the o-proj residual_add slot
``epi_inputs["residual"][:, 0, :].reshape(N).cast(fp32)`` (decode_routes.py:97-98) fold to a
zero-copy view of its ordinary producer under an EXTENDED typed-view contract, without touching
the M5 combine contract?

This probe is an evaluation, not an implementation. It never edits kernel_program.py /
decode_routes.py / decode_kernels.py; the extended validator below is a probe-owned predicate
that reports what a production extension would have to prove. The mechanical proof is the
schedule itself: with the fold applied, the E_32_32_4_86a2-shaped boundary copy must vanish
from the linearized graph; without the ABI it must remain.

The real chain (verified against model.py's block construction): the residual producer is the
previous block's output, ``MS(CONTIGUOUS(GETTUPLE(precompiled FUNCTION)))`` (the
``@function(precompile=True)`` boundary at model.py:600/658 plus the model's own
``.contiguous()`` and runtime-activation wrapper). The consumer chain is pure movement
(RESHAPE/RESHAPE over the producer), so the only materialization is the opaque boundary's
preserve-or-materialize copy.
"""
from __future__ import annotations

import dataclasses, json
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.function import function
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
                                         execute_promoted_program, _is_pure_contiguous_view)
from tinygrad.llm.memory_semantics import runtime_activation
from tinygrad.uop.ops import Ops

N = 4096
EPI_RESADD = f"q4k_g3_lanemap_gemv_epi_resadd_{N}_{N}"
LEGACY = f"q4k_g3_lanemap_gemv_{N}_{N}"

# Ops that never move data by themselves: CONTIGUOUS over a pure view is an identity physical
# layout, MEMORY_SEMANTIC is a role marker. Both are transparent to the index map.
_TRANSPARENT = frozenset((Ops.CONTIGUOUS, Ops.MEMORY_SEMANTIC))
_MOVEMENT = frozenset((Ops.RESHAPE, Ops.SLICE, Ops.PERMUTE, Ops.EXPAND))


@dataclasses.dataclass(frozen=True)
class ExtendedResidualViewRequest:
  """Probe-owned opt-in descriptor for the residual slot. Mirrors the fields a production
  TypedViewRequest would need; deliberately NOT a TypedViewRequest so the M5 ABI is untouched."""
  slot: int = 2
  dtype: Any = dtypes.float32
  flat_shape: tuple[int, ...] = (N,)
  route_role: str = "attn_qo"
  kind: str = "residual_add"


def residual_chain(producer: Tensor) -> Tensor:
  """The exact decode_routes.py:97-98 residual slot expression: slice row 0, flatten, identity cast.
  No explicit .contiguous(): the opaque boundary adds it at custom_kernel time."""
  return producer[:, 0, :].reshape(N).cast(dtypes.float32)


def _block_output_producer() -> Tensor:
  """Reconstruct the real residual producer: the previous block's output. model.py:616-624
  computes h = attn_out (epi open) / x + attn_out, then FFNBlock.__call__ (model.py:658) returns
  ``_prefill_semantic(..., _run(...).contiguous())`` where _run is an ``@function(precompile=True,
  allow_implicit=True)`` boundary (model.py:600). The epi-open decode chain therefore hands the
  NEXT block a residual whose uop is MS(CONTIGUOUS(GETTUPLE(FUNCTION(...))))."""
  @function(precompile=True, allow_implicit=True)
  def block(x: Tensor, ffn_out: Tensor) -> Tensor:
    h = runtime_activation(x + ffn_out)
    return runtime_activation((h + ffn_out).contiguous())
  x, ffn_out = Tensor.empty(1, 1, N), Tensor.empty(1, 1, N)
  return runtime_activation(block(x, ffn_out).contiguous())


def _layer0_producer() -> Tensor:
  """Block-0 residual: token_embd(tokens).float() (model.py:1241) - an fp16 matmul upcast to
  fp32, wrapped in the runtime-activation marker. Its base is a CAST over a REDUCE, which has no
  buffer identity and no precompiled output identity: the extended contract must fail closed."""
  tokens = Tensor.empty(1, 1, 4096, dtype=dtypes.int32)
  emb = Tensor.empty(4096, 4096, dtype=dtypes.float16)
  return runtime_activation(tokens @ emb).float()


def _buffer_producer() -> Tensor:
  """Plain realized-shaped fp32 producer (contrast form): a BUFFER base with buffer identity."""
  return Tensor.empty(1, 1, N)


def _is_pure_view(chain: UOp) -> bool:
  """Probe purity proof: every leg from the request down to the producer base must be a pure
  view. RESHAPE is identity on the flat index by construction; SLICE must be offset-0; PERMUTE
  and EXPAND must be identity; CONTIGUOUS and MEMORY_SEMANTIC are transparent physical/role
  markers. dtype is preserved through every leg. This is intentionally structural: the M5
  rewrite cannot see through the precompiled GETTUPLE boundary, so a rewrite-only proof would
  wrongly reject the real chain."""
  cur = chain
  while cur.op in _TRANSPARENT | _MOVEMENT:
    if cur.op is Ops.SLICE:
      offsets, _ = cur.arg
      if any(o != 0 for o in offsets): return False
    if cur.op is Ops.PERMUTE and tuple(cur.arg) != tuple(range(len(cur.shape))): return False
    if cur.op is Ops.EXPAND and tuple(cur.arg) != tuple(cur.shape): return False
    if cur.dtype is not chain.dtype: return False
    cur = cur.src[0]
  return True


def _residual_producer_identity(base: UOp) -> bool:
  """The extended contract's base rule: an ordinary producer with a concrete buffer identity
  (BUFFER/SLICE/PARAM, or a precompiled function output - the real block boundary), or an
  AFTER with a declared typed output (M5 path, kept intact)."""
  if base.op is Ops.AFTER:
    from tinygrad.llm.kernel_program import _DECLARED_TYPED_OUTPUTS
    return _DECLARED_TYPED_OUTPUTS.get(base) is not None
  if base.op is Ops.CONTIGUOUS:
    return base.src[0].has_precompiled_output_identity() or base.src[0].has_buffer_identity() or base.src[0].op is Ops.AFTER
  return base.has_buffer_identity() or base.has_precompiled_output_identity()


def extended_validated_typed_view(uop: UOp, request: ExtendedResidualViewRequest,
                                  program: KernelProgram) -> tuple[UOp | None, str]:
  """EXTENDED residual-slot validator (probe-owned). Returns ``(base_uop, "ok")`` when the
  residual chain is a pure offset-0 view of an ordinary producer and the residual_add opt-in is
  exact; else ``(None, reason)``. Every failure keeps the generic flat-buffer ABI (the copy)."""
  # (a) residual-slot opt-in: slot 2, attn_qo, residual_add, q4k GEMV consumer
  if request.slot != 2: return None, "not the residual slot"
  if request.route_role != "attn_qo": return None, f"wrong consumer route_role {request.route_role!r}"
  if request.kind != "residual_add": return None, f"wrong epilogue kind {request.kind!r}"
  if not program.route_id.startswith("decode_q4k") or not program.program_id.endswith(".gemv"):
    return None, "program is not a q4k GEMV consumer"
  # (b) request chain is a movement-only view: dtype/numel preserved through every leg
  chain = uop.src[0] if uop.op is Ops.CONTIGUOUS else uop
  if chain.dtype is not request.dtype: return None, "request dtype mismatch"
  if chain.numel() != N: return None, "request numel mismatch"
  if chain.dtype is not uop.dtype: return None, "view dtype is not preserved through the request chain"
  # (c) purity: offset-0 row-major identity view (transparent legs stripped, M5 rewrite)
  if not _is_pure_view(chain): return None, "view is not a contiguous offset-0 reshape"
  # (d) producer: ordinary producer with buffer identity (or declared typed output)
  base = chain.base
  if not _residual_producer_identity(base): return None, "producer has no buffer/precompiled-output identity"
  return base, "ok"


def _gemv_program(epilogue: Q4KGEMVEpilogue | None = None) -> KernelProgram:
  return KernelProgram("decode_q4k_g3_generated", "quant_linear_decode.q4k_generated_g3.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_kernel(N, N, epilogue=epilogue),
    output_spec=OutputSpec((N,), dtypes.float32))


def _copy_calls(linear: UOp) -> list[dict]:
  """The E_32_32_4_86a2-shaped copy class: a CALL with exactly two same-shape/same-dtype buffers."""
  out = []
  for u in linear.toposort():
    if u.op is not Ops.CALL: continue
    bufs = [s.buf_uop for s in u.src[1:]]
    if len(bufs) == 2 and bufs[0].shape == bufs[1].shape and bufs[0].dtype == bufs[1].dtype:
      out.append({"name": getattr(getattr(u.src[0], "arg", None), "name", None),
                  "shape": list(bufs[0].shape), "dtype": str(bufs[0].dtype)})
  return out


def _schedule(producer: Tensor, words: Tensor, xv: Tensor, folded: bool, request: ExtendedResidualViewRequest | None) -> dict:
  """One schedule of the epi_resadd consumer: ``folded=True`` substitutes the extended validator's
  base (zero-copy view); ``folded=False`` is the generic flat-buffer ABI (today's behavior)."""
  resid = residual_chain(producer)
  program = _gemv_program(Q4KGEMVEpilogue("residual_add"))
  verdict = {"fold": False, "reason": "no extended request supplied"}
  if request is not None:
    view, reason = extended_validated_typed_view(resid.uop, request, program)
    verdict = {"fold": view is not None, "reason": reason, "base_op": None if view is None else str(view.op)}
  inp = Tensor(resid.uop.base) if (folded and verdict["fold"]) else resid
  out = execute_promoted_program(Tensor.empty(N, dtype=dtypes.float32), words, xv, inp, program=program)
  linear, _ = out.linear_with_vars()
  names = [getattr(getattr(u.src[0], "arg", None), "name", None) for u in linear.toposort() if u.op is Ops.CALL]
  gemv = [u for u in linear.toposort() if u.op is Ops.CALL
          and getattr(getattr(u.src[0], "arg", None), "name", None) == EPI_RESADD]
  residual_buf = None
  if gemv:
    bufs = [s.buf_uop for s in gemv[0].src[1:]]
    if len(bufs) >= 4: residual_buf = {"shape": list(bufs[3].shape), "dtype": str(bufs[3].dtype)}
  return {"verdict": verdict, "names": names, "copies": _copy_calls(linear),
          "gemv_residual_buf": residual_buf}


def evaluate_producer_form(form: str, producer: Tensor) -> dict:
  request = ExtendedResidualViewRequest()
  # Chain facts and the validator verdict are captured on a FRESH uop chain. Scheduling mutates
  # the graph (callify rewrites producer AFTERs to their buffers), so each row schedules its own
  # fresh producer to keep the no-ABI contrast honest.
  chain = residual_chain(producer).uop
  program = _gemv_program(Q4KGEMVEpilogue("residual_add"))
  view, reason = extended_validated_typed_view(chain, request, program)
  verdict = {"fold": view is not None, "reason": reason, "base_op": None if view is None else str(view.op)}
  base_row = _schedule(_fresh(form), Tensor.empty(N * N // 16, dtype=dtypes.uint32),
                       Tensor.empty(N, dtype=dtypes.float16), False, request)
  fold_row = _schedule(_fresh(form), Tensor.empty(N * N // 16, dtype=dtypes.uint32),
                       Tensor.empty(N, dtype=dtypes.float16), True, request)
  return {
    "form": form,
    "request_chain_op": str(chain.op),
    "chain_base_op": str(chain.base.op),
    "chain_base_shape": list(chain.base.shape),
    "chain_numel": chain.numel(),
    "validator": verdict,
    "without_abi": {"names": base_row["names"], "copies": base_row["copies"]},
    "with_fold": {"names": fold_row["names"], "copies": fold_row["copies"],
                  "gemv_residual_buf": fold_row["gemv_residual_buf"]},
  }


def _fresh(form: str) -> Tensor:
  """A new producer instance of the given form, so scheduling one row cannot rewrite the next."""
  if form == "block_output": return _block_output_producer()
  if form == "layer0_embedding": return _layer0_producer()
  if form == "plain_buffer": return _buffer_producer()
  raise ValueError(f"unknown producer form {form!r}")


def probe() -> dict:
  """Hermetic fold probe: the real block-output producer form, plus the fail-closed contrasts."""
  return {
    "schema": "tinygrad.m4_residual_boundary_fold_probe.v1",
    "consumer": EPI_RESADD,
    "legacy": LEGACY,
    "rows": [evaluate_producer_form("block_output", _block_output_producer()),
             evaluate_producer_form("layer0_embedding", _layer0_producer()),
             evaluate_producer_form("plain_buffer", _buffer_producer())],
  }


def main() -> None:
  print(json.dumps(probe(), indent=1))


if __name__ == "__main__":
  main()
