"""Hermetic tests for the M4 residual-boundary fold probe
(docs/task_workflow/input/m4-variant-reopen-boundary-p0-scope-20260806.md section 4, probe 1).

The probe evaluates whether the o-proj residual_add slot
``epi_inputs["residual"][:, 0, :].reshape(N).cast(fp32)`` can fold to a zero-copy view of the
ordinary block-output producer under an EXTENDED typed-view contract. The M5 combine contract
must not be weakened: the M5 validator in kernel_program.py is untouched, and the probe's
extended validator is a separate, residual-slot-only opt-in.
"""
import pytest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue
from tinygrad.uop.ops import Ops

from extra.llm_research.decode.m4_residual_boundary_fold_probe import (
  EPI_RESADD, ExtendedResidualViewRequest, _block_output_producer, _fresh, _gemv_program,
  _is_pure_view, _layer0_producer, _schedule, evaluate_producer_form,
  extended_validated_typed_view, probe, residual_chain)


def test_real_block_output_chain_is_pure_view_of_ordinary_producer():
  producer = _block_output_producer()
  chain = residual_chain(producer).uop
  # The real chain bottoms out at the block's CONTIGUOUS-over-precompiled-boundary producer.
  assert chain.op is Ops.RESHAPE
  assert chain.base.op is Ops.CONTIGUOUS
  assert chain.numel() == 4096
  assert _is_pure_view(chain) is True


def test_extended_validator_folds_the_real_residual_chain():
  producer = _block_output_producer()
  chain = residual_chain(producer).uop
  program = _gemv_program(Q4KGEMVEpilogue("residual_add"))
  view, reason = extended_validated_typed_view(chain, ExtendedResidualViewRequest(), program)
  assert reason == "ok"
  assert view is not None and view.op is Ops.CONTIGUOUS


def test_fold_removes_copy_class_with_zero_materialization():
  row = evaluate_producer_form("block_output", _block_output_producer())
  # The exact precompiled-output identity preserves the block-output chain through the
  # transport contiguous, so the generic ABI is already copy-free for this producer form.
  assert row["without_abi"]["copies"] == []
  # With the extended fold the GEMV binds the producer's flat buffer.
  assert row["with_fold"]["copies"] == []
  assert row["with_fold"]["gemv_residual_buf"] == {"shape": [4096], "dtype": "dtypes.float"}
  assert row["validator"]["fold"] is True
  assert row["validator"]["base_op"] == "Ops.CONTIGUOUS"


def test_fail_closed_without_abi_keeps_the_copy():
  # The no-ABI arm is the same consumer graph the M4 census measured: copy present per token.
  row = evaluate_producer_form("block_output", _block_output_producer())
  assert [n for n in row["without_abi"]["names"] if n == EPI_RESADD] == [EPI_RESADD]


def test_layer0_embedding_producer_fails_closed():
  producer = _layer0_producer()
  chain = residual_chain(producer).uop
  program = _gemv_program(Q4KGEMVEpilogue("residual_add"))
  view, reason = extended_validated_typed_view(chain, ExtendedResidualViewRequest(), program)
  assert view is None
  assert "producer has no buffer/precompiled-output identity" in reason
  row = evaluate_producer_form("layer0_embedding", producer)
  assert row["validator"]["fold"] is False
  # Fail-closed ABI: the layer-0 chain (CAST/REDUCE, no identity) keeps its materializing
  # kernel before the GEMV instead of folding to a view of the producer.
  assert row["without_abi"]["names"] == ["test", EPI_RESADD]
  assert EPI_RESADD in row["without_abi"]["names"]


def test_plain_buffer_producer_folds():
  # Contrast form: a raw BUFFER producer is accepted by the extended contract (buffer identity).
  producer = _fresh("plain_buffer")
  view, reason = extended_validated_typed_view(residual_chain(producer).uop,
                                               ExtendedResidualViewRequest(),
                                               _gemv_program(Q4KGEMVEpilogue("residual_add")))
  assert reason == "ok" and view is not None


def test_extended_opt_in_is_residual_slot_specific():
  producer = _block_output_producer()
  chain = residual_chain(producer).uop
  program = _gemv_program(Q4KGEMVEpilogue("residual_add"))
  # Wrong slot: the activation slot (1) must never fold under this contract.
  assert extended_validated_typed_view(chain, ExtendedResidualViewRequest(slot=1), program)[0] is None
  # Wrong route_role.
  assert extended_validated_typed_view(chain, ExtendedResidualViewRequest(route_role="ffn_down"), program)[0] is None
  # Wrong epilogue kind (the contract is residual_add only; ffn_down prelude stays rejected).
  assert extended_validated_typed_view(chain, ExtendedResidualViewRequest(kind="ffn_down_fused"), program)[0] is None
  # Wrong dtype: the residual slot is fp32; an fp16 request must fail closed.
  assert extended_validated_typed_view(chain, ExtendedResidualViewRequest(dtype=dtypes.float16), program)[0] is None


def test_impure_chain_rejects_to_generic_abi():
  producer = _block_output_producer()
  # A data-moving permute between the producer and the request breaks the pure-view proof.
  impure = producer.transpose(1, 2).reshape(1, 1, 4096)
  chain = impure[:, 0, :].reshape(4096).cast(dtypes.float32).uop
  view, reason = extended_validated_typed_view(chain, ExtendedResidualViewRequest(),
                                               _gemv_program(Q4KGEMVEpilogue("residual_add")))
  assert view is None
  assert "contiguous offset-0 reshape" in reason


def test_m5_validator_still_rejects_the_residual_chain():
  # The M5 combine validator (kernel_program._validated_typed_view) must keep rejecting this
  # chain: the residual producer is not a declared typed AFTER. The M5 ABI is untouched.
  from tinygrad.llm.kernel_program import TypedViewRequest, _validated_typed_view
  producer = _block_output_producer()
  chain = residual_chain(producer).uop
  program = _gemv_program(Q4KGEMVEpilogue("residual_add"))
  view, reason = _validated_typed_view(chain, TypedViewRequest(slot=2, dtype=dtypes.float32,
    flat_shape=(4096,), route_role="attn_qo", requires_combine_fusion=True), program)
  assert view is None
  assert "contiguous" in reason or "AFTER" in reason or "declared" in reason


def test_probe_schema_is_stable():
  result = probe()
  assert result["schema"] == "tinygrad.m4_residual_boundary_fold_probe.v1"
  assert result["consumer"] == EPI_RESADD
  assert {r["form"] for r in result["rows"]} == {"block_output", "layer0_embedding", "plain_buffer"}
