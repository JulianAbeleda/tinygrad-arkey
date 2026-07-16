from __future__ import annotations

import pytest

from tinygrad import Tensor
from tinygrad.llm.memory_semantics import (KV_CACHE, PREFILL_ACTIVATION, PREFILL_OUTPUT, PREFILL_SCRATCH,
                                           RUNTIME_ACTIVATION, RUNTIME_INPUT, RUNTIME_OUTPUT, RUNTIME_SCRATCH,
                                           memory_semantic_owner, prefill_output)
from tinygrad.llm.model import (GatedDeltaNetBlock, SSMConfig, Transformer, TransformerBlock, TransformerConfig,
                                _runtime_input_boundary)
from tinygrad.schedule.memory import collect_memory_plan_manifests


def _config(*, kv_quant:bool=False) -> TransformerConfig:
  return TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
    vocab_size=32, head_dim=4, rope_theta=10001.0, rope_dim=4, v_head_dim=4, max_context=8,
    kv_quant=kv_quant)


def _prepare(block:TransformerBlock, *, prefill:bool, v2:bool=False) -> None:
  block._is_prefill, block._prefill_v2 = prefill, v2
  block._use_flash, block._ring_full = False, False


def _rows(manifests): return [row for manifest in manifests for row in manifest.buffers]


def _assert_no_known_owner_conflicts(manifests) -> None:
  # The planner itself rejects conflicting live claims. Repeat the structural invariant over the collected artifact.
  for manifest in manifests:
    for index in range(len(manifest.indices)):
      live = [row for row in manifest.buffers if row.first_index <= index <= row.last_index]
      for n, row in enumerate(live):
        for other in live[:n]:
          overlap = row.arena_identity == other.arena_identity and max(row.byte_range[0], other.byte_range[0]) < \
                    min(row.byte_range[1], other.byte_range[1])
          if overlap and row.semantic_owner != "unknown" and other.semantic_owner != "unknown":
            assert row.semantic_owner == other.semantic_owner


def _assert_buffer_identity_owners_are_stable(manifests) -> None:
  owners = {}
  for row in _rows(manifests):
    if row.semantic_owner == "unknown": continue
    if row.identity in owners and owners[row.identity] != row.semantic_owner:
      # Cached schedule-transient storage may truthfully serve the same logical
      # role in prefill and decode on non-overlapping invocations.
      assert {owners[row.identity], row.semantic_owner} == {PREFILL_SCRATCH, RUNTIME_SCRATCH}
    else: owners[row.identity] = row.semantic_owner


@pytest.mark.parametrize("kv_quant", [False, True])
def test_prefill_attention_temporaries_are_scratch_without_relabeling_persistent_kv(kv_quant):
  block = TransformerBlock(_config(kv_quant=kv_quant))
  _prepare(block, prefill=True, v2=True)
  x = Tensor.rand(1, 3, 8)
  block._init_state(x)

  with collect_memory_plan_manifests() as manifests: block._attention(x, 0).realize()
  rows = _rows(manifests)
  assert any(row.semantic_owner == PREFILL_SCRATCH for row in rows)
  assert any(row.semantic_owner == PREFILL_ACTIVATION for row in rows)
  persistent = {f"buffer:{block.cache_kv.uop.buf_uop.key.hex()}"}
  if kv_quant: persistent.add(f"buffer:{block.cache_kv_scale.uop.buf_uop.key.hex()}")
  assert persistent <= {row.identity for row in rows}
  assert all(row.semantic_owner == KV_CACHE for row in rows if row.identity in persistent)
  _assert_no_known_owner_conflicts(manifests)


def test_transformer_block_residual_materialization_has_phase_specific_runtime_role():
  prefill_block = TransformerBlock(_config())
  _prepare(prefill_block, prefill=True)
  with collect_memory_plan_manifests() as manifests:
    prefill_block(Tensor.rand(1, 3, 8), 0).realize()
  assert any(row.semantic_owner == PREFILL_ACTIVATION for row in _rows(manifests))
  _assert_no_known_owner_conflicts(manifests)

  decode_block = TransformerBlock(_config())
  _prepare(decode_block, prefill=False)
  with collect_memory_plan_manifests() as decode_manifests:
    decode_block(Tensor.rand(1, 1, 8), 0).realize()
  decode_owners = {row.semantic_owner for row in _rows(decode_manifests)}
  assert RUNTIME_ACTIVATION in decode_owners
  assert not ({PREFILL_ACTIVATION, PREFILL_OUTPUT, PREFILL_SCRATCH} & decode_owners)


def test_transformer_decode_return_buffer_is_runtime_output():
  model = Transformer(_config())
  tokens, temperature = Tensor([[1]]), Tensor([1.0])
  with collect_memory_plan_manifests() as manifests: model.forward(tokens, 0, temperature).realize()
  assert RUNTIME_OUTPUT in {row.semantic_owner for row in _rows(manifests)}


def test_transformer_forward_return_buffer_is_prefill_output():
  model = Transformer(_config())
  for block in model.blk: _prepare(block, prefill=True, v2=True)
  tokens, temperature = Tensor([[1, 2, 3]]), Tensor([1.0])
  with collect_memory_plan_manifests() as manifests: model.forward(tokens, 0, temperature).realize()
  owners = {row.semantic_owner for row in _rows(manifests)}
  assert PREFILL_ACTIVATION in owners
  assert PREFILL_OUTPUT in owners
  _assert_no_known_owner_conflicts(manifests)


def test_generate_reuses_prefill_sample_on_process_selected_device_without_reclassification():
  model = Transformer(_config())
  with collect_memory_plan_manifests() as manifests:
    # The second independent request replays the decode JIT captured by the
    # first. This catches prompt-view/feedback input binding mismatches.
    generated = [list(zip(range(2), model.generate([1, 2, 3], chunk_size=2, temperature=0.0))) for _ in range(2)]
  assert all(len(run) == 2 for run in generated)
  assert any(row.semantic_owner == PREFILL_OUTPUT for row in _rows(manifests))
  feedback_index = next(i for i in range(len(manifests)-1, -1, -1)
                        if {RUNTIME_INPUT, RUNTIME_OUTPUT} <= {row.semantic_owner for row in manifests[i].buffers})
  prior_rows, feedback_rows = _rows(manifests[:feedback_index]), manifests[feedback_index].buffers
  prior_runtime_input_ids = {row.identity for row in prior_rows if row.semantic_owner == RUNTIME_INPUT}
  feedback_output_ids = {row.identity for row in feedback_rows if row.semantic_owner == RUNTIME_OUTPUT}
  feedback_input_ids = {row.identity for row in feedback_rows if row.semantic_owner == RUNTIME_INPUT}
  assert feedback_output_ids
  assert feedback_input_ids and feedback_input_ids.isdisjoint(feedback_output_ids)
  assert feedback_input_ids - prior_runtime_input_ids
  _assert_no_known_owner_conflicts(manifests)
  _assert_buffer_identity_owners_are_stable(manifests)


def test_feedback_runtime_input_boundary_is_lazy(monkeypatch):
  source = prefill_output((Tensor([1]) + 1).contiguous()).realize()
  source_owner = next(memory_semantic_owner(uop) for uop in source.uop.toposort()
                      if memory_semantic_owner(uop) is not None)
  assert source_owner == PREFILL_OUTPUT

  def forbidden_realize(*args, **kwargs): raise AssertionError("runtime input boundary realized eagerly")
  monkeypatch.setattr(Tensor, "realize", forbidden_realize)
  decode_input = _runtime_input_boundary(source)

  assert memory_semantic_owner(decode_input) == RUNTIME_INPUT
  assert not decode_input.uop.is_realized
  assert memory_semantic_owner(source) == PREFILL_OUTPUT


def test_recurrent_block_state_and_decode_producers_have_structural_roles():
  block = GatedDeltaNetBlock(_config(), SSMConfig(conv_kernel=3, state_size=4, group_count=1,
                                                   time_step_rank=1, inner_size=4))
  _prepare(block, prefill=False)
  x = Tensor.rand(1, 1, 8)
  block._init_state(x)
  with collect_memory_plan_manifests() as manifests:
    block(x, 0).realize()
  owners = {row.semantic_owner for row in _rows(manifests)}
  assert RUNTIME_SCRATCH in owners
  assert RUNTIME_ACTIVATION in owners
  assert all(row.semantic_owner != "unknown" for row in _rows(manifests)
             if row.identity in {f"buffer:{block.conv_state.uop.buf_uop.key.hex()}",
                                 f"buffer:{block.recurrent_state.uop.buf_uop.key.hex()}"})
