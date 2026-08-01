import importlib.util

import pytest

from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec, describe_flash_prefill_attention


def test_flash_prefill_descriptor_promoted_out_of_extra():
  assert importlib.util.find_spec("tinygrad.schedule.wmma.flash_prefill") is not None
  assert importlib.util.find_spec("extra.llm_research.prefill.flash_prefill_attention_spec") is None


def test_flash_prefill_descriptor_validates_and_serializes():
  spec = describe_flash_prefill_attention(32, 8, 16, 512, causal=True, scale=0.125, valid_kv=384, query_start=368)
  assert spec.validate() is spec
  assert spec.emitted_kernel_names == ("amd_gfx1100_q16_grid_hd128_loop_attention",)
  assert spec.to_json() == {
    "Hq": 32, "Hkv": 8, "Hd": 128, "q_tokens": 16, "kv_tokens": 512, "causal": True,
    "valid_kv": 384, "query_start": 368, "acc_blocks": 8, "output_block_base": 0,
    "phase_abi_v1": False, "scale": 0.125, "target": "amd_gfx1100",
  }

  with pytest.raises(ValueError, match="positive 16-multiple head_dim <=128"):
    FlashPrefillAttentionSpec(32, 8, 16, 512, True, 0.125, Hd=72).validate()


def test_flash_prefill_descriptor_emit_threads_builder_arguments(monkeypatch):
  import tinygrad.schedule.wmma as wmma

  captured = {}

  def fake_builder(q, k, v, out, **kwargs):
    captured.update({"args": (q, k, v, out), **kwargs})
    return "emitted"

  monkeypatch.setattr(wmma, "amd_gfx1100_q16_grid_hd128_loop_attention", fake_builder)
  spec = FlashPrefillAttentionSpec(40, 8, 16, 1024, True, 0.125, valid_kv=768, query_start=752,
                                   acc_blocks=4, output_block_base=4, phase_abi_v1=True)
  emit = spec.emit(kernel_info="kernel-info")
  assert emit("out", "q", "k", "v") == "emitted"
  fm = captured.pop("fragment_model")
  assert fm.target == "amd_gfx1100" and fm.score_elements == 8 and fm.calls_per_tile == 1
  assert captured == {
    "args": ("q", "k", "v", "out"), "q_tokens": 16, "q_heads": 40, "kv_heads": 8,
    "kv_tokens": 1024, "scale": 0.125, "causal": True, "valid_kv": 768, "query_start": 752,
    "output_block_base": 4, "acc_blocks": 4, "phase_abi_v1": True, "head_dim": 128,
    "kernel_info": "kernel-info",
  }

  nv_spec = FlashPrefillAttentionSpec(32, 8, 16, 512, True, 0.125, target="nv_sm120")
  assert nv_spec.emitted_kernel_names == ("nv_sm120_q16_grid_hd128_loop_attention",)


def test_flash_prefill_nv_target_threads_the_nv_fragment_model(monkeypatch):
  import tinygrad.schedule.wmma as wmma

  captured = {}

  def fake_builder(q, k, v, out, **kwargs):
    captured.update(kwargs)
    return "emitted"

  monkeypatch.setattr(wmma, "amd_gfx1100_q16_grid_hd128_loop_attention", fake_builder)
  spec = FlashPrefillAttentionSpec(32, 8, 16, 512, True, 0.125, target="nv_sm120")
  spec.emit()("out", "q", "k", "v")
  fm = captured["fragment_model"]
  assert fm.target == "nv_sm120" and fm.calls_per_tile == 2 and fm.score_elements == 8
  assert captured["kernel_info"].name == "nv_sm120_q16_grid_hd128_loop_attention"
