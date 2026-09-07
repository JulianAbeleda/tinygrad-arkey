import pytest

from tinygrad.llm.fused_attention import _flash_prefill_identity
from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec

def _spec(**kwargs):
  args=dict(Hq=32,Hkv=8,Hd=128,q_tokens=512,kv_tokens=512,causal=True,
    scale=128**-0.5,valid_kv=512,query_start=0,target="nv_sm120",warps_per_cta=4)
  return FlashPrefillAttentionSpec(**(args|kwargs))

def test_q_rope_stage_changes_complete_identity_and_kernel_name():
  control, staged = _spec(), _spec(q_rope_stage=True)
  assert _flash_prefill_identity(control) != _flash_prefill_identity(staged)
  assert _flash_prefill_identity(staged) == _flash_prefill_identity(_spec(q_rope_stage=True))
  assert control.emitted_kernel_names != staged.emitted_kernel_names
  assert staged.emitted_kernel_names == ("nv_sm120_q16_grid_hd128_loop_attention_q_rope_stage",)
  assert staged.to_json()["q_rope_stage"] is True and staged.to_json()["warps_per_cta"] == 4

@pytest.mark.parametrize("change", [
  {"causal":False}, {"scale":0.125}, {"valid_kv":496}, {"query_start":16},
  {"phase_abi_v1":True}, {"acc_blocks":4}, {"output_block_base":4},
])
def test_every_codegen_field_changes_identity(change):
  args=_spec().to_json() | change
  assert _flash_prefill_identity(FlashPrefillAttentionSpec(**args)) != _flash_prefill_identity(_spec())

def test_q_rope_stage_fails_closed_outside_exact_nv_geometry():
  with pytest.raises(ValueError, match="exact NV pp512"):
    _spec(q_rope_stage=True,warps_per_cta=1).validate()
