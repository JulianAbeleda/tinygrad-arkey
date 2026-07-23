import pytest

from extra.qk.shared_attention_capture import SharedAttentionSynchronization, _derive_synchronization


HIP_WAIT = """for (int tile=0; tile<4; tile++) {
  *(buf0+lane) = value;
  __builtin_amdgcn_s_waitcnt(64519);
  half val1 = (*(buf0+lane));
}"""
ISA_WAIT = """loop:
  ds_store_b16(...)
  s_waitcnt(64519)
  ds_load_u16(...)
  s_cbranch loop
"""


def test_single_wave_capture_requires_ordered_lds_wait():
  assert _derive_synchronization((32,),32,HIP_WAIT,ISA_WAIT) == SharedAttentionSynchronization("wave",1,1,0)
  with pytest.raises(ValueError,match="missing, duplicated, or misordered"):
    _derive_synchronization((32,),32,HIP_WAIT.replace("__builtin_amdgcn_s_waitcnt(64519);", ""),ISA_WAIT)
  with pytest.raises(ValueError,match="missing, duplicated, or misordered"):
    _derive_synchronization((32,),32,HIP_WAIT,ISA_WAIT.replace("s_waitcnt(64519)\n  ds_load", "ds_load\n  s_waitcnt(64519)"))


def test_zero_barrier_capture_rejected_without_single_wave_proof():
  with pytest.raises(ValueError,match="requires one workgroup barrier"):
    _derive_synchronization((64,),32,HIP_WAIT,ISA_WAIT)
