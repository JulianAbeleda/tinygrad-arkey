import ctypes
from scratchpad.llama_cuda_q4k_gate_up_oracle import ENTRY, FusionArgs, UInt3, params

def test_fused_q4k_entry_and_abi_are_pinned():
  assert 'type12' in ENTRY and 'ELb1ELb0' in ENTRY
  assert ctypes.sizeof(FusionArgs) == 32 and ctypes.sizeof(UInt3) == 12

def test_fused_gate_pointer_is_in_the_device_fusion_argument():
  # Source-level construction is deliberately tested without CUDA allocation.
  assert FusionArgs._fields_[1][0] == 'gate'
