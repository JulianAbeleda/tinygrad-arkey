from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute, generated_warmstart_entry
from tinygrad.llm.qk_layout import Q4_K, Q6_K


def test_generated_warmstart_entry_preserves_plan_identity_without_production_admission():
  for quant in (Q4_K,Q6_K):
    row=PackedWmmaRoute(quant,"ffn_gate_up",(512,12288,4096),(256,64,32,8,1,1),f"generated-{quant.name}")
    entry=generated_warmstart_entry(row)
    assert entry["canonical_identity"]==f"generated-{quant.name}"
    assert entry["context"].canonical_identity==entry["canonical_identity"]
    assert entry["context"].geometry.tile==(256,64,32)
    assert entry["context"].geometry.waves==(8,1)
    assert entry["transform"].quant_format is quant
    assert entry["transform"].rows==12288 and entry["transform"].k==4096
