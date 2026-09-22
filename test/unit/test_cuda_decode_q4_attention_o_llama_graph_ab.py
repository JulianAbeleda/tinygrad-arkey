from scratchpad.cuda_decode_q4_attention_o_llama_graph_ab import ENTRY_Q4_FUSED, RESADD_PREFIX, select_attention_o

def test_attention_o_selection_survives_graph_boundaries():
  groups=[[7,21],[3,17,31,45,59],[23,37,51,65,79,93,107],list(range(100,158))]
  seen=0; selected=[]; ordinals=[]
  for group in groups:
    got,mapping=select_attention_o(group,seen); selected+=got; ordinals += [mapping[x] for x in got]; seen+=len(group)
  assert ordinals==list(range(1,71,2))
  assert 71 not in ordinals

def test_fused_entry_and_residual_add_identity_are_pinned():
  assert "ELi1ELb1ELb0EE" in ENTRY_Q4_FUSED
  assert RESADD_PREFIX.startswith("E_32_32_4_02a9")
