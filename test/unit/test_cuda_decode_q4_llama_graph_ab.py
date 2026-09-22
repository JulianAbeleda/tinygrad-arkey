from scratchpad.cuda_decode_q4_llama_graph_ab import select_attention_q

def test_attention_q_selection_survives_graph_boundaries():
  groups=[[7,21],[3,17,31,45,59],[23,37,51,65,79,93,107],[9]]
  seen=0; selected=[]; ordinals=[]
  for group in groups:
    got,mapping=select_attention_q(group,seen); selected += got
    ordinals += [mapping[x] for x in got]; seen += len(group)
  assert ordinals == list(range(0,16,2))
  assert selected == [7,3,31,59,37,65,93,9]
