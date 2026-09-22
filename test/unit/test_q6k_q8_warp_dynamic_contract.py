from extra.llm_research.decode.q6k_q8_warp_partial_microgate import ownership_coordinates


def test_dynamic_candidate_retains_prior_flat_four_warp_ownership():
  rows=ownership_coordinates()
  assert len(rows)==1024
  assert {warp for warp,_,_,_,_ in rows}=={0,1,2,3}
  assert {block for _,_,block,_,_ in rows}==set(range(16))
  elems={block*256+group*16+pos4*4+i for _,_,block,group,pos4 in rows for i in range(4)}
  assert elems==set(range(4096))
