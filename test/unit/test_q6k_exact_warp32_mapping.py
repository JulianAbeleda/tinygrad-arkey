import unittest

from extra.llm_research.decode.q6k_exact_warp32_microgate import K, K_BLOCKS, ownership_coordinates


class TestQ6KExactWarp32Mapping(unittest.TestCase):
  def test_every_q6_element_is_owned_once(self):
    coords=ownership_coordinates()
    self.assertEqual(len(coords),K)
    self.assertEqual({lane for lane,_,_,_ in coords},set(range(32)))
    elems=[blk*256+grp*16+pos for _,blk,grp,pos in coords]
    self.assertEqual(len(elems),len(set(elems)))
    self.assertEqual(set(elems),set(range(K)))

  def test_each_pair_step_is_adjacent_and_warp_coalesced(self):
    coords=ownership_coordinates()
    for blk in range(K_BLOCKS):
      for pair in range(8):
        rows=[(lane,grp,pos) for lane,b,g,p in coords if b==blk and g//2==pair for grp,pos in [(g,p)]]
        self.assertEqual(len(rows),32)
        self.assertEqual({grp for _,grp,_ in rows},{2*pair,2*pair+1})
        self.assertEqual({pos for _,_,pos in rows},set(range(16)))

  def test_mapping_is_not_closed_flat_four_warp_shape(self):
    coords=ownership_coordinates()
    self.assertEqual(max(lane for lane,_,_,_ in coords)+1,32)
    self.assertNotEqual(max(lane for lane,_,_,_ in coords)+1,128)


if __name__ == "__main__": unittest.main()
