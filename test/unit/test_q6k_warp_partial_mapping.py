import unittest

from extra.llm_research.decode.q6k_q8_warp_partial_microgate import ownership_coordinates, flat_ownership_coordinates

class TestQ6KWarpPartialMapping(unittest.TestCase):
  def test_exact_q6_block_coverage_and_partial_ownership(self):
    rows = ownership_coordinates()
    self.assertEqual(len(rows), 1024)  # one DP4A chunk per packed four values
    self.assertEqual({r[0] for r in rows}, {0,1,2,3})
    self.assertEqual({r[2] for r in rows}, set(range(16)))
    self.assertEqual({r[3] for r in rows}, set(range(16)))
    elems = {(blk*256 + grp*16 + pos4*4 + i) for _,_,blk,grp,pos4 in rows for i in range(4)}
    self.assertEqual(elems, set(range(4096)))

  def test_each_warp_produces_exactly_one_established_partial(self):
    rows = ownership_coordinates()
    for warp in range(4):
      self.assertEqual(len([r for r in rows if r[0] == warp]), 256)
      self.assertEqual({r[2] for r in rows if r[0] == warp}, set(range(warp*4, warp*4+4)))

  def test_flat_local_spelling_is_exactly_the_same_ownership(self):
    self.assertEqual(ownership_coordinates(), flat_ownership_coordinates())
