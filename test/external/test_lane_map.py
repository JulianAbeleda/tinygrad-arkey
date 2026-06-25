#!/usr/bin/env python3
import unittest

from tinygrad.codegen.opt.tc import LaneMap, get_amd, get_cuda, metal


def _legacy_remaps(tc):
  local_axes, upcast_axes, reduce_axes = len(tc.get_local_axes()), len(tc.get_upcast_axes()), len(tc.get_reduce_axes())
  fwd_st = [f"l{i}" for i in range(local_axes)] + [f"u{i}" for i in range(upcast_axes)] + [f"r{i}" for i in range(reduce_axes)]
  return [dict(zip(fwd_st, sum(s, ()))) for s in tc.swizzle]


def _legacy_permutes(tc, shape_str):
  ret = [[shape_str.index(remap[ss]) if ss in remap else i for i,ss in enumerate(shape_str)] for remap in _legacy_remaps(tc)]
  return tuple(ret[0]), tuple(ret[1])


def _all_tcs():
  out = []
  for arch in ("sm75", "sm80", "sm89"):
    out += get_cuda(arch)
  for arch in ("gfx1100", "gfx1200", "gfx942", "gfx950"):
    out += get_amd(arch)
  out += metal
  return out


class TestLaneMap(unittest.TestCase):
  def test_all_known_tensor_cores_match_legacy_swizzle_permutes(self):
    for tc in _all_tcs():
      with self.subTest(tc=str(tc), dtype_in=tc.dtype_in, dtype_out=tc.dtype_out):
        self.assertEqual(tc.lane_map.remaps(), _legacy_remaps(tc))
        self.assertEqual(tc.permutes_for_shape_str(tc.base_shape_str()), _legacy_permutes(tc, tc.base_shape_str()))

  def test_validate_rejects_wrong_part_count(self):
    tc = _all_tcs()[0]
    lm = LaneMap((tc.swizzle[0],), len(tc.get_local_axes()), len(tc.get_upcast_axes()), len(tc.get_reduce_axes()),
                 tc.opts, tc.dims, tc.threads, tc.elements_per_thread)
    with self.assertRaises(AssertionError): lm.validate()

  def test_validate_rejects_local_size_mismatch(self):
    tc = _all_tcs()[0]
    bad = ((tc.swizzle[0][0][1:], tc.swizzle[0][1], tc.swizzle[0][2]), tc.swizzle[1])
    lm = LaneMap(bad, len(tc.get_local_axes()), len(tc.get_upcast_axes()), len(tc.get_reduce_axes()),
                 tc.opts, tc.dims, tc.threads, tc.elements_per_thread)
    with self.assertRaises(AssertionError): lm.validate()

  def test_validate_rejects_elements_per_thread_mismatch(self):
    tc = _all_tcs()[0]
    lm = LaneMap(tc.swizzle, len(tc.get_local_axes()), len(tc.get_upcast_axes()), len(tc.get_reduce_axes()),
                 tc.opts, tc.dims, tc.threads, (tc.elements_per_thread[0]*2, tc.elements_per_thread[1], tc.elements_per_thread[2]))
    with self.assertRaises(AssertionError): lm.validate()


if __name__ == "__main__":
  unittest.main()
