"""Pins PM4_IB_WRAP_DRAIN and NV_CMDQ_WRAP_DRAIN, the same defect class as KERNARGS_WRAP_DRAIN
(test_kernargs_wrap_audit.py) at two more wrapping BumpAllocator sites (docs/gpu-fault-fix-scope-20260725.md):

  - ops_amd.py pm4_ib_alloc: the PM4 command stream the CP fetches asynchronously as an indirect buffer lives
    inside this allocation. Exposed on every ungraphed AQL dispatch (AMD_AQL=1 or xccs>1), because an
    ungraphed queue never calls .bind() so binded_device stays None and the wrap-prone branch is the only
    branch taken -- not a rare cross-device corner case. Verified live on gfx1100 with AMD_AQL=1 forced.

  - ops_nv.py cmdq_allocator: the command buffer a GPFIFO entry points GPU-side execution at lives inside
    this allocation, same shape as the AMD site above (binded_device is None for every ungraphed dispatch).
    UNTESTED -- no NVIDIA GPU on this machine to reproduce or A/B; ships on by default on the strength of the
    analytic match to the two proven sites, not a measurement.

A detector or guard that silently regresses to off is worse than none, because a quiet run then reads as
evidence. These tests exist so that cannot happen unnoticed.
"""
import pathlib, unittest
import tinygrad.device as D


class TestWrapDrainDefaults(unittest.TestCase):
  def test_pm4_ib_wrap_drain_is_on_by_default(self):
    self.assertEqual(D.PM4_IB_WRAP_DRAIN.value, 1, "the PM4 IB wrap drain must ship enabled")

  def test_nv_cmdq_wrap_drain_is_on_by_default(self):
    self.assertEqual(D.NV_CMDQ_WRAP_DRAIN.value, 1, "the NV cmdq wrap drain must ship enabled")


class TestCallSitesStillHooked(unittest.TestCase):
  def test_pm4_ib_guard_is_wired(self):
    src = (pathlib.Path(D.__file__).parent/"runtime"/"ops_amd.py").read_text()
    self.assertIn("PM4_IB_WRAP_DRAIN", src, "ops_amd.py lost the pm4_ib_alloc wrap drain")
    self.assertIn("dev.pm4_ib_alloc.wraps", src, "the wrap-counter check at the pm4_ib_alloc call site is gone")

  def test_nv_cmdq_guard_is_wired(self):
    src = (pathlib.Path(D.__file__).parent/"runtime"/"ops_nv.py").read_text()
    self.assertIn("NV_CMDQ_WRAP_DRAIN", src, "ops_nv.py lost the cmdq_allocator wrap drain")
    self.assertIn("dev.cmdq_allocator.wraps", src, "the wrap-counter check at the cmdq_allocator call site is gone")


if __name__ == "__main__":
  unittest.main()
