from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops

from test.unit.test_nv_q6_region_load_panel1 import _ast


TARGET=Target.parse("NV:CUDA:sm_120")


def _source(candidate:bool) -> str:
  program=to_program(_ast(candidate),CUDARenderer(TARGET))
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def test_q8_panel1_region_renders_18_direct_copies_without_named_loads():
  default,candidate=_source(False),_source(True)
  lines=candidate.splitlines()
  offsets=tuple(4608+i*256 for i in range(18))
  copies=[line.strip() for line in lines if "buf0" in line and "data2_1769472" in line and "=" in line]
  assert len(copies) == 18
  assert all(sum(f"+{offset}" in line for line in copies) == 1 for offset in offsets)
  assert not any("unsigned int val" in line and "data2_1769472" in line and
                 any(f"+{offset}" in line for offset in offsets) for line in lines)
  assert candidate.count("__syncthreads();") == default.count("__syncthreads();") == 4
