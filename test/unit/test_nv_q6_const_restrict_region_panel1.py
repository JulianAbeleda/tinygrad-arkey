import hashlib, re

from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, ParamArg, PostBarrierRegion, RegionLoad

import extra.llm_research.prefill.bench_nv_q6_oracle_true_late_panel1 as gate
import extra.llm_research.prefill.nv_q6_oracle_broad_cta as impl


TARGET=Target.parse("NV:CUDA:sm_120")
GATE12_REGION_SOURCE_SHA="206ebe0ea6214fccfa6c389c19e6b4e6f1d9e0fcc38557495552710555e90017"
Q8_QUALIFIER_RE=re.compile(r"const unsigned int \*__restrict__ (data2_\d+)")


def _ast(region:bool,qualified:bool):
  original=gate.q6_oracle_broad_cta_kernel
  def builder(*args,q8_panel1_schedule="early",**kwargs):
    return impl.q6_oracle_broad_cta_kernel(*args,region_load_q8_panel1=region,
      const_restrict_q8=qualified,**kwargs)
  gate.q6_oracle_broad_cta_kernel=builder
  try: return gate._ast("true_late_tail" if region else "early")
  finally: gate.q6_oracle_broad_cta_kernel=original


def _source(region:bool,qualified:bool) -> str:
  program=to_program(_ast(region,qualified),CUDARenderer(TARGET))
  return next(x.arg for x in program.src if x.op is Ops.SOURCE)


def test_gate13_qualifies_only_q8_and_preserves_region_source():
  region,candidate=_source(True,False),_source(True,True)
  match=Q8_QUALIFIER_RE.search(candidate)
  assert match is not None and candidate.count("__restrict__") == 1
  normalized=candidate.replace(match.group(0),f"unsigned int* {match.group(1)}")
  assert normalized == region
  assert hashlib.sha256(region.encode()).hexdigest() == GATE12_REGION_SOURCE_SHA
  signature=re.search(r'extern "C" __global__ void __launch_bounds__\(256\) \w+\([^)]*\)',candidate).group(0)
  assert f"const unsigned int *__restrict__ {match.group(1)}" in signature
  assert "__launch_bounds__(256)" in signature and "__launch_bounds__(256,1)" not in signature


def test_gate13_has_18_direct_copies_existing_barriers_and_no_q8_store():
  default,candidate=_source(False,False),_source(True,True)
  q8_name=Q8_QUALIFIER_RE.search(candidate).group(1)
  offsets=tuple(4608+i*256 for i in range(18))
  copies=[line.strip() for line in candidate.splitlines() if "buf0" in line and q8_name in line and "=" in line]
  assert len(copies) == 18 and all(sum(f"+{offset}" in line for line in copies)==1 for offset in offsets)
  assert candidate.count("__syncthreads();") == default.count("__syncthreads();") == 4
  assert not any(q8_name in line.split("=",1)[0] for line in candidate.splitlines() if "=" in line)


def test_gate13_annotation_is_exactly_the_region_load_owner():
  ast=_ast(True,True);topo=ast.toposort()
  qualified=[x for x in topo if x.op is Ops.PARAM and isinstance(x.arg,ParamArg) and x.arg.const_restrict]
  regions=[x for x in topo if x.op is Ops.IF and isinstance(x.arg,PostBarrierRegion)]
  markers=[x for x in topo if x.op is Ops.AFTER and isinstance(x.arg,RegionLoad)]
  loads=[x for x in topo if x.op is Ops.LOAD and any(y in markers for y in x.src[1:])]
  assert len(qualified) == len(regions) == len(markers) == 1 and len(loads) == 18
  assert {x.src[0].src[0] for x in loads} == {qualified[0]}
  assert not any(qualified[0] in x.src[0].pointer_base_params() for x in topo if x.op is Ops.STORE)


def test_gate13_default_rejects_incidental_qualification():
  ast=_ast(False,False)
  assert not any(x.op is Ops.PARAM and isinstance(x.arg,ParamArg) and x.arg.const_restrict for x in ast.toposort())
