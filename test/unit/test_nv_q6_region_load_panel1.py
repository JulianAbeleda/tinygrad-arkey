from tinygrad.uop.ops import Ops, PostBarrierRegion, RegionLoad, dtypes

import extra.llm_research.prefill.bench_nv_q6_oracle_true_late_panel1 as gate
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import q6_oracle_broad_cta_kernel


def _ast(region_load:bool):
  original=gate.q6_oracle_broad_cta_kernel
  def compatibility_builder(*args, q8_panel1_schedule="early", **kwargs):
    return q6_oracle_broad_cta_kernel(*args, region_load_q8_panel1=(q8_panel1_schedule != "early"), **kwargs)
  gate.q6_oracle_broad_cta_kernel=compatibility_builder
  try: return gate._ast("true_late_tail" if region_load else "early")
  finally: gate.q6_oracle_broad_cta_kernel=original


def _marker(load):
  return next((x for x in load.src[1:] if x.op is Ops.AFTER and isinstance(x.arg,RegionLoad)),None)


def test_q8_panel1_is_one_region_with_18_loads_and_publications():
  ast=_ast(True); topo=ast.toposort()
  regions=[x for x in topo if x.op is Ops.IF and isinstance(x.arg,PostBarrierRegion)]
  markers=[x for x in topo if x.op is Ops.AFTER and isinstance(x.arg,RegionLoad)]
  loads=[x for x in topo if x.op is Ops.LOAD and _marker(x) is not None]
  assert len(regions) == 1 and len(markers) == 1 and len(loads) == 18
  assert markers[0].src == (regions[0],)
  assert all(x.dtype is dtypes.uint and markers[0] not in x.src[0].backward_slice_with_self for x in loads)
  owners={x.src[0].src[0] for x in loads}
  assert len(owners) == 1 and next(iter(owners)).op is Ops.PARAM
  publications=[x for x in topo if x.op is Ops.STORE and len(x.src) >= 2 and x.src[1] in loads]
  assert len(publications) == 18
  ends=[x for x in topo if x.op is Ops.ENDIF and isinstance(x.arg,PostBarrierRegion)]
  assert len(ends) == 1 and ends[0].src[0] is regions[0] and len(ends[0].src) == 19
  assert set(ends[0].src[1:]) == set(publications)


def test_default_anchor_has_no_region_load():
  topo=_ast(False).toposort()
  assert not any((x.op is Ops.AFTER and isinstance(x.arg,RegionLoad)) or
                 (x.op in {Ops.IF,Ops.ENDIF} and isinstance(x.arg,PostBarrierRegion)) for x in topo)
