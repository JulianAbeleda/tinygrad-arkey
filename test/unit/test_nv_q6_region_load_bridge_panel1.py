from tinygrad.uop.ops import Ops, PostBarrierRegion, RegionLoadBridge

import extra.llm_research.prefill.bench_nv_q6_oracle_true_late_panel1 as gate
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import q6_oracle_broad_cta_kernel


def _ast(bridge:bool):
  original=gate.q6_oracle_broad_cta_kernel
  def builder(*args,q8_panel1_schedule="early",**kwargs):
    candidate=q8_panel1_schedule != "early"
    kwargs["prefetch_second_panel"]=not candidate
    return q6_oracle_broad_cta_kernel(*args,region_load_bridge_q8_panel1=candidate,**kwargs)
  gate.q6_oracle_broad_cta_kernel=builder
  try: return gate._ast("true_late_tail" if bridge else "early")
  finally: gate.q6_oracle_broad_cta_kernel=original


def test_q8_panel1_bridge_has_exact_region_contract():
  topo=_ast(True).toposort()
  regions=[x for x in topo if x.op is Ops.IF and isinstance(x.arg,PostBarrierRegion)]
  markers=[x for x in topo if x.op is Ops.AFTER and isinstance(x.arg,RegionLoadBridge)]
  loads=[x for x in topo if x.op is Ops.LOAD and any(s in markers for s in x.src[1:])]
  stores=[x for x in topo if x.op is Ops.STORE and len(x.src)==2 and x.src[1] in loads]
  ends=[x for x in topo if x.op is Ops.ENDIF and isinstance(x.arg,PostBarrierRegion)]
  assert len(regions)==len(markers)==len(ends)==1 and len(loads)==len(stores)==18
  assert markers[0].src==(regions[0],) and all(markers[0] not in x.src[0].backward_slice_with_self for x in loads)
  assert ends[0].src[0] is regions[0] and set(ends[0].src[1:])==set(stores)


def test_q8_panel1_anchor_has_no_bridge_region():
  topo=_ast(False).toposort()
  assert not any(isinstance(x.arg,(PostBarrierRegion,RegionLoadBridge)) for x in topo)
