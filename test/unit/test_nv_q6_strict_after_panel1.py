from tinygrad.uop.ops import LoadSchedule, Ops, dtypes

import extra.llm_research.prefill.bench_nv_q6_oracle_true_late_panel1 as gate
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import q6_oracle_broad_cta_kernel


def _ast(scheduled:bool):
  original=gate.q6_oracle_broad_cta_kernel
  def compatibility_builder(*args, q8_panel1_schedule="early", **kwargs):
    return q6_oracle_broad_cta_kernel(*args, schedule_after_q8_panel1=(q8_panel1_schedule != "early"), **kwargs)
  gate.q6_oracle_broad_cta_kernel=compatibility_builder
  try: return gate._ast("true_late_tail" if scheduled else "early")
  finally: gate.q6_oracle_broad_cta_kernel=original


def _load_token(load):
  return next((x for x in load.src[1:] if x.op is Ops.AFTER and isinstance(x.arg, LoadSchedule)), None)


def test_q8_panel1_has_18_opaque_scheduled_loads():
  ast=_ast(True)
  loads=[x for x in ast.toposort() if x.op is Ops.LOAD and _load_token(x) is not None]
  assert len(loads) == 18
  assert all(x.dtype is dtypes.uint for x in loads)
  tokens={_load_token(x) for x in loads}
  assert len(tokens) == 1
  token=next(iter(tokens))
  assert token is not None and len(token.src) == 1 and token.src[0].dtype.scalar() in dtypes.floats
  assert all(token not in x.src[0].backward_slice_with_self for x in loads)


def test_default_anchor_has_no_scheduled_load():
  assert not any(x.op is Ops.AFTER and isinstance(x.arg, LoadSchedule) for x in _ast(False).toposort())
