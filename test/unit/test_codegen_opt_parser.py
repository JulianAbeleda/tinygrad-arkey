import pytest

from tinygrad.codegen.opt import Opt, OptOps, parse_opt


def test_parse_opt_accepts_single_and_axis_forms():
  assert parse_opt("UPCAST") == Opt(OptOps.UPCAST)
  assert parse_opt("local:1:16") == Opt(OptOps.LOCAL, 1, 16)


@pytest.mark.parametrize("spec", ["", "UPCAST:0", "UPCAST:0:1:2", "NOT_AN_OPT", "UPCAST:x:4"])
def test_parse_opt_rejects_malformed_or_unknown_specs(spec):
  with pytest.raises((ValueError, KeyError)):
    parse_opt(spec)


def test_quant_modules_reuse_core_parser_and_route_shims_are_gone():
  from extra.llm_research.quant.q4_k_gemv_primitive import parse_opt as q4_parse
  from extra.llm_research.quant.q6_k_gemv_primitive import parse_opt as q6_parse
  assert q4_parse is parse_opt
  assert q6_parse is parse_opt
  assert not __import__("pathlib").Path("tinygrad/llm/route_ops.py").exists()
