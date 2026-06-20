#!/usr/bin/env python3
"""Gate 3 — PREFILL_GRAPH_GEMM fallback audit (default-on readiness).

Structural test of `route_pf16_graph_gemm`: valid supported shapes route (return a Tensor), and every
unsupported / ineligible case returns None WITHOUT misrouting or allocating an output. Also confirms the
model-level flag gate keeps default behavior unchanged when PREFILL_GRAPH_GEMM is absent.

No model load needed — uses mock `lin` objects (the route reads `_pf16_w`, `bias`, `_prefill_graph_role`).

Run: DEV=AMD PREFILL_GRAPH_GEMM=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_fallback_audit.py
"""
from __future__ import annotations
import json, os, pathlib


class MockLin:
  def __init__(self, w=None, bias=None, role=None):
    if w is not None: self._pf16_w = w
    self.bias = bias
    if role is not None: self._prefill_graph_role = role


def main() -> int:
  from tinygrad import Tensor, dtypes
  from tinygrad.helpers import getenv  # cached (lru_cache); the route reads PREFILL_GRAPH_GEMM_ROLES via it
  from extra.qk_prefill_graph_gemm_route import route_pf16_graph_gemm
  def set_roles(v):  # mutate env AND clear getenv's cache so the route sees the new value
    if v is None: os.environ.pop("PREFILL_GRAPH_GEMM_ROLES", None)
    else: os.environ["PREFILL_GRAPH_GEMM_ROLES"] = v
    getenv.cache_clear()

  GATEUP = (12288, 4096)  # (out_f, in_f) gate/up
  DOWN = (4096, 12288)    # down
  def w_of(out_f, in_f): return Tensor.randn(out_f, in_f).cast(dtypes.float16).contiguous().realize()
  def x_of(t, in_f): return Tensor.randn(t, in_f).cast(dtypes.float16).contiguous().realize()

  cases = []
  def case(name, expect, fn):
    set_roles(None)
    try:
      got = fn()
      routed = got is not None
      if routed: got.realize()  # supported cases must actually run
      ok = (routed == (expect == "tensor"))
      cases.append({"case": name, "expect": expect, "got": "tensor" if routed else "None", "pass": ok, "err": None})
    except Exception as ex:
      cases.append({"case": name, "expect": expect, "got": "EXCEPTION", "pass": False, "err": repr(ex)[:140]})

  # 1-2: valid supported shapes route
  case("valid_gateup_T512", "tensor", lambda: route_pf16_graph_gemm(MockLin(w_of(*GATEUP)), x_of(512, GATEUP[1])))
  case("valid_down_T512",   "tensor", lambda: route_pf16_graph_gemm(MockLin(w_of(*DOWN)), x_of(512, DOWN[1])))
  # 3: unsupported T (!=512) -> None, no exception
  case("unsupported_T256",  "none",   lambda: route_pf16_graph_gemm(MockLin(w_of(*GATEUP)), x_of(256, GATEUP[1])))
  # 4: unsupported output shape not a multiple of bn(128) -> _kernel None
  case("unsupported_outf_nonmultiple", "none", lambda: route_pf16_graph_gemm(MockLin(w_of(12300, 4096)), x_of(512, 4096)))
  # 4b: unsupported K not a multiple of bk(32)
  case("unsupported_inf_nonmultiple",  "none", lambda: route_pf16_graph_gemm(MockLin(w_of(12288, 4100)), x_of(512, 4100)))
  # 5: missing realized _pf16_w -> None
  case("missing_pf16_w",    "none",   lambda: route_pf16_graph_gemm(MockLin(w=None), x_of(512, 4096)))
  # 6: bias present -> None
  case("bias_present",      "none",   lambda: route_pf16_graph_gemm(MockLin(w_of(*GATEUP), bias=Tensor.zeros(12288)), x_of(512, GATEUP[1])))

  # 7: role filter EXCLUDES this role -> None
  def role_excluded():
    set_roles("ffn_down")
    return route_pf16_graph_gemm(MockLin(w_of(*GATEUP), role="ffn_gate"), x_of(512, GATEUP[1]))
  # 8: role filter INCLUDES this role -> tensor
  def role_included():
    set_roles("ffn_gate,ffn_up")
    return route_pf16_graph_gemm(MockLin(w_of(*GATEUP), role="ffn_gate"), x_of(512, GATEUP[1]))
  # 8b: role filter set but lin has no role -> None (conservative)
  def role_missing():
    set_roles("ffn_gate")
    return route_pf16_graph_gemm(MockLin(w_of(*GATEUP), role=None), x_of(512, GATEUP[1]))
  for nm, exp, fn in [("role_filter_excludes", "none", role_excluded), ("role_filter_includes", "tensor", role_included),
                      ("role_filter_set_but_role_missing", "none", role_missing)]:
    try:
      got = fn(); routed = got is not None
      if routed: got.realize()
      cases.append({"case": nm, "expect": exp, "got": "tensor" if routed else "None", "pass": routed == (exp == "tensor"), "err": None})
    except Exception as ex:
      cases.append({"case": nm, "expect": exp, "got": "EXCEPTION", "pass": False, "err": repr(ex)[:140]})
    finally: set_roles(None)

  # 9: default behavior unchanged when flag absent — the route is gated in model._pf16 by PREFILL_GRAPH_GEMM.
  # Structural: confirm the only call site is flag-gated.
  src = pathlib.Path("tinygrad/llm/model.py").read_text()
  flag_gated = "if PREFILL_GRAPH_GEMM and w is not None:" in src and "route_pf16_graph_gemm(lin, x)" in src
  cases.append({"case": "flag_gated_call_site", "expect": "gated", "got": "gated" if flag_gated else "UNGATED",
                "pass": flag_gated, "err": None})

  npass = sum(c["pass"] for c in cases); n = len(cases)
  no_exc_on_unsupported = all(c["got"] != "EXCEPTION" for c in cases if c["expect"] == "none")
  verdict = ("PASS_PREFILL_GRAPH_GEMM_FALLBACK_AUDIT" if npass == n and no_exc_on_unsupported
             else "BLOCKED_PREFILL_GRAPH_GEMM_FALLBACK_AUDIT")
  result = {"date": "2026-06-20", "gate": 3, "schema": "prefill_graph_gemm_fallback_audit_v1",
            "cases": cases, "passed": npass, "total": n, "no_exception_on_unsupported": no_exc_on_unsupported,
            "default_unchanged_when_flag_absent": flag_gated, "verdict": verdict}
  out = pathlib.Path("bench/amd-broad-backend-roadmap"); out.mkdir(parents=True, exist_ok=True)
  (out / "prefill_graph_gemm_fallback_audit_result.json").write_text(json.dumps(result, indent=2) + "\n")
  for c in cases: print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['case']:34} expect={c['expect']:7} got={c['got']}" + (f"  {c['err']}" if c['err'] else ""))
  print(f"\n{verdict}  ({npass}/{n})")
  return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
  raise SystemExit(main())
