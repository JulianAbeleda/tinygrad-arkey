from tinygrad.llm.cooperative_mmq_gate import (
  COOPERATIVE_MMQ_ROUTE, ROLLBACK_ROUTE, admit_cooperative_mmq, canonical_candidate_identity,
)


def candidate():
  return {"descriptor": {"quant": "Q4_K", "m_tile": 16, "n_tile": 16, "k_tile": 256},
          "provenance": "research", "rollback_route": ROLLBACK_ROUTE}


def evidence(c):
  return {name: {"passed": True} for name in ("compile", "correctness", "guard", "resources",
                                                "dynamic_owner_compile", "dynamic_owner_correctness",
                                                "dynamic_owner_instruction")} | {
    "candidate_identity": canonical_candidate_identity(c), "fallback_used": False,
  }


def test_cooperative_mmq_is_default_off_and_rolls_back():
  decision = admit_cooperative_mmq(candidate=candidate(), evidence=evidence(candidate()))
  assert decision.status == "default_off" and decision.route == ROLLBACK_ROUTE
  assert decision.rollback_route == ROLLBACK_ROUTE


def test_complete_identity_bound_candidate_can_be_admitted_explicitly():
  c = candidate()
  decision = admit_cooperative_mmq(candidate=c, evidence=evidence(c), enabled=True)
  assert decision.admitted and decision.route == COOPERATIVE_MMQ_ROUTE
  assert decision.candidate_identity == canonical_candidate_identity(c)


def test_identity_or_correctness_failure_is_fail_closed():
  c = candidate(); ev = evidence(c)
  ev["candidate_identity"] = "tampered"
  ev["correctness"] = {"passed": False}
  decision = admit_cooperative_mmq(candidate=c, evidence=ev, enabled=True)
  assert decision.status == "blocked" and decision.route == ROLLBACK_ROUTE
  assert "candidate identity mismatch" in decision.blockers
  assert "full-output correctness gate not passed" in decision.blockers


def test_dynamic_owner_evidence_is_required_before_execution():
  c = candidate(); ev = evidence(c)
  del ev["dynamic_owner_instruction"]
  decision = admit_cooperative_mmq(candidate=c, evidence=ev, enabled=True)
  assert not decision.admitted and decision.route == ROLLBACK_ROUTE
  assert "dynamic-owner instruction evidence not passed" in decision.blockers


def test_missing_candidate_and_fallback_claim_are_blocked():
  missing = admit_cooperative_mmq(candidate=None, evidence={}, enabled=True)
  assert missing.status == "blocked" and missing.route == ROLLBACK_ROUTE
  c = candidate(); ev = evidence(c); ev["fallback_used"] = True
  blocked = admit_cooperative_mmq(candidate=c, evidence=ev, enabled=True)
  assert blocked.status == "blocked" and "fallback_used must be explicitly false" in blocked.blockers
