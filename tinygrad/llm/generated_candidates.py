from tinygrad.llm.route_ops import qk_generated_candidates_attr as _qk_attr

BUILTIN_GENERATED_CANDIDATES = _qk_attr("BUILTIN_GENERATED_CANDIDATES")
CandidateSelection = _qk_attr("CandidateSelection")
GeneratedCandidateRegistry = _qk_attr("GeneratedCandidateRegistry")
builtin_registry = _qk_attr("builtin_registry")
select_generated_candidate = _qk_attr("select_generated_candidate")

__all__ = [
  "BUILTIN_GENERATED_CANDIDATES", "CandidateSelection", "GeneratedCandidateRegistry", "builtin_registry",
  "select_generated_candidate",
]
