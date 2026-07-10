from tinygrad.llm.route_ops import qk_runtime_specs_attr as _qk_attr

ACTIVATION_FORMATS = _qk_attr("ACTIVATION_FORMATS")
GENERATED_PROVENANCE = _qk_attr("GENERATED_PROVENANCE")
LOWERING_STRATEGIES = _qk_attr("LOWERING_STRATEGIES")
OP_FAMILIES = _qk_attr("OP_FAMILIES")
PHASES = _qk_attr("PHASES")
PROVENANCE = _qk_attr("PROVENANCE")
QUANT_FORMATS = _qk_attr("QUANT_FORMATS")
ROLES = _qk_attr("ROLES")
ActivationQuantSpec = _qk_attr("ActivationQuantSpec")
GeneratedCandidate = _qk_attr("GeneratedCandidate")
QuantizedTensorSpec = _qk_attr("QuantizedTensorSpec")
RuntimeOpSpec = _qk_attr("RuntimeOpSpec")

__all__ = [
  "ACTIVATION_FORMATS", "GENERATED_PROVENANCE", "LOWERING_STRATEGIES", "OP_FAMILIES", "PHASES", "PROVENANCE",
  "QUANT_FORMATS", "ROLES", "ActivationQuantSpec", "GeneratedCandidate", "QuantizedTensorSpec", "RuntimeOpSpec",
]
