from tinygrad.llm.route_ops import qk_quant_specs_attr as _qk_attr

ACTIVATION_FORMAT_SPECS = _qk_attr("ACTIVATION_FORMAT_SPECS")
QUANT_FORMAT_SPECS = _qk_attr("QUANT_FORMAT_SPECS")
ActivationFormatSpec = _qk_attr("ActivationFormatSpec")
QuantFormatSpec = _qk_attr("QuantFormatSpec")
activation_spec = _qk_attr("activation_spec")
quant_spec = _qk_attr("quant_spec")

__all__ = [
  "ACTIVATION_FORMAT_SPECS", "QUANT_FORMAT_SPECS", "ActivationFormatSpec", "QuantFormatSpec", "activation_spec",
  "quant_spec",
]
