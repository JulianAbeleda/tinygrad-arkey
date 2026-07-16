"""Research descriptions and emitters for core prefill route data."""
from __future__ import annotations

from extra.qk.mmq_ds4_logical_emitter import packed_fused_candidate
from extra.qk.quant_specs import activation_spec, quant_spec
from extra.qk.runtime_specs import RuntimeOpSpec

def runtime_op_spec(spec, *, activation_format: str = "fp16", lowering_strategy: str = "packed_dequant_dot",
                    device: str = "unknown") -> RuntimeOpSpec:
  qfmt = "Q4_K" if spec.quant == "q4k" else "Q6_K" if spec.quant == "q6k" else "unknown"
  return RuntimeOpSpec("QuantizedLinear", "prefill", spec.role or "unknown", {"M": spec.m, "N": spec.n, "K": spec.k},
                       quant_spec(qfmt).tensor_spec(), activation_spec(activation_format).activation_spec(),
                       lowering_strategy=lowering_strategy, device=device, route_id=f"prefill_{spec.quant}_{spec.route}")

def run_cooperative_q4k(candidate, lin, x_batch, spec, x):
  workload, descriptor = candidate.get("workload", {}), candidate.get("descriptor", {})
  if workload.get("shape") != {"M": spec.m, "N": spec.n, "K": spec.k}: return None
  if {descriptor.get(k) for k in ("m_tile", "n_tile", "k_tile")} - {None, 16, 256}: return None
  fused = packed_fused_candidate(spec.m, spec.n, spec.k, role=spec.role)
  from tinygrad.llm.prefill_routes import prefill_output
  from tinygrad.llm import route_ops
  words = lin.prefill_packed_weight().to(x.device)
  values, scales, sums = route_ops.pack_q8_1_mmq_fused(x_batch.reshape(spec.m, spec.k), fused)
  return prefill_output(route_ops.emit_q4k_q8_mmq_ds4(words, values, scales, sums, fused).reshape(1, spec.m, spec.n))

__all__ = ["runtime_op_spec", "run_cooperative_q4k"]
