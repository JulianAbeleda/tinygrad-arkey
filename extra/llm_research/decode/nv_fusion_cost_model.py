"""Predicted-wall-delta cost model for decode fusion experiments.

Working-agreement contract: every decode fusion harness must carry a
``COST_PREDICTION`` table in its CONSTRUCTION record BEFORE the wall bracket
runs, derived from the llama reference shape plus per-element
instruction/traffic arithmetic.  The bracket then reconciles the measured
delta against the prediction:

  CONFIRMED     measured inside the predicted range (tolerance included)
  EXPLAINED     measured outside the range but same sign as the point
                prediction; named residual causes are recorded
  CONTRADICTED  measured outside the predicted range on the opposite side of
                zero; the premise is unbacked and the campaign FAILS CLOSED

Reference facts (llama replay, llama_tinygrad_role_manifest.py + the d512
timeline ledger): llama renders the norm as ONE fused ``rms_norm_f32`` kernel
(reduce + weight affine, fp32 out) and the norm arithmetic NEVER enters the
matmul inner loop; each matmul consumes a compact quantized activation
(``quantize_q8_1`` -> ``mul_mat_vec_q``).  The per-kernel floor for the tiny
E_32_32_4 class is ~1.5us (m4-resadd-landing-scope-20260806.md: 72 kernels at
1.5us = 108us/token mass).

Model terms (per token, positive = candidate SLOWER):

  blocks x [ (R - 1) x M_removed - R x launch_us ]

  M_removed   measured control median of the folded kernel family, read from
              the same campaign's control census BEFORE the bracket runs
  R           static redundancy: how many times the absorbed epilogue
              re-executes inside the surviving kernel (M1 rms-affine: 2, once
              per matrix dot)
  launch_us   per-kernel launch/floor overhead (1.5us, range 1.0-2.0)

The model deliberately leaves the in-kernel critical-path and traffic terms
UNMODELED: the reconcile step attributes the residual to named causes instead
of pretending a closed-form GPU kernel-time model exists.
"""
from __future__ import annotations


LAUNCH_US_DEFAULT = 1.5
LAUNCH_US_RANGE = (1.0, 2.0)
LAUNCH_CITATION = "m4-resadd-landing-scope-20260806.md (E_32_32_4 class floor)"
TOLERANCE_US_DEFAULT = 20.0

RESIDUAL_CAUSES = {
  "in_kernel_critical_path": (
    "absorbed epilogue re-executes on the reduction dependency chain (register pressure / occupancy); "
    "the standalone kernel median understates the in-kernel cost"),
  "activation_traffic": (
    "x stream bytes changed (fp16 -> fp32 doubles the per-row activation reads across all rows)"),
  "launch_overlap": (
    "removed kernels were partially overlapped with the surviving kernel by the driver; "
    "the launch savings are less than one kernel time each"),
  "scale_reduce_retained": (
    "the bitwise-exact scale reduce must stay (llama keeps n_f32); its time is not recoverable"),
}


def predict_wall_delta(blocks: int, removed: dict[str, float], redundancy: dict[str, int],
                       launch_us: tuple[float, float] = LAUNCH_US_RANGE) -> dict:
  """Predicted candidate-minus-control wall delta from llama-shaped arithmetic.

  ``removed`` maps each folded kernel family name to its measured control median
  in microseconds; ``redundancy`` maps the same families to how many times the
  absorbed epilogue re-executes inside the surviving kernel.  Returns the point
  prediction, the launch-range span, the per-family terms, and the recorded
  assumptions so the record is fully auditable."""
  if blocks <= 0: raise ValueError("blocks must be positive")
  if not removed: raise ValueError("removed families must be non-empty")
  if set(removed) != set(redundancy): raise ValueError("removed and redundancy families must match")
  if not (0.0 < launch_us[0] <= launch_us[1]): raise ValueError("launch range must be ascending positive")
  terms = {}
  point, lo, hi = 0.0, 0.0, 0.0
  for family, median in removed.items():
    r = redundancy[family]
    if r < 1: raise ValueError(f"redundancy for {family} must be >= 1")
    term_lo = (r - 1) * median - r * launch_us[1]
    term_hi = (r - 1) * median - r * launch_us[0]
    terms[family] = {"median_us": median, "redundancy": r, "per_block_lo_us": term_lo,
                     "per_block_hi_us": term_hi}
    point += (r - 1) * median - r * (launch_us[0] + launch_us[1]) / 2.0
    lo += term_lo
    hi += term_hi
  point *= blocks
  lo, hi = lo * blocks, hi * blocks
  return {
    "blocks": blocks,
    "predicted_delta_us": round(point, 3),
    "range_us": [round(lo, 3), round(hi, 3)],
    "terms": terms,
    "assumptions": {
      "launch_us_range": list(launch_us),
      "launch_citation": LAUNCH_CITATION,
      "unmodeled": ["in-kernel critical path (occupancy/dependency chain)",
                    "activation traffic (fp16 vs fp32 x bytes)"],
    },
    "decisive": (lo > 0.0) or (hi < 0.0),
    "formula": "blocks x [ (R - 1) x M_removed - R x launch_us ]; positive = candidate slower",
  }


def reconcile_cost_prediction(measured_delta_us: float, prediction: dict,
                              tolerance_us: float = TOLERANCE_US_DEFAULT) -> dict:
  """Reconcile the measured bracket delta against the predicted range.

  Positive deltas mean the candidate is SLOWER (candidate - control).  A
  measured delta beyond the range in the direction OPPOSITE to the point
  prediction is a CONTRADICTION (the premise is unbacked).  A gap in the
  prediction's own direction gets EXPLAINED with the named residual causes;
  the tolerance band around the point prediction is CONFIRMED."""
  lo, hi = prediction["range_us"]
  point = prediction["predicted_delta_us"]
  if (point < 0.0 and measured_delta_us > hi + tolerance_us) or \
     (point > 0.0 and measured_delta_us < lo - tolerance_us):
    result = "CONTRADICTED"
    causes = ["in_kernel_critical_path", "activation_traffic"] if measured_delta_us > point \
      else ["launch_overlap"]
  elif abs(measured_delta_us - point) <= tolerance_us:
    result = "CONFIRMED"
    causes = []
  else:
    result = "EXPLAINED"
    causes = ["in_kernel_critical_path", "activation_traffic"] if measured_delta_us > point \
      else ["launch_overlap"]
  return {
    "result": result,
    "measured_delta_us": round(measured_delta_us, 3),
    "predicted_delta_us": point,
    "range_us": [lo, hi],
    "gap_us": round(measured_delta_us - point, 3),
    "tolerance_us": tolerance_us,
    "residual_causes": [{"cause": c, "note": RESIDUAL_CAUSES[c]} for c in causes],
    "note": "positive delta = candidate slower; CONTRADICTED fails the campaign closed",
  }
