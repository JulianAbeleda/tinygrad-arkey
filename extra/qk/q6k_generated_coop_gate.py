#!/usr/bin/env python3
"""TG-P3 gate: prove the spec-driven Q6_K route (extra/qk/q6k_route_spec.py) is a LOSSLESS, route-bound, speed-
equivalent replacement for the shipped hand template (extra/qk/quant/q6_k_gemv_primitive.py).

Runs on AMD (DEV=AMD). For each tracked Q6_K decode shape it:
  * emits the shipped kernel and the generated (spec) kernel over the SAME packed-Q6_K weights + x,
  * asserts the outputs are byte-identical (tobytes()),
  * times both (median of N iters) and reports the generated/shipped ratio,
  * records the distinct program names (route identity).

Writes bench/tg-p3-q6k-generated-coop/{latest.json,summary.md,microgate.json,wd.json,route_policy.json is written
by the BoltBeam emitter step}. Verdict TG_P3_PASS_Q6K_GENERATED_COOP or a precise blocker.
"""
from __future__ import annotations

import pathlib, statistics, time

import numpy as np

from tinygrad import Tensor, dtypes, Device

from extra.qk.quant.q6_k_gemv_primitive import q6k_coop_partial_kernel, q6k_gemv_partial_kernel, parse_opt
from extra.qk.layout import Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES
from extra.qk.q6k_route_spec import spec_for_role, emit_q6k_gemv_kernel

ROOT = pathlib.Path(__file__).resolve().parents[2]

# tracked 8B Q6_K decode shapes + their shipped route parameters (from tinygrad/llm/decode_routes.py Q6_K branches):
#   ffn_down 4096x12288 coop rt4 (parts=1) ; lm_head 151936x4096 coop rt4 (parts=1) ; attn_v 1024x4096 partial parts=4
CASES = [
  {"role": "ffn_down", "rows": 4096, "k": 12288, "parts": 1, "row_tile": 4, "use_coop": True},
  {"role": "lm_head", "rows": 151936, "k": 4096, "parts": 1, "row_tile": 4, "use_coop": True},
  {"role": "attn_v", "rows": 1024, "k": 4096, "parts": 4, "row_tile": 4, "use_coop": False},
]


def _packed_q6k(rows:int, k:int, seed:int) -> Tensor:
  # valid-shaped packed Q6_K bytes (viewed as uint16 halfwords). Content is a fixed non-NaN pattern: d-half (offset
  # 104 halfword = bytes 208..209) set to fp16 1.0 (0x3C00); everything else a deterministic byte ramp. Both emitters
  # run identical ops over identical bytes -> identical output; the gate proves the LOWERING is lossless.
  k_blocks = k // Q6_K_BLOCK_ELEMS
  nbytes = rows * k_blocks * Q6_K_BLOCK_BYTES
  rng = np.random.RandomState(seed)
  b = (rng.randint(0, 256, nbytes, dtype=np.uint8))
  b = b.reshape(rows * k_blocks, Q6_K_BLOCK_BYTES)
  b[:, 208] = 0x00; b[:, 209] = 0x3C          # d = fp16 1.0 per block (no NaN)
  halfs = np.frombuffer(b.tobytes(), dtype=np.uint16).copy()
  return Tensor(halfs, device=Device.DEFAULT).realize()


def _run(kernel_fn, out_shape, halfs, x):
  partials = Tensor.empty(*out_shape, dtype=dtypes.float32, device=Device.DEFAULT)
  out = partials.custom_kernel(halfs, x, fxn=kernel_fn)[0]
  return out.sum(axis=1).reshape(1, 1, out_shape[0]).realize()


def _time(fn, iters=30):
  fn().realize() if hasattr(fn(), "realize") else None
  Device[Device.DEFAULT].synchronize()
  ts = []
  for _ in range(iters):
    t0 = time.perf_counter(); fn(); Device[Device.DEFAULT].synchronize(); ts.append(time.perf_counter() - t0)
  return statistics.median(ts)


def build():
  results, all_identical, ratios = [], True, []
  for c in CASES:
    rows, k, parts = c["rows"], c["k"], c["parts"]
    halfs = _packed_q6k(rows, k, seed=hash((rows, k)) & 0xffff)
    x = Tensor(np.random.RandomState(1).randn(k).astype(np.float16), device=Device.DEFAULT).realize()
    if c["use_coop"]:
      shipped_fn = q6k_coop_partial_kernel(rows, k, c["row_tile"]); ship_shape = (rows, 16)
    else:
      shipped_fn = q6k_gemv_partial_kernel(rows, k, parts, (parse_opt("LOCAL:0:32"),)); ship_shape = (rows, parts)
    spec = spec_for_role(rows, k, role=c["role"], parts=parts, row_tile=c["row_tile"], use_coop=c["use_coop"],
                         opts=(parse_opt("LOCAL:0:32"),) if not c["use_coop"] else ())
    gen_fn = emit_q6k_gemv_kernel(spec)
    gen_shape = (rows, spec.partial_axis_extent)

    a = _run(shipped_fn, ship_shape, halfs, x).numpy()
    b = _run(gen_fn, gen_shape, halfs, x).numpy()
    identical = a.tobytes() == b.tobytes()
    all_identical = all_identical and identical

    t_ship = _time(lambda: _run(q6k_coop_partial_kernel(rows, k, c["row_tile"]) if c["use_coop"]
                                else q6k_gemv_partial_kernel(rows, k, parts, (parse_opt("LOCAL:0:32"),)), ship_shape, halfs, x))
    t_gen = _time(lambda: _run(emit_q6k_gemv_kernel(spec), gen_shape, halfs, x))
    ratio = t_gen / t_ship if t_ship else float("nan")
    ratios.append(ratio)
    results.append({"role": c["role"], "rows": rows, "k": k, "route_family": spec.route_family,
                    "identical_bytes": identical, "shipped_ms": round(t_ship*1e3, 4), "generated_ms": round(t_gen*1e3, 4),
                    "gen_over_shipped": round(ratio, 4), "spec": spec.to_json(), "kernel_name": spec.kernel_name})

  worst = max(ratios) if ratios else float("nan")
  # speed-equivalent tolerance: generated within +5% of shipped (same math + structure, only program name differs)
  speed_ok = worst <= 1.05
  verdict = ("TG_P3_PASS_Q6K_GENERATED_COOP" if all_identical and speed_ok
             else "TG_P3_REFUTE_Q6K_GENERATED_REGRESSION" if all_identical else "TG_P3_BLOCKED_Q6K_IR_CANNOT_REEMIT")
  latest = {"scope": "TG-P3 Q6_K generated coop lossless+speed gate", "verdict": verdict,
            "all_identical_bytes": all_identical, "worst_gen_over_shipped": round(worst, 4),
            "speed_equivalent_tol": 1.05, "cases": results,
            "route_identity": {"generated_names": [r["kernel_name"] for r in results],
                               "shipped_names": ["q6k_coop_partial_*", "q6k_gemv_partial_*"]}}
  return latest


if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("q6k_generated_coop"))
