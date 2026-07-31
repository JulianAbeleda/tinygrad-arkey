#!/usr/bin/env python3
"""M1d: confirm or kill the C-fragment overcount hypothesis, compile-only, no GPU.

Reuses scratchpad/m1c_isolate_cause.py's payload construction (same QUANT/ROLE/SHAPE/GEOMETRY,
same PackedWmmaRoute-shaped local row, same candidate_payload/derive_packed_weight_candidate/
full_kernel_workload machinery) but stops before any device is opened or any kernel is dispatched.

Technique (proven in test/unit/test_warp_shfl_xor_renderer_lowering.py): build the Tensor graph's
AST via .schedule_linear(), then call `to_program(ast, renderer)` directly with a renderer built
from `Target.parse(...)` -- this renders (and, for AMD, cross-compiles) without ever calling
Device[...], so it works for AMD on a machine with no AMD GPU. Uses `sys.settrace` to capture the
exact local variables at the specific lines in tinygrad/codegen/opt/postrange.py that the M1c doc's
hypothesis is about (tc_upcast_axes, c_axes, accumulator_total, the WMMA node's declared vector
width) -- no production file is edited; the trace only reads frame locals.
"""
from __future__ import annotations
import sys, copy, json, math
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target, Context
from tinygrad.codegen import to_program
from tinygrad.renderer.cstyle import MetalRenderer, HIPRenderer
from tinygrad.uop.ops import Ops

from extra.llm_research.prefill.packed_wmma_correctness_canary import candidate_payload
from extra.llm_research.runtime_specs import derive_packed_weight_candidate, full_kernel_workload
from extra.llm_research.prefill.current_prefill_execution_adapter import admit_current_prefill
from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.codegen.opt import Opt, OptOps

QUANT, ROLE, SHAPE = "Q4_K", "ffn_gate_up", (512, 12288, 4096)
PROFILE = "qwen3_8b_q4k_m_gfx1100"
GEOMETRY = (256, 64, 32, 8, 1, 1)  # tm,tn,tk,wm,wn,bc -- AMD's real ffn_gate_up geometry, reused verbatim (M1c).

TARGETS = {
  "METAL": ("METAL:METAL:Apple9", lambda t: MetalRenderer(t)),
  "AMD": ("AMD:HIP:gfx1100", lambda t: HIPRenderer(t)),
}


def _payload_for_local_row(profile: str, row: PackedWmmaRoute) -> dict:
  payload = copy.deepcopy(candidate_payload(profile, row.role))
  if tuple(payload["workload"]["shape"][key] for key in ("m", "n", "k")) != row.shape:
    raise ValueError(f"oracle workload does not match row {row}")
  g, schedule = row.geom, payload["schedule"]
  schedule["tile"] = {"m": g["tm"], "n": g["tn"], "k": g["tk"]}
  schedule["waves"] = {"m": g["wm"], "n": g["wn"]}
  schedule["threads"] = g["wm"] * g["wn"] * 32
  a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80
  schedule["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  schedule["lds"]["strides"] = {"a": 80, "b": 80}
  schedule["pipeline"]["buffer_count"] = g["bc"]
  return payload


def _packed_half_carrier(src, transform, n: int, k: int):
  from tinygrad import dtypes as dt
  blocks, halfwords = n * k // transform.block_elems, transform.block_bytes // 2
  return src.bitcast(dt.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128 - halfwords))) \
    .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(n, k).bitcast(dt.half)


def _build_program(final_payload: dict, canonical_identity: str, context, device: str, renderer):
  """Line-for-line the AST-construction half of compile_current_prefill_program, minus compile_linear
  (which would call Device[device].renderer -- the exact thing we must not do for AMD here).

  IMPORTANT: `to_program` (which triggers Kernel.apply_opts, where the warmstart-table lookup that
  attaches `candidate_context` onto the ast happens) MUST run *inside* the `warmstart_candidate_state`
  context manager, exactly as `compile_current_prefill_program` does by nesting `compile_linear(...)`
  inside its `with warmstart_candidate_state(...), Context(DEV=device):` block. A first version of this
  script called to_program() after that `with` block had already exited -- the global candidate-context
  table was gone by then, so the precontract path never engaged and a plain generic WMMA was rendered
  instead. That was a bug in the harness, not evidence about the hypothesis; fixed here."""
  m, n, k = full_kernel_workload(final_payload).shape
  packed_dtype = context.packed_weight.storage_dtype if context.packed_weight is not None else None
  key = warmstart_key({m, n}, k, packed_dtype)
  opts = {key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}
  with warmstart_candidate_state(opts, {key: context}), Context(DEV=device):
    a = Tensor.empty(m, k, dtype=dtypes.half, device=device)
    transform = context.packed_weight
    if transform is None:
      b = Tensor.empty(n, k, dtype=dtypes.half, device=device)
    else:
      packed = Tensor.empty(transform.packed_bytes // transform.storage_width, dtype=transform.storage_dtype, device=device)
      b = _packed_half_carrier(packed, transform, n, k)
    linear = (a @ b.transpose()).schedule_linear()
    calls = [c for c in linear.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
    if len(calls) != 1: raise ValueError(f"expected exactly one SINK CALL, found {len(calls)} (of {len(linear.src)} total)")
    ast = calls[0].src[0]
    return to_program(ast, renderer)


# ---- tracer: capture postrange.py locals at the exact lines the hypothesis is about, no file edits ----
import tinygrad.codegen.opt.postrange as postrange_mod
POSTRANGE_FILE = postrange_mod.__file__

TARGET_LINES = {420, 421, 435, 448, 456, 464, 466, 473, 494, 497, 498, 499, 505, 508, 522, 523}
captured: list[dict] = []


def _tracer(frame, event, arg):
  if frame.f_code.co_filename != POSTRANGE_FILE:
    return _tracer
  if event == "line" and frame.f_lineno in TARGET_LINES:
    loc = frame.f_locals
    snap = {"line": frame.f_lineno}
    if "tc" in loc:
      tc = loc["tc"]
      snap["tc_dims"] = getattr(tc, "dims", None)
      snap["tc_elements_per_thread"] = getattr(tc, "elements_per_thread", None)
    if "tc_upcast_axes" in loc and isinstance(loc["tc_upcast_axes"], tuple) and len(loc["tc_upcast_axes"]) == 3:
      try: snap["tc_upcast_axes_lens"] = tuple(len(v) for v in loc["tc_upcast_axes"])
      except TypeError: pass
    if "c_axes" in loc:
      try: snap["len_c_axes"] = len(loc["c_axes"])
      except TypeError: pass
    if "factors" in loc:
      f = loc["factors"]
      snap["subtiles_m"] = getattr(f, "subtiles_m", None)
      snap["subtiles_n"] = getattr(f, "subtiles_n", None)
    if "accumulator_total" in loc:
      snap["accumulator_total"] = loc["accumulator_total"]
    if "register_mode" in loc:
      snap["register_mode"] = loc["register_mode"]
    if "candidate_pipeline" in loc:
      snap["candidate_pipeline_is_none"] = loc["candidate_pipeline"] is None
    if "candidate_geometry" in loc:
      snap["candidate_geometry_is_none"] = loc["candidate_geometry"] is None
    if "candidate_axes" in loc:
      snap["candidate_axes_is_none"] = loc["candidate_axes"] is None
    if "candidate_contract" in loc:
      snap["candidate_contract_is_none"] = loc["candidate_contract"] is None
    captured.append(snap)
  return _tracer


def render_one(device: str):
  target_str, make_renderer = TARGETS[device]
  local_route = PackedWmmaRoute(QUANT, ROLE, SHAPE, GEOMETRY, canonical_identity=f"m1d-probe-{device.lower()}")
  payload = _payload_for_local_row(PROFILE, local_route)
  entry = derive_packed_weight_candidate(payload, QUANT)
  final_payload = entry.to_json()["payload"]
  admission = admit_current_prefill(final_payload, entry.canonical_identity)

  renderer = make_renderer(Target.parse(target_str))

  # AMD's real cross-compiler (amd_comgr, invoked from tinygrad/runtime/support/compiler_amd.py) crashes
  # (SIGBUS) on this machine when actually asked to assemble HIP source to a GCN binary -- a native-library
  # instability unrelated to what this diagnosis is about. The postrange optimization we're tracing (where
  # the C-fragment axes/accumulator_total hypothesis lives) runs entirely BEFORE that compile step, inside
  # to_program's `full_rewrite_to_sink` call. Stub out the compile step so we still get real SOURCE text and
  # a real postrange trace without touching that crashing native path -- purely diagnostic, no GPU involved
  # either way (this would never have executed on a GPU; it's a cross-compile call for a target this Mac has
  # no runtime for).
  compiler_patch = None
  if device == "AMD":
    import unittest.mock
    compiler_patch = unittest.mock.patch.object(type(renderer.compiler), "compile", lambda self, src: b"")

  captured.clear()
  sys.settrace(_tracer)
  err = None
  program = None
  try:
    if compiler_patch is not None:
      with compiler_patch:
        program = _build_program(final_payload, admission.canonical_identity, admission.context, device, renderer)
    else:
      program = _build_program(final_payload, admission.canonical_identity, admission.context, device, renderer)
  except Exception as exc:  # noqa: BLE001 -- we want to report, not hide, whatever fires
    err = f"{type(exc).__name__}: {exc}"
  finally:
    sys.settrace(None)

  result = {"device": device, "target": target_str, "tensor_cores_elements_per_thread": [tuple(tc.elements_per_thread) for tc in renderer.tensor_cores],
             "error": err, "trace": list(captured)}
  if program is not None:
    source = next((u.arg for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
    result["source_len"] = len(source) if source else None
    if source:
      import re
      result["wmma_call_count"] = source.count("__WMMA")
      result["simdgroup_multiply_accumulate_count"] = source.count("simdgroup_multiply_accumulate")
      result["v_wmma_count"] = source.count("v_wmma")
      # grab a representative accumulator-array declaration / WMMA wrapper signature line for evidence
      wmma_wrapper_lines = [l for l in source.splitlines() if "__WMMA" in l and ("float" in l or "half" in l or "static" in l)][:3]
      result["wmma_wrapper_sample_lines"] = wmma_wrapper_lines
      with open(f"/tmp/m1d_{device.lower()}_source.c", "w") as sf:
        sf.write(source)
  return result


def main():
  out = {}
  for device in ("METAL", "AMD"):
    print(f"=== rendering {device} ===")
    try:
      out[device] = render_one(device)
    except Exception as exc:
      out[device] = {"device": device, "fatal_error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out[device], indent=2, default=str))
  with open("/tmp/m1d_trace_result.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
  print("wrote /tmp/m1d_trace_result.json")


if __name__ == "__main__":
  main()
