#!/usr/bin/env python3
"""Default-off whole-model gate for ordered pp512 Q4 IMMA scratch.

This is deliberately separate from ``prefill_whole_synced``.  The packed-v4
research route owns gate/up outside the generated GEMM registry, so the
authority harness's generated-route census correctly cannot attribute it.
This gate instead proves the route's own executable ABI: 72 producer/main/
fixup calls, one ordered partial/id workspace, exact replay binding, full-logit
quality, and synchronized R9 wall.  It never changes the production default.
"""
from __future__ import annotations

import argparse, json, os, pathlib, time
from collections import Counter
from contextlib import contextmanager
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.generate import load_model_and_tokenizer
from tinygrad.llm.prefill_route_observer import prefill_route_scope
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_q4_imma_provider import PARTIAL_SLOTS

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
FP16_REFERENCE_MS = 83.793


def _program_calls(linear, name:str):
  return [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM and u.src[0].arg.name == name]


def _all_program_names(linear) -> Counter:
  return Counter(u.src[0].arg.name for u in linear.toposort()
                 if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM)


@contextmanager
def _prefill_compile_scope(model):
  import tinygrad.codegen.opt.postrange as pr
  merged = {**(model._pf16_warmstart or {}), **(model._packed_wmma_warmstart or {})}
  with prefill_route_scope(True), pr.warmstart_candidate_state(merged, model._packed_wmma_warmstart_contexts): yield


def _configure_prefill(model, binding=None):
  for q4k_linear in model._q4k_linears.linears: q4k_linear.decode_enabled = False
  for block in model.blk:
    block._use_flash, block._prefill_v2, block._is_prefill = True, True, True
    block._ring_freqs, block._ring_full = None, False
    if binding is not None: block._nv_q4_imma_pp512_binding = binding


def _call_and_sync(model, chunk, temp, dev):
  out = model(chunk, 0, temp, use_flash=True)
  out.realize(); dev.synchronize()
  return out


def _full_logits(model, chunk, temp, binding, *, packed:bool):
  _configure_prefill(model, binding if packed else None)
  if packed:
    @TinyJit
    def run(tokens, temperature):
      binding.begin_trace()
      return model.forward_greedy_with_logits(tokens, 0, temperature)
  else:
    @TinyJit
    def run(tokens, temperature):
      return model.forward_greedy_with_logits(tokens, 0, temperature)
  with _prefill_compile_scope(model):
    # TinyJit needs capture plus replay; return the replay result so rebinding
    # is covered by the same quality comparison.
    result = None
    for _ in range(3): result = run(chunk, temp)
  Device[Device.DEFAULT].synchronize()
  assert result is not None
  token, logits = result
  return int(token.numpy().reshape(-1)[0]), logits.numpy().copy(), run


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=("timing", "candidate-logits", "fp16-logits", "compare"), default="timing")
  ap.add_argument("--model", default=MODEL)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--warmups", type=int, default=3)
  ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--fp16-reference-ms", type=float, default=FP16_REFERENCE_MS)
  ap.add_argument("--logits-npz", default="")
  ap.add_argument("--candidate-npz", default="")
  ap.add_argument("--fp16-npz", default="")
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  if args.phase == "compare":
    if not args.candidate_npz or not args.fp16_npz: raise SystemExit("compare requires --candidate-npz and --fp16-npz")
    candidate, fp16 = np.load(args.candidate_npz), np.load(args.fp16_npz)
    candidate_logits, fp16_logits = candidate["logits"], fp16["logits"]
    candidate_token, fp16_token = int(candidate["token"]), int(fp16["token"])
    diff = np.abs(candidate_logits.astype(np.float32)-fp16_logits.astype(np.float32))
    quality = {"candidate_token":candidate_token, "fp16_token":fp16_token,
      "same_token":candidate_token == fp16_token,
      "finite":bool(np.isfinite(candidate_logits).all() and np.isfinite(fp16_logits).all()),
      "max_abs":float(diff.max()), "mean_abs":float(diff.mean()),
      "allclose_rtol_0p02_atol_0p5":bool(np.allclose(candidate_logits, fp16_logits, rtol=0.02, atol=0.5))}
    report = {"schema":"tinygrad.nv_q4_imma_ordered_model_quality.v1",
      "status":"PASS" if quality["same_token"] and quality["finite"] and quality["allclose_rtol_0p02_atol_0p5"] else "FAIL",
      "correctness":quality}
    payload=json.dumps(report, indent=2, sort_keys=True); print(payload)
    if args.out:
      path=pathlib.Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(payload+"\n")
    if report["status"] != "PASS": raise SystemExit(1)
    return

  packed = args.phase in ("timing", "candidate-logits")
  if packed and os.environ.get("NV_Q4_IMMA_PP512") != "1":
    raise SystemExit("fail closed: packed research phases require NV_Q4_IMMA_PP512=1")
  if not packed and os.environ.get("NV_Q4_IMMA_PP512") is not None:
    raise SystemExit("fail closed: fp16 comparator requires NV_Q4_IMMA_PP512 unset")

  dev = Device[Device.DEFAULT]
  if not str(Device.DEFAULT).startswith("NV"): raise SystemExit("NV device required")
  model, _ = load_model_and_tokenizer(args.model, args.max_context, seed=20260617)
  binding = None
  if packed:
    from extra.llm_research.prefill.nv_q4_imma_pp512_binding import binding_for
    binding = binding_for("NV")
    binding.prepare_outputs(len(model.blk)*2)
  chunk = Tensor([[(i*7)%1000 for i in range(512)]], dtype="int32").contiguous()
  temp = Tensor([0.0])

  if args.phase in ("candidate-logits", "fp16-logits"):
    token, logits, jit = _full_logits(model, chunk, temp, binding, packed=packed)
    if not args.logits_npz: raise SystemExit("logit phase requires --logits-npz")
    path=pathlib.Path(args.logits_npz); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path, token=np.int64(token), logits=logits)
    report={"schema":"tinygrad.nv_q4_imma_ordered_model_logits.v1", "status":"PASS",
      "arm":"candidate" if packed else "fp16", "token":token, "finite":bool(np.isfinite(logits).all()),
      "shape":list(logits.shape), "min":float(logits.min()), "max":float(logits.max()),
      "program_names":dict(sorted(_all_program_names(jit.captured.linear).items())) if jit.captured else {}}
    report["status"]="PASS" if report["finite"] else "FAIL"
    payload=json.dumps(report,indent=2,sort_keys=True); print(payload)
    if args.out:
      out=pathlib.Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(payload+"\n")
    if report["status"] != "PASS": raise SystemExit(1)
    return

  assert binding is not None

  # Exercise the exact production-shaped entry and its concrete pp512 TinyJit.
  for _ in range(args.warmups): _call_and_sync(model, chunk, temp, dev)
  from tinygrad.device import Compiled
  profile_start = len(Compiled.profile_events)
  samples = []
  timed_token = None
  for _ in range(args.rounds):
    dev.synchronize(); st = time.perf_counter()
    timed_token = _call_and_sync(model, chunk, temp, dev)
    samples.append((time.perf_counter()-st)*1e3)
  from extra.llm_research.prefill.prefill_whole_synced import profile_range_summary
  profile = profile_range_summary(list(Compiled.profile_events[profile_start:]))

  captured = model.prefill_v2_jits[(0, False)].captured
  if captured is None: raise RuntimeError("production-shaped prefill graph did not capture")
  linear = captured.linear
  producer_name = binding.producer_fp16.arg.name
  main_name, fixup_name = binding.main_program.arg.name, binding.fixup_program.arg.name
  producers = _program_calls(linear, producer_name)
  mains = _program_calls(linear, main_name)
  fixups = _program_calls(linear, fixup_name)
  partial_bases = {u.src[2].buf_uop for u in mains + fixups}
  id_bases = {u.src[3].buf_uop for u in mains}

  # B1's computed-input contract is visible at the native producer ABI: each
  # producer reads a PARAM/AFTER-carried value already owned by the enclosing
  # block call.  A standalone materialization would instead appear as a new
  # buffer-producing kernel directly between the block norm and producer.
  producer_inputs = [u.src[1] for u in producers]
  # A SLICE is a zero-byte view of the enclosing [B,T,K] function output, not
  # a device launch or allocation.  Gate/up consume two views of each block's
  # one norm result, hence 72 calls over exactly 36 physical input bases.
  computed_input_owned = all(x.op in {Ops.PARAM, Ops.AFTER, Ops.SLICE} or x.has_precompiled_output_identity() for x in producer_inputs) \
                         and len({x.buf_uop for x in producer_inputs}) == len(model.blk)
  q8_input_bases = len({x.buf_uop for x in producer_inputs}) if producer_inputs else 0

  workspace_bytes = PARTIAL_SLOTS*128*128*4 + PARTIAL_SLOTS*4
  candidate_ms = min(samples)
  program_names = _all_program_names(linear)
  call_census = {"producer":len(producers), "main":len(mains), "fixup":len(fixups)}
  graph_ok = call_census == {"producer":72, "main":72, "fixup":72}
  workspace_ok = len(partial_bases) == len(id_bases) == 1
  status = "PASS" if graph_ok and workspace_ok and computed_input_owned else "FAIL"
  report = {
    "schema":"tinygrad.nv_q4_imma_ordered_model_gate.v1", "status":status,
    "route":{"default_enabled":False, "required_env":"NV_Q4_IMMA_PP512=1", "fail_closed":True},
    "call_census":call_census,
    "computed_input":{"owned_without_adapter":computed_input_owned, "producer_input_physical_bases":q8_input_bases,
                      "producer_input_ops":dict(Counter(x.op.name for x in producer_inputs))},
    "ordered_workspace":{"partial_physical_bases":len(partial_bases), "id_physical_bases":len(id_bases),
      "bounded_bytes":workspace_bytes, "old_72_workspace_bytes":72*workspace_bytes,
      "saved_bytes":71*workspace_bytes},
    "wall":{"samples_ms":[round(x,4) for x in samples], "candidate_min_ms":round(candidate_ms,4),
      "fp16_reference_ms":args.fp16_reference_ms, "candidate_minus_fp16_ms":round(candidate_ms-args.fp16_reference_ms,4),
      "candidate_tok_s":round(512/candidate_ms*1000,2), "fp16_reference_tok_s":round(512/args.fp16_reference_ms*1000,2)},
    "device_profile":profile,
    "program_names":dict(sorted(program_names.items())),
  }
  payload = json.dumps(report, indent=2, sort_keys=True)
  print(payload)
  if args.out:
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(payload+"\n")
  if status != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
