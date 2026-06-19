#!/usr/bin/env python3
from __future__ import annotations

import contextlib, hashlib, json, pathlib, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTDIR = ROOT / "bench/qk-att-inmodel-role-join"
OUT = OUTDIR / "result.json"
MODEL = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")


class ProgramCapture:
  def __init__(self):
    self.rows: list[dict[str, Any]] = []

  @staticmethod
  def _hash(lib: bytes | None) -> str | None:
    return hashlib.sha256(lib).hexdigest()[:16] if lib else None

  def add(self, prg: Any, global_size: tuple[int, ...], local_size: tuple[int, ...] | None, vals: tuple[Any, ...]) -> None:
    name = getattr(prg, "name", type(prg).__name__)
    role = None
    if "q4k_coop_partial_4096_4096" in name: role = "attn_q/o_native_q4k_coop"
    elif "q4k_gemv" in name and "4096_4096" in name: role = "attn_q/o_native_q4k_other"
    elif "sum" in name.lower() or name.startswith("r_"): role = "role_reduce_or_glue"
    self.rows.append({
      "program_name": name,
      "role": role,
      "runtime_class": type(prg).__name__,
      "global_size": list(global_size or ()),
      "local_size": list(local_size or ()),
      "vals_len": len(vals or ()),
      "kernargs_alloc_size": getattr(prg, "kernargs_alloc_size", None),
      "prof_prg_counter": getattr(prg, "prof_prg_counter", None),
      "lib_sha16": self._hash(getattr(prg, "lib", None)),
      "lib_bytes": len(getattr(prg, "lib", b"") or b""),
    })

  def summary(self) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in self.rows:
      key = (row["program_name"], tuple(row["global_size"]), tuple(row["local_size"]))
      sk = repr(key)
      if sk not in by_name:
        by_name[sk] = {k: row[k] for k in row}
        by_name[sk]["calls"] = 0
      by_name[sk]["calls"] += 1
    variants = list(by_name.values())
    return {
      "program_call_count": len(self.rows),
      "variants": variants,
      "q4k_native_coop_present": any(v.get("role") == "attn_q/o_native_q4k_coop" for v in variants),
      "fallback_dense_suspected": not any("q4k" in v["program_name"] for v in variants),
    }


@contextlib.contextmanager
def capture_hcq_programs(cap: ProgramCapture):
  from tinygrad.runtime.support import hcq
  orig = hcq.HCQProgram.__call__

  def wrapped(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, timeout=None):
    cap.add(self, tuple(global_size or ()), tuple(local_size or ()) if local_size is not None else None, tuple(vals or ()))
    return orig(self, *bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, timeout=timeout)

  hcq.HCQProgram.__call__ = wrapped
  try:
    yield
  finally:
    hcq.HCQProgram.__call__ = orig


class CaptureLinear:
  def __init__(self, linear):
    self.linear = linear
    self.last_input = None

  def __getattr__(self, name):
    return getattr(self.linear, name)

  def __call__(self, x):
    from tinygrad import dtypes
    self.last_input = x.cast(dtypes.float32).contiguous().realize()
    return self.linear(x)


def capture_attn_output_activation(model: Any, tok: Any):
  from tinygrad import Tensor, dtypes
  from extra.qk_nll_eval import CALIB_TEXT

  block = model.blk[0]
  original = block.attn_output
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  block._init_state(x)
  cap = CaptureLinear(original)
  block.attn_output = cap
  try:
    _ = block._attention(block.attn_norm(x), 0).realize()
  finally:
    block.attn_output = original
  if cap.last_input is None:
    raise RuntimeError("failed to capture blk.0.attn_output activation")
  return block, original, cap.last_input


def load_model():
  from tinygrad import Device
  from extra.llm_generate import load_model_and_tokenizer
  if Device.DEFAULT != "AMD": raise RuntimeError(f"requires DEV=AMD, got {Device.DEFAULT!r}")
  model, tok = load_model_and_tokenizer(str(MODEL), 4096, seed=20260619)
  for lin in getattr(getattr(model, "_q4k_linears", None), "linears", []) or []:
    lin.decode_enabled = True
  return model, tok


def load_surface_reference() -> dict[str, Any]:
  path = ROOT / "bench/qk-att-primitive-atlas/decode_mmvq.json"
  if not path.exists(): return {"available": False}
  data = json.loads(path.read_text())
  rows = {}
  for row in data.get("rows", []):
    rows[row["label"]] = {
      "body_like_packet_count": row["trace"].get("body_like_packet_count"),
      "packet_top": row["trace"].get("packet_top"),
      "target": row.get("target"),
    }
  return {"available": True, "rows": rows}


def main() -> int:
  from tinygrad import Device
  from extra.qk_att_primitive_atlas import ATTInterval, build_export

  OUTDIR.mkdir(parents=True, exist_ok=True)
  build, helper_run, export = build_export()
  result: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "ATT in-model role join",
    "target_role": "blk.0.attn_output",
    "helper": {"build": build, "run": helper_run, "export_ok": bool(isinstance(export, dict) and export.get("ok"))},
    "activation": None,
    "interval": None,
    "programs": None,
    "surface_reference": load_surface_reference(),
    "gates": {},
    "verdict": "NOT_RUN",
  }
  if not isinstance(export, dict) or not export.get("ok"):
    result["verdict"] = "HELPER_EXPORT_FAIL"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 1

  model, tok = load_model()
  _block, linear, activation = capture_attn_output_activation(model, tok)
  result["activation"] = {
    "shape": list(activation.shape),
    "linear_type": type(linear).__name__,
    "linear_name": getattr(linear, "name", None),
    "decode_enabled": bool(getattr(linear, "decode_enabled", False)),
    "out_features": getattr(linear, "out_features", None),
    "in_features": getattr(linear, "in_features", None),
    "parts": getattr(linear, "parts", None),
    "kernel_mode": getattr(linear, "kernel_mode", None),
  }

  # Warm compile outside the ATT interval.
  warm = linear(activation).realize()
  Device["AMD"].synchronize(timeout=10000)
  result["activation"]["warm_output_shape"] = list(warm.shape)

  cap = ProgramCapture()
  att = ATTInterval(export)

  def role_call() -> dict[str, Any]:
    with capture_hcq_programs(cap):
      t0 = time.perf_counter()
      out = linear(activation).realize()
      Device["AMD"].synchronize(timeout=10000)
      wall_ms = (time.perf_counter() - t0) * 1000.0
    return {"output_shape": list(out.shape), "wall_ms": round(wall_ms, 6)}

  interval = att.trace("blk0_attn_output_inmodel_role", role_call)
  programs = cap.summary()
  result["interval"] = interval
  result["programs"] = programs
  trace_body = int(interval["trace"].get("body_like_packet_count") or 0)
  program_ok = programs["program_call_count"] > 0
  native_ok = bool(programs["q4k_native_coop_present"])
  result["gates"] = {
    "att_start_stop_sync": "PASS" if interval["start"]["sync_ok"] and interval["stop"]["sync_ok"] else "FAIL",
    "att_body_packets": "PASS" if trace_body > 0 else "FAIL",
    "programs_captured": "PASS" if program_ok else "FAIL",
    "q4k_native_coop_present": "PASS" if native_ok else "FAIL",
    "decode_primitives_enabled": "PASS" if result["activation"]["decode_enabled"] else "FAIL",
  }
  if all(v == "PASS" for v in result["gates"].values()):
    result["verdict"] = "PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP"
  elif interval["start"]["sync_ok"] and interval["stop"]["sync_ok"] and trace_body > 0 and program_ok:
    result["verdict"] = "PARTIAL_INMODEL_ROLE_JOIN"
  else:
    result["verdict"] = "FAIL_INMODEL_ROLE_JOIN"
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  summary = {
    "verdict": result["verdict"],
    "gates": result["gates"],
    "trace_body_packets": trace_body,
    "program_call_count": programs["program_call_count"],
    "variants": programs["variants"],
    "activation": result["activation"],
  }
  (OUTDIR / "summary.md").write_text("# ATT in-model role join summary\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), **summary}, indent=2, sort_keys=True))
  return 0 if result["verdict"] != "FAIL_INMODEL_ROLE_JOIN" else 1


if __name__ == "__main__":
  raise SystemExit(main())
