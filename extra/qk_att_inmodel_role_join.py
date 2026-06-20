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
    elif "q4k_gemv" in name and "12288_4096" in name: role = "ffn_gate/up_native_q4k_gemv"
    elif "q4k_coop" in name and "12288_4096" in name: role = "ffn_gate/up_native_q4k_coop"
    elif "q6k_coop_partial_4096_12288" in name: role = "ffn_down_native_q6k_coop"
    elif "q6k_coop_partial_151936_4096" in name: role = "lm_head_native_q6k_coop"
    elif "q6k_gemv" in name: role = "q6k_native_other"
    elif "q8" in name.lower() and ("gate" in name.lower() or "ffn" in name.lower()): role = "ffn_gate/up_q8_route"
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
      "ffn_gateup_native_present": any(str(v.get("role") or "").startswith("ffn_gate/up_native") for v in variants),
      "q8_gateup_present": any(v.get("role") == "ffn_gate/up_q8_route" for v in variants),
      "q6k_native_coop_present": any(v.get("role") in ("ffn_down_native_q6k_coop", "lm_head_native_q6k_coop") for v in variants),
      "fallback_dense_suspected": not any(("q4k" in v["program_name"] or "q6k" in v["program_name"]) for v in variants),
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


def first_token_hidden(model: Any, tok: Any):
  from tinygrad import Tensor, dtypes
  from extra.qk_nll_eval import CALIB_TEXT
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  return x


def capture_ffn_down_activation(model: Any, tok: Any):
  block = model.blk[0]
  original = block.ffn_down
  x = first_token_hidden(model, tok)
  block._init_state(x)
  cap = CaptureLinear(original)
  block.ffn_down = cap
  try:
    h = x + block._attention(block.attn_norm(x), 0)
    _ = block._feed_forward(block.ffn_norm(h)).realize()
  finally:
    block.ffn_down = original
  if cap.last_input is None:
    raise RuntimeError("failed to capture blk.0.ffn_down activation")
  return block, original, cap.last_input


class PairLinear:
  def __init__(self, left, right):
    self.left, self.right = left, right
    self.name = "ffn_gateup_pair"
    self.decode_enabled = bool(getattr(left, "decode_enabled", False)) and bool(getattr(right, "decode_enabled", False))
    self.out_features = [getattr(left, "out_features", None), getattr(right, "out_features", None)]
    self.in_features = getattr(left, "in_features", getattr(right, "in_features", None))
    self.parts = [getattr(left, "parts", None), getattr(right, "parts", None)]
    self.kernel_mode = "pair"

  def __call__(self, x):
    return self.left(x).realize(), self.right(x).realize()


def capture_ffn_norm_activation(model: Any, tok: Any):
  block = model.blk[0]
  x = first_token_hidden(model, tok)
  block._init_state(x)
  h = (x + block._attention(block.attn_norm(x), 0)).realize()
  return block, block.ffn_norm(h).realize()


def capture_ffn_gate_activation(model: Any, tok: Any):
  block, xh = capture_ffn_norm_activation(model, tok)
  return block, block.ffn_gate, xh


def capture_ffn_up_activation(model: Any, tok: Any):
  block, xh = capture_ffn_norm_activation(model, tok)
  return block, block.ffn_up, xh


def capture_ffn_gateup_pair_activation(model: Any, tok: Any):
  block, xh = capture_ffn_norm_activation(model, tok)
  return block, PairLinear(block.ffn_gate, block.ffn_up), xh


def capture_lm_head_activation(model: Any, tok: Any):
  x = first_token_hidden(model, tok)
  for block in model.blk:
    block._init_state(x)
    x = block(x, 0).realize()
  hidden = model.output_norm(x).realize()
  return model, model.output, hidden


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


def role_capture(model: Any, tok: Any, role: str):
  if role == "attn_output": return capture_attn_output_activation(model, tok)
  if role == "ffn_gate": return capture_ffn_gate_activation(model, tok)
  if role == "ffn_up": return capture_ffn_up_activation(model, tok)
  if role == "ffn_gateup_pair": return capture_ffn_gateup_pair_activation(model, tok)
  if role == "ffn_down": return capture_ffn_down_activation(model, tok)
  if role == "lm_head": return capture_lm_head_activation(model, tok)
  raise ValueError(f"unsupported role {role!r}")


def q6_surface_role(role: str):
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.model import Q6KPrimitiveLinear
  from extra.q6_k_gemv_primitive import parse_opt
  from extra.qk_layout import GGML_Q6_K, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, read_metadata, tensor_shape

  tensor = {"ffn_down": "blk.0.ffn_down.weight", "lm_head": "output.weight"}[role]
  meta = read_metadata(MODEL)
  info = next(i for i in meta.infos if i.name == tensor)
  if info.typ != GGML_Q6_K: raise RuntimeError(f"{tensor} is not Q6_K: type={info.typ}")
  rows, k = tensor_shape(info)
  row_bytes = k // Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES
  nbytes = rows * row_bytes
  halfs = Tensor(MODEL, dtype=dtypes.uint16)[(meta.data_start + info.off)//2:(meta.data_start + info.off + nbytes)//2].to("AMD").contiguous().realize()
  lin = Q6KPrimitiveLinear(None, None, halfs, rows, k, 1, (parse_opt("LOCAL:0:64"),), tensor, nbytes, nbytes, "sidecar")
  lin.decode_enabled = True
  rng = np.random.default_rng(20260619 + (0 if role == "ffn_down" else 1))
  x = Tensor(rng.standard_normal((1, 1, k)).astype(np.float32), dtype=dtypes.float32, device="AMD").contiguous().realize()
  return None, lin, x


def realize_output(out: Any):
  if isinstance(out, tuple): return tuple(x.realize() if hasattr(x, "realize") else x for x in out)
  if isinstance(out, list): return [x.realize() if hasattr(x, "realize") else x for x in out]
  return out.realize() if hasattr(out, "realize") else out


def output_shape(out: Any):
  if isinstance(out, tuple): return [list(getattr(x, "shape", ())) for x in out]
  if isinstance(out, list): return [list(getattr(x, "shape", ())) for x in out]
  return list(getattr(out, "shape", ()))


def run_role(model: Any, tok: Any, export: dict[str, Any], role: str) -> dict[str, Any]:
  from tinygrad import Device
  from extra.qk_att_primitive_atlas import ATTInterval, build_export

  result: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "ATT in-model role join",
    "target_role": role,
    "activation": None,
    "interval": None,
    "programs": None,
    "surface_reference": load_surface_reference(),
    "gates": {},
    "verdict": "NOT_RUN",
  }

  capture_mode = "inmodel_activation"
  if model is None:
    _owner, linear, activation = q6_surface_role(role)
    capture_mode = "q6_surface_fallback"
  else:
    _owner, linear, activation = role_capture(model, tok, role)
  result["activation"] = {
    "capture_mode": capture_mode,
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
  warm = realize_output(linear(activation))
  Device["AMD"].synchronize(timeout=10000)
  result["activation"]["warm_output_shape"] = output_shape(warm)

  cap = ProgramCapture()
  att = ATTInterval(export)

  def role_call() -> dict[str, Any]:
    with capture_hcq_programs(cap):
      t0 = time.perf_counter()
      out = realize_output(linear(activation))
      Device["AMD"].synchronize(timeout=10000)
      wall_ms = (time.perf_counter() - t0) * 1000.0
    return {"output_shape": output_shape(out), "wall_ms": round(wall_ms, 6)}

  interval = att.trace("blk0_attn_output_inmodel_role", role_call)
  programs = cap.summary()
  result["interval"] = interval
  result["programs"] = programs
  trace_body = int(interval["trace"].get("body_like_packet_count") or 0)
  program_ok = programs["program_call_count"] > 0
  native_ok = bool(programs["q4k_native_coop_present"] or programs["q6k_native_coop_present"] or
                   programs.get("ffn_gateup_native_present") or programs.get("q8_gateup_present"))
  result["gates"] = {
    "att_start_stop_sync": "PASS" if interval["start"]["sync_ok"] and interval["stop"]["sync_ok"] else "FAIL",
    "att_body_packets": "PASS" if trace_body > 0 else "FAIL",
    "programs_captured": "PASS" if program_ok else "FAIL",
    "native_coop_present": "PASS" if native_ok else "FAIL",
    "decode_primitives_enabled": "PASS" if result["activation"]["decode_enabled"] else "FAIL",
  }
  if all(v == "PASS" for v in result["gates"].values()):
    result["verdict"] = "PASS_INMODEL_ROLE_JOIN_NATIVE_COOP"
  elif interval["start"]["sync_ok"] and interval["stop"]["sync_ok"] and trace_body > 0 and program_ok:
    result["verdict"] = "PARTIAL_INMODEL_ROLE_JOIN"
  else:
    result["verdict"] = "FAIL_INMODEL_ROLE_JOIN"
  return result


def main() -> int:
  from extra.qk_att_primitive_atlas import build_export

  OUTDIR.mkdir(parents=True, exist_ok=True)
  roles = sys.argv[1:] or ["attn_output"]
  build, helper_run, export = build_export()
  final: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "ATT in-model role join",
    "roles_requested": roles,
    "helper": {"build": build, "run": helper_run, "export_ok": bool(isinstance(export, dict) and export.get("ok"))},
    "roles": {},
    "gates": {},
    "verdict": "NOT_RUN",
  }
  if not isinstance(export, dict) or not export.get("ok"):
    final["verdict"] = "HELPER_EXPORT_FAIL"
    OUT.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return 1

  load_error = None
  try:
    model, tok = load_model()
  except Exception as e:
    load_error = repr(e)
    model, tok = None, None
    final["model_load_error"] = load_error
    if not all(r in ("ffn_down", "lm_head") for r in roles):
      final["verdict"] = "MODEL_LOAD_FAIL"
      OUT.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
      print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": final["verdict"], "model_load_error": load_error}, indent=2))
      return 1
  for role in roles:
    final["roles"][role] = run_role(model, tok, export, role)
    (OUTDIR / f"{role}.json").write_text(json.dumps(final["roles"][role], indent=2, sort_keys=True) + "\n")

  final["gates"] = {role: row["verdict"] for role, row in final["roles"].items()}
  if all(str(v).startswith("PASS") for v in final["gates"].values()):
    final["verdict"] = "PASS_ALL_ROLE_JOINS"
  elif any(str(v).startswith("PASS") or str(v).startswith("PARTIAL") for v in final["gates"].values()):
    final["verdict"] = "PARTIAL_ROLE_JOINS"
  else:
    final["verdict"] = "FAIL_ROLE_JOINS"
  OUT.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")

  summary = {
    "verdict": final["verdict"],
    "roles": {
      role: {
        "verdict": row["verdict"],
        "gates": row["gates"],
        "trace_body_packets": int((row["interval"] or {}).get("trace", {}).get("body_like_packet_count") or 0),
        "program_call_count": (row["programs"] or {}).get("program_call_count"),
        "variants": (row["programs"] or {}).get("variants"),
        "activation": row.get("activation"),
      } for role, row in final["roles"].items()
    },
  }
  (OUTDIR / "summary.md").write_text("# ATT in-model role join summary\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), **summary}, indent=2, sort_keys=True))
  return 0 if final["verdict"] != "FAIL_ROLE_JOINS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
