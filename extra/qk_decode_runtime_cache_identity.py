#!/usr/bin/env python3
"""PCG-1 / FMI-4 B2 decode runtime-cache identity probe.

Read-only diagnostic. It compares actual HCQ program/runtime identities for:
  1. one warm in-model decode step
  2. direct same-process calls to representative installed Q4_K/Q6_K linears

The goal is not timing. It asks whether the in-model path silently uses a
different compiled program/cache key/launch contract than the intended role
surface. If identities match, B2 closes and the remaining decode gap is not a
bounded runtime-cache wiring issue.
"""
from __future__ import annotations

import contextlib, hashlib, json, os, pathlib, re, sys
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-fused-mmvq-integration"

SHAPE_ROLE = {
  (151936, 4096): "lm_head",
  (4096, 12288): "ffn_down",
  (12288, 4096): "ffn_gate/up",
  (4096, 4096): "attn_q/o",
  (1024, 4096): "attn_k/v",
}
GEMV_RE = re.compile(r"q([46])k_.*?_(\d+)_(\d+)(?:_|$)")


class Capture:
  def __init__(self):
    self.scope = "unset"
    self.rows: list[dict[str, Any]] = []
    self.runtime_seen: dict[int, dict[str, Any]] = {}

  def classify(self, name: str) -> tuple[str | None, tuple[int, int] | None]:
    m = GEMV_RE.search(name)
    if not m: return None, None
    shape = (int(m.group(2)), int(m.group(3)))
    return SHAPE_ROLE.get(shape), shape

  def add_runtime(self, device: str, ast: Any, runtime: Any, existed: bool) -> None:
    info = getattr(ast, "arg", None)
    lib = getattr(runtime, "lib", b"") or b""
    self.runtime_seen[id(runtime)] = {
      "device": device,
      "ast_key": ast.key.hex()[:24] if isinstance(getattr(ast, "key", None), bytes) else str(getattr(ast, "key", ""))[:24],
      "ast_name": getattr(info, "name", None),
      "program_name": getattr(runtime, "name", None),
      "cache_existed_before_get": existed,
      "program_object_id": id(runtime),
      "prof_prg_counter": getattr(runtime, "prof_prg_counter", None),
      "kernargs_alloc_size": getattr(runtime, "kernargs_alloc_size", None),
      "lib_sha16": hashlib.sha256(lib).hexdigest()[:16] if lib else None,
      "lib_bytes": len(lib) if lib else 0,
      "program_info_global_size": list(getattr(info, "global_size", ()) or ()),
      "program_info_local_size": list(getattr(info, "local_size", ()) or ()),
    }

  def add_call(self, runtime: Any, global_size: tuple[int, ...], local_size: tuple[int, ...] | None, vals: tuple[Any, ...]) -> None:
    name = getattr(runtime, "name", "")
    role, shape = self.classify(name)
    if role is None: return
    meta = self.runtime_seen.get(id(runtime), {})
    self.rows.append({
      "scope": self.scope,
      "role": role,
      "shape": list(shape) if shape else None,
      "program_name": name,
      "program_object_id": id(runtime),
      "prof_prg_counter": getattr(runtime, "prof_prg_counter", None),
      "global_size": list(global_size or ()),
      "local_size": list(local_size or ()),
      "vals_len": len(vals or ()),
      "ast_key": meta.get("ast_key"),
      "cache_existed_before_get": meta.get("cache_existed_before_get"),
      "kernargs_alloc_size": meta.get("kernargs_alloc_size"),
      "lib_sha16": meta.get("lib_sha16"),
      "lib_bytes": meta.get("lib_bytes"),
    })


@contextlib.contextmanager
def install_capture(cap: Capture):
  import tinygrad.engine.realize as realize
  from tinygrad.runtime.support import hcq

  orig_get_runtime = realize.get_runtime
  orig_call = hcq.HCQProgram.__call__

  def wrapped_get_runtime(device: str, ast: Any, cache=True):
    existed = (ast.key, device) in realize.runtime_cache
    rt = orig_get_runtime(device, ast, cache=cache)
    cap.add_runtime(device, ast, rt, existed)
    return rt

  def wrapped_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, timeout=None):
    cap.add_call(self, tuple(global_size or ()), tuple(local_size or ()) if local_size is not None else None, tuple(vals or ()))
    return orig_call(self, *bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, timeout=timeout)

  realize.get_runtime = wrapped_get_runtime
  hcq.HCQProgram.__call__ = wrapped_call
  try:
    yield
  finally:
    realize.get_runtime = orig_get_runtime
    hcq.HCQProgram.__call__ = orig_call


def role_of_linear(lin: Any) -> str | None:
  shape = (getattr(lin, "out_features", None), getattr(lin, "in_features", None))
  if shape == (4096, 4096): return "attn_q/o"
  if shape == (12288, 4096): return "ffn_gate/up"
  if shape == (4096, 12288): return "ffn_down"
  if shape[0] and shape[0] >= 100000: return "lm_head"
  if shape == (1024, 4096): return "attn_k/v"
  return None


def pick_representatives(model: Any) -> dict[str, list[Any]]:
  reps: dict[str, list[Any]] = defaultdict(list)
  seen: set[tuple[Any, ...]] = set()
  for lin in getattr(getattr(model, "_q4k_linears", None), "linears", []) or []:
    role = role_of_linear(lin)
    key = (role, type(lin).__name__, getattr(lin, "out_features", None), getattr(lin, "in_features", None),
           getattr(lin, "parts", None), getattr(lin, "kernel_mode", None))
    if role and key not in seen:
      reps[role].append(lin)
      seen.add(key)
  return reps


def canonical_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  by: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
  for r in rows:
    key = (
      r["program_name"], tuple(r["global_size"]), tuple(r["local_size"]),
      r.get("ast_key"), r.get("kernargs_alloc_size"), r.get("lib_sha16"),
    )
    item = by[r["role"]].setdefault(key, {k: r[k] for k in r if k not in ("scope",)})
    item["calls"] = item.get("calls", 0) + 1
  return {role: sorted(items.values(), key=lambda x: (-x["calls"], x["program_name"])) for role, items in sorted(by.items())}


def compare(inmodel: dict[str, list[dict[str, Any]]], standalone: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
  out = []
  for role in sorted(set(inmodel) | set(standalone)):
    im = inmodel.get(role, [])
    st = standalone.get(role, [])
    im_keys = {(r["program_name"], tuple(r["global_size"]), tuple(r["local_size"]), r.get("ast_key")) for r in im}
    st_keys = {(r["program_name"], tuple(r["global_size"]), tuple(r["local_size"]), r.get("ast_key")) for r in st}
    im_no_ast = {(r["program_name"], tuple(r["global_size"]), tuple(r["local_size"])) for r in im}
    st_no_ast = {(r["program_name"], tuple(r["global_size"]), tuple(r["local_size"])) for r in st}
    out.append({
      "role": role,
      "inmodel_variants": im,
      "standalone_variants": st,
      "identity_match": im_no_ast.issubset(st_no_ast) and bool(im_no_ast),
      "inmodel_only": [list(x) for x in sorted(im_keys - st_keys)],
      "standalone_only": [list(x) for x in sorted(st_keys - im_keys)],
      "inmodel_only_without_ast_key": [list(x) for x in sorted(im_no_ast - st_no_ast)],
    })
  return out


def summarize_decision(comparisons: list[dict[str, Any]]) -> tuple[str, str]:
  high_share = {"ffn_gate/up", "ffn_down", "lm_head", "attn_q/o"}
  mismatched = [c["role"] for c in comparisons if c["role"] in high_share and c["standalone_variants"] and not c["identity_match"]]
  missing = [c["role"] for c in comparisons if c["role"] in high_share and not c["standalone_variants"]]
  if missing:
    return "B2_INCONCLUSIVE_MISSING_STANDALONE_ROLE", f"Missing direct-call identity rows for high-share roles: {missing}"
  if mismatched:
    return "B2_MISMATCH_FOUND_SCOPE_FIX", f"High-share roles have in-model vs direct-call identity mismatch: {mismatched}"
  return "B2_CLOSED_NO_RUNTIME_CACHE_MISMATCH", "Representative high-share roles reuse the same program/cache/launch identity in-model and direct-call; remaining gap is not a bounded runtime-cache wiring issue."


def main() -> None:
  model_path = next((a for a in sys.argv[1:] if a.endswith(".gguf")),
                    os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  from tinygrad import Tensor, Context
  from extra.llm_generate import load_model_and_tokenizer

  cap = Capture()
  with install_capture(cap):
    model, tok = load_model_and_tokenizer(model_path, 2048, seed=20260617)
    for lin in getattr(getattr(model, "_q4k_linears", None), "linears", []) or []:
      lin.decode_enabled = True

    ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox. " * 40)
    sp, tokid = 64, int(ids[64])
    with Context(DEBUG=0):
      model.logits(Tensor([ids[:sp]], dtype="int32").contiguous(), 0).realize()
      for _ in range(2):
        cap.scope = "inmodel_warm"
        model.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
      cap.scope = "inmodel_probe"
      model.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()

      reps = pick_representatives(model)
      standalone_roles = {}
      for role, lins in reps.items():
        standalone_roles[role] = []
        for idx, lin in enumerate(lins):
          cap.scope = f"standalone_{role}_{idx}"
          x = Tensor.randn(1, 1, lin.in_features, dtype="float32").contiguous().realize()
          lin.decode_enabled = True
          lin(x).realize()
          standalone_roles[role].append({
            "name": getattr(lin, "name", ""),
            "class": type(lin).__name__,
            "shape": [getattr(lin, "out_features", None), getattr(lin, "in_features", None)],
            "parts": getattr(lin, "parts", None),
          })

  inmodel_rows = [r for r in cap.rows if r["scope"] == "inmodel_probe"]
  standalone_rows = [r for r in cap.rows if r["scope"].startswith("standalone_")]
  inmodel = canonical_rows(inmodel_rows)
  standalone = canonical_rows(standalone_rows)
  comparisons = compare(inmodel, standalone)
  status, reason = summarize_decision(comparisons)

  result = {
    "schema": "decode_mmvq_runtime_cache_identity_v1",
    "phase": "PCG-1/FMI-4-B2",
    "status": status,
    "reason": reason,
    "model": pathlib.Path(model_path).name,
    "method": {
      "runtime_hook": "tinygrad.engine.realize.get_runtime",
      "launch_hook": "tinygrad.runtime.support.hcq.HCQProgram.__call__",
      "comparison": "same-process warm in-model decode step vs direct calls to representative installed linears",
      "identity_key": ["program_name", "global_size", "local_size", "ast_key"],
    },
    "standalone_representatives": standalone_roles,
    "inmodel": inmodel,
    "standalone": standalone,
    "comparisons": comparisons,
    "raw_row_count": len(cap.rows),
    "decision": "Scope bounded fix" if status.startswith("B2_MISMATCH") else "Close B2; continue only with renderer/scheduler or artifact/import for large decode movement.",
  }

  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "runtime_cache_identity.json").write_text(json.dumps(result, indent=2) + "\n")
  lines = ["# Decode MMVQ runtime/cache identity", "", f"Verdict: `{status}`.", "", reason, ""]
  lines.append("| role | identity match | in-model variants | standalone variants |")
  lines.append("|---|---:|---:|---:|")
  for c in comparisons:
    lines.append(f"| `{c['role']}` | `{c['identity_match']}` | `{len(c['inmodel_variants'])}` | `{len(c['standalone_variants'])}` |")
  lines += ["", result["decision"], ""]
  (OUT / "runtime_cache_identity_summary.md").write_text("\n".join(lines))
  print(json.dumps({"status": status, "reason": reason, "out": str(OUT / "runtime_cache_identity.json")}, indent=2))


if __name__ == "__main__":
  main()
