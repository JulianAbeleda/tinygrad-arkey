"""C1 — artifact cache helper for the PMS/TG pure-search stack. Stable JSON/code/runtime fingerprints so static
artifacts can be reused by exact input+code hash, correctness artifacts only with matching route/code/model/runtime
fingerprints, and speed artifacts treated as historical unless an exact runtime fingerprint is explicitly accepted.

Phase C1 of docs/pure-machine-search-artifact-cache-scope-20260630.md. This is the FINGERPRINT/IO layer only — it does
NOT wire into the generators or evaluator (C2/C3) and changes no defaults. No GPU is required for Class A validation.

Cache classes:
  A_static      : profile/quant/target/grammar/template/candidate generation -> reuse by hash(inputs + code).
  B_correctness : token/logit/route attribution -> reuse only if inputs+code+runtime fingerprints all match.
  C_speed       : W==D / whole-prefill / PMC -> historical by default; promotion reruns unless caller accepts cached.

Run self-test (negative stale tests): PYTHONPATH=. python3 extra/qk/artifact_cache.py
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, pathlib, datetime

SCHEMA = "qk_artifact_cache_v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]
# fields that must NEVER enter a cache key (volatile / non-deterministic)
_VOLATILE = ("generated_at", "timestamp", "wall_clock", "_log", "duration_ms")

def file_sha256(path: str) -> str:
  p = pathlib.Path(path)
  if not p.is_absolute(): p = ROOT / p
  if not p.exists(): return f"MISSING:{path}"
  h = hashlib.sha256()
  with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
  return h.hexdigest()

def _strip_volatile(obj):
  """Recursively drop volatile fields + the 'cache' meta block so hashing is stable across re-emits."""
  if isinstance(obj, dict):
    return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE and k != "cache"}
  if isinstance(obj, list): return [_strip_volatile(v) for v in obj]
  return obj

def json_sha256(obj: object) -> str:
  """Stable hash of normalized JSON: volatile fields stripped, keys sorted, compact separators."""
  canon = json.dumps(_strip_volatile(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  return hashlib.sha256(canon.encode()).hexdigest()

def code_fingerprint(paths: list[str]) -> dict:
  """Per-file sha256 of declared code paths (sorted), plus a combined hash."""
  per = {p: file_sha256(p) for p in sorted(paths)}
  combined = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
  return {"files": per, "combined": combined}

def runtime_fingerprint() -> dict:
  """Best-effort runtime identity (no hard GPU dependency). Used only for Class B/C reuse, never Class A."""
  def _git(*a):
    try: return subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return None
  rocm = None
  try: rocm = subprocess.run(["bash", "-lc", "cat /opt/rocm/.info/version 2>/dev/null"], capture_output=True, text=True, timeout=5).stdout.strip() or None
  except Exception: pass
  fp = {
    "git_head": _git("rev-parse", "HEAD"),
    "python_version": sys.version.split()[0],
    "gpu_target_id": os.environ.get("DEV", "AMD"),
    "rocm_version": rocm,
    "route_env_keys": {k: os.environ[k] for k in sorted(os.environ)
                       if any(k.startswith(pre) for pre in ("Q4K_", "Q6K_", "PREFILL_", "DECODE_", "BUBBLEBEAM_", "BEAM_"))},
  }
  fp["combined"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
  return fp

def build_cache_key(kind: str, inputs: dict, code_paths: list[str], runtime: bool = False) -> dict:
  """Returns the cache_key + its components. kind in {A_static,B_correctness,C_speed}. runtime=True folds the runtime
  fingerprint into the key (required for B/C reuse; never for A)."""
  inputs_hash = json_sha256(inputs)
  code_hash = code_fingerprint(code_paths)["combined"]
  runtime_hash = runtime_fingerprint()["combined"] if runtime else None
  material = json.dumps({"kind": kind, "inputs": inputs_hash, "code": code_hash, "runtime": runtime_hash}, sort_keys=True)
  return {"cache_key": hashlib.sha256(material.encode()).hexdigest(), "inputs_hash": inputs_hash,
          "code_hash": code_hash, "runtime_hash": runtime_hash, "class": kind}

def cache_meta(kind: str, inputs: dict, code_paths: list[str], runtime: bool = False, **ids) -> dict:
  k = build_cache_key(kind, inputs, code_paths, runtime)
  return {"schema": SCHEMA, "class": kind, "cache_key": k["cache_key"], "inputs_hash": k["inputs_hash"],
          "code_hash": k["code_hash"], "runtime_hash": k["runtime_hash"],
          "profile_id": ids.get("profile_id"), "route_id": ids.get("route_id"), "target_id": ids.get("target_id"),
          "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "validity": "fresh" if kind != "C_speed" else "historical_only"}

def load_if_fresh(path: str, expected_key: str) -> dict | None:
  """Return the artifact dict iff it exists and its cache.cache_key == expected_key; else None (miss)."""
  p = pathlib.Path(path)
  if not p.is_absolute(): p = ROOT / p
  if not p.exists(): return None
  try: d = json.load(open(p))
  except Exception: return None
  if isinstance(d, dict) and d.get("cache", {}).get("cache_key") == expected_key: return d
  return None

def write_artifact(path: str, payload: dict, meta: dict) -> None:
  p = pathlib.Path(path)
  if not p.is_absolute(): p = ROOT / p
  p.parent.mkdir(parents=True, exist_ok=True)
  out = dict(payload); out["cache"] = meta
  json.dump(out, open(p, "w"), indent=2)

def emit_artifact(out_dir, payload: dict, md_lines=None, *, kind: str = "derived_artifact",
                  inputs: dict | None = None, code_paths: list[str] | None = None, runtime: bool = False, **ids):
  """The ONE writer for a tool's latest.json (+ optional summary.md): writes the payload with a stamped cache block so
  the artifact carries provenance and can be reused/invalidated. kind = 'input_artifact' (a config/descriptor the human
  edits) | 'derived_artifact' (generated; keyed on inputs + code_paths so a threshold/grammar/source change invalidates
  it). Output is byte-identical to a hand-rolled json.dump(...indent=2) EXCEPT the added top-level 'cache' block."""
  d = pathlib.Path(out_dir)
  if not d.is_absolute(): d = ROOT / d
  d.mkdir(parents=True, exist_ok=True)
  meta = cache_meta("A_static", inputs or {}, code_paths or [], runtime=runtime, **ids)
  meta["artifact_kind"] = kind
  write_artifact(d / "latest.json", payload, meta)
  if md_lines is not None:
    (d / "summary.md").write_text("\n".join(md_lines) if isinstance(md_lines, (list, tuple)) else md_lines)
  return d / "latest.json"

def _selftest() -> int:
  import tempfile
  ok = True
  inputs = {"profile": "qwen3_8b_q4_k_m_gfx1100", "grammar": {"bg": [1,2,4,8,16], "wpg": 8}, "generated_at": "VOLATILE-IGNORE-ME"}
  codep = ["extra/qk/artifact_cache.py"]
  k1 = build_cache_key("A_static", inputs, codep)["cache_key"]
  # stability: same inputs (different volatile) -> same key
  inputs2 = dict(inputs); inputs2["generated_at"] = "DIFFERENT-VOLATILE"
  k_stable = build_cache_key("A_static", inputs2, codep)["cache_key"]
  ok &= (k1 == k_stable); print(f"  [stable] volatile field ignored: {k1 == k_stable}")
  # negative: change an input -> stale
  inputs3 = dict(inputs); inputs3["grammar"] = {"bg": [1,2,4], "wpg": 4}
  k_inputchg = build_cache_key("A_static", inputs3, codep)["cache_key"]
  ok &= (k1 != k_inputchg); print(f"  [stale-on-input-change] key differs: {k1 != k_inputchg}")
  # negative: change code set -> stale (add a second file)
  k_codechg = build_cache_key("A_static", inputs, codep + ["extra/qk/route_manifest.py"])["cache_key"]
  ok &= (k1 != k_codechg); print(f"  [stale-on-code-change] key differs: {k1 != k_codechg}")
  # class separation: A vs B (runtime) differ
  kB = build_cache_key("B_correctness", inputs, codep, runtime=True)["cache_key"]
  ok &= (k1 != kB); print(f"  [class/runtime separation] A != B(runtime): {k1 != kB}")
  # load_if_fresh round-trip
  with tempfile.TemporaryDirectory() as td:
    fp = os.path.join(td, "a.json"); meta = cache_meta("A_static", inputs, codep, profile_id="p")
    write_artifact(fp, {"result": 42}, meta)
    hit = load_if_fresh(fp, meta["cache_key"]); miss = load_if_fresh(fp, "WRONGKEY")
    ok &= (hit is not None and hit["result"] == 42 and miss is None)
    print(f"  [load_if_fresh] hit-on-match + miss-on-mismatch: {hit is not None and miss is None}")
  print("\nC1 self-test:", "PASS" if ok else "FAIL")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(_selftest())
