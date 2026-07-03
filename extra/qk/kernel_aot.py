"""AOT kernel bundle: ship a prebuilt, fingerprinted, golden-validated compiled-kernel cache so cold start is warm.

Background (audited against the pinned tinygrad):
  * The compile cache is a SQLite DB at ~/.cache/tinygrad/cache.db (helpers.CACHEDB). Compiled kernels live in a table
    named by the compiler's `cachekey` -- for AMD HIP that is "compile_hip_<arch>" (e.g. "compile_hip_gfx1100"); the row
    key is the RENDERED KERNEL SOURCE STRING and the value is the compiled lib bytes (device.py Compiler.compile_cached).
  * That cachekey embeds ARCH ONLY -- not the comgr/ROCm/driver/renderer versions. So a prebuilt DB shipped to a box
    with a different toolchain still KEY-HITS: correct binary format is not guaranteed by the key. This module adds the
    missing fingerprint + a load-time GOLDEN acceptance probe so a stale binary can never run-but-miscompute silently.

This module is machine-search-native AOT: the search/generate/compile happens once (offline / build machine), and the
compiled RESULT ships pinned + fingerprinted. It does NOT change any kernel or route -- it only materializes what the
runtime JIT would have compiled, and gates its reuse. Correctness authority stays the golden probe (never the fingerprint
alone, because the dangerous variable is precisely the one you forgot to fingerprint).

Split of costs (see startup_measure.py): shipping this bundle kills the COMPILE subprocess cost (b). The graph-capture /
render cost (a) still runs on warm start (the cache key is the rendered src, so trace->lower->render happens before the
lookup). Measure (a) before investing in serializing the schedule (a separate, harder increment).
"""
from __future__ import annotations
import os, sqlite3, hashlib, json, subprocess, contextlib, pathlib
from tinygrad.helpers import CACHEDB, getenv

BUNDLE_VERSION = 1

# ---------------------------------------------------------------------------------------------------------------------
# Codegen fingerprint: everything that can change the emitted binary. A prebuilt bundle is valid ONLY for a matching
# fingerprint; otherwise we fall back to normal cold compile (a SAFE miss). This closes the arch-only-key gap.
# ---------------------------------------------------------------------------------------------------------------------
def _comgr_version() -> str:
  try:
    import ctypes
    from tinygrad.runtime.autogen import comgr
    comgr.amd_comgr_get_version(ctypes.byref(mj := ctypes.c_uint64()), ctypes.byref(mn := ctypes.c_uint64()))
    return f"{mj.value}.{mn.value}"
  except Exception: return "unknown"

def _rocm_version() -> str:
  for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
    with contextlib.suppress(Exception):
      return pathlib.Path(p).read_text().strip()
  with contextlib.suppress(Exception):
    return subprocess.run(["hipconfig", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
  return "unknown"

def _kfd_version() -> str:
  with contextlib.suppress(Exception):
    return pathlib.Path("/sys/module/amdgpu/version").read_text().strip()
  return "unknown"

def _renderer_sha() -> str:
  # hash the AMD renderer + compiler source so a codegen change (that leaves the toolchain versions unchanged) still
  # invalidates the bundle. Falls back to the repo git sha.
  h = hashlib.sha256()
  for rel in ("tinygrad/renderer/cstyle.py", "tinygrad/runtime/support/compiler_amd.py", "tinygrad/renderer/__init__.py"):
    with contextlib.suppress(Exception):
      h.update(pathlib.Path(_repo_root() / rel).read_bytes())
  return h.hexdigest()[:16] if h.digest() else "unknown"

def _repo_root() -> pathlib.Path: return pathlib.Path(__file__).resolve().parents[2]

def codegen_fingerprint(arch:str|None=None) -> dict:
  arch = arch or getenv("AOT_ARCH", "gfx1100")
  fp = {"bundle_version": BUNDLE_VERSION, "arch": arch, "comgr": _comgr_version(), "rocm": _rocm_version(),
        "kfd": _kfd_version(), "renderer_sha": _renderer_sha()}
  fp["digest"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
  return fp

# ---------------------------------------------------------------------------------------------------------------------
# Bundle pack / unpack over the SQLite compile cache. Pure sqlite/file I/O -- no GPU. The bundle is a portable sqlite DB
# holding ONLY the compiled-kernel table(s) plus a `_aot_meta` row (fingerprint + golden digest + provenance).
# ---------------------------------------------------------------------------------------------------------------------
def _cache_tables(db:str) -> list[str]:
  with contextlib.closing(sqlite3.connect(db)) as c:
    return [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

def pack_bundle(out_path:str, tables:list[str]|None=None, golden:dict|None=None, src_db:str|None=None) -> dict:
  """Export the compiled-kernel table(s) from the live cache.db into a portable bundle DB at out_path.
  tables: which cachekey tables to include (default: all 'compile_hip_*'). golden: {'digest':..., 'meta':...} probe data."""
  src_db = src_db or CACHEDB
  fp = codegen_fingerprint()
  tables = tables or [t for t in _cache_tables(src_db) if t.startswith(("compile_hip_", "compile_hipcc_", "AMDGPU_"))]
  with contextlib.closing(sqlite3.connect(out_path)) as dst:
    dst.execute("CREATE TABLE IF NOT EXISTS _aot_meta (k TEXT PRIMARY KEY, v TEXT)")
    dst.execute("INSERT OR REPLACE INTO _aot_meta VALUES ('fingerprint', ?)", (json.dumps(fp),))
    dst.execute("INSERT OR REPLACE INTO _aot_meta VALUES ('golden', ?)", (json.dumps(golden or {}),))
    dst.execute("INSERT OR REPLACE INTO _aot_meta VALUES ('tables', ?)", (json.dumps(tables),))
    n = 0
    with contextlib.closing(sqlite3.connect(src_db)) as srcc:
      for t in tables:
        rows = srcc.execute(f"SELECT * FROM '{t}'").fetchall()
        cols = [d[0] for d in srcc.execute(f"SELECT * FROM '{t}' LIMIT 0").description]
        dst.execute(f"CREATE TABLE IF NOT EXISTS '{t}' ({', '.join(cols)}, PRIMARY KEY ({cols[0]}))")
        dst.executemany(f"INSERT OR REPLACE INTO '{t}' VALUES ({', '.join('?' for _ in cols)})", rows)
        n += len(rows)
    dst.commit()
  return {"path": out_path, "fingerprint": fp, "tables": tables, "kernels": n, "golden": bool(golden)}

def bundle_meta(bundle_path:str) -> dict:
  with contextlib.closing(sqlite3.connect(bundle_path)) as c:
    return {r[0]: json.loads(r[1]) for r in c.execute("SELECT k, v FROM _aot_meta").fetchall()}

def install_bundle(bundle_path:str, dst_db:str|None=None) -> int:
  """Copy the bundle's compiled-kernel rows into the live cache.db (INSERT OR IGNORE -> never clobber a locally-compiled
  binary). Returns rows installed. Caller MUST have already checked the fingerprint + passed the golden probe."""
  dst_db = dst_db or CACHEDB
  os.makedirs(os.path.dirname(dst_db), exist_ok=True)
  meta = bundle_meta(bundle_path); n = 0
  with contextlib.closing(sqlite3.connect(dst_db)) as dst, contextlib.closing(sqlite3.connect(bundle_path)) as src:
    for t in meta.get("tables", []):
      cols = [d[0] for d in src.execute(f"SELECT * FROM '{t}' LIMIT 0").description]
      dst.execute(f"CREATE TABLE IF NOT EXISTS '{t}' ({', '.join(cols)}, PRIMARY KEY ({cols[0]}))")
      rows = src.execute(f"SELECT * FROM '{t}'").fetchall()
      dst.executemany(f"INSERT OR IGNORE INTO '{t}' VALUES ({', '.join('?' for _ in cols)})", rows)
      n += len(rows)
    dst.commit()
  return n

# ---------------------------------------------------------------------------------------------------------------------
# Acceptance: fingerprint match is necessary-not-sufficient. The GOLDEN PROBE (run one fixed-seed forward, compare logits
# to the bundle's stored digest within tolerance) is the authority -- it catches the codegen input we FORGOT to
# fingerprint (the only defense against run-but-miscompute). golden_probe_fn is injected (needs the GPU/model) so this
# module stays importable/testable without a device.
# ---------------------------------------------------------------------------------------------------------------------
def fingerprint_matches(bundle_path:str) -> tuple[bool, dict, dict]:
  want = bundle_meta(bundle_path).get("fingerprint", {})
  have = codegen_fingerprint(arch=want.get("arch"))
  return (want.get("digest") == have.get("digest"), want, have)

def accept_bundle(bundle_path:str, golden_probe_fn=None, install:bool=True) -> dict:
  """Gate + optionally install a bundle. golden_probe_fn(golden_meta:dict)->(ok:bool, detail:dict) runs the model probe
  (GPU); if None, ONLY the fingerprint is checked and the caller is told the probe was skipped (unsafe for real use)."""
  ok_fp, want, have = fingerprint_matches(bundle_path)
  res = {"fingerprint_ok": ok_fp, "want": want, "have": have, "probe_ok": None, "installed": 0, "used": False}
  if not ok_fp: return {**res, "reason": "fingerprint mismatch -> cold compile (safe)"}
  if golden_probe_fn is not None:
    res["probe_ok"], res["probe_detail"] = golden_probe_fn(bundle_meta(bundle_path).get("golden", {}))
    if not res["probe_ok"]: return {**res, "reason": "golden probe FAILED -> reject bundle, cold compile (safe, loud)"}
  else:
    res["reason"] = "probe SKIPPED (no probe_fn) -- fingerprint-only, do NOT trust in production"
  if install: res["installed"] = install_bundle(bundle_path)
  res["used"] = True
  return res
