#!/usr/bin/env python3
"""Shared evaluator-contract helper for performance-claiming harnesses.

Authority: structure/Development/performance-primitive-research-principles.md, section "Harnesses Are Performance
Primitives Too" -- grounded in MLPerf Inference (representative workload + quality + comparable scenario +
reproducible submission), the SPEC RG reproducible-evaluation methodology (preserve artifacts + environment; a
complex system is not one timing number), Ansor/TVM (generate -> measure-on-real-hw -> feedback), and Triton
(expose hardware-relevant tiling/dataflow knobs).

RULE (the contract): a performance claim is valid only when the artifact captures
  workload | comparator | correctness/quality | timing authority | environment | repeats/noise | candidate
  metadata | promotion policy.

This module CENTRALIZES that contract (one authority point, per the repo's core principles) so every harness can
`stamp()` its artifact with provenance + a structured comparator block + a self-audit of the 13 required fields,
instead of each script re-deriving (and forgetting) them. It is import-safe with NO GPU / tinygrad dependency.

Usage in a harness:
    from extra.qk.harness_contract import stamp, repro_band
    band = repro_band(samples_us)                       # {n,median,min,max,mean,spread_pct,mad}
    art = {... existing fields ..., "repro_band": band_by_ctx}
    art = stamp(art, comparator_id="gqa_coop_vec",
                comparator_why="shipped default decode-attention primitive; the reigning local A/B winner",
                timing_authority="local throughput proxy (back-to-back perf_counter, clock-pinned) -- NOT W==D",
                ledger_links=["docs/...result.md", "BoltBeam refutation ledger#<id>"])
    # art now carries art["harness_contract"] = {provenance, comparator, timing_authority, ledger_links,
    #                                             contract_audit:{present,missing,conformance}}
"""
from __future__ import annotations
import os, pathlib, statistics, subprocess, sys, time

from extra.qk.paths import DEFAULT_MODEL_GGUF  # qk_paths imports only pathlib -> safe for light tooling

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The current decode-attention comparator / reigning winner -- a light, importable mirror of the shipped
# extra/qk/flash_decode.py:FLASH_DECODE_DEFAULT_VARIANT (which imports tinygrad, so cannot be imported by light
# tooling/harnesses). test/unit/test_comparator_ssot.py asserts the two never drift. If the shipped default changes,
# update BOTH (the test will fail until they agree) so every A/B compares against the real current winner.
DECODE_COMPARATOR = "gqa_coop_vec"

# Default model + the single child-subprocess env builder (the env-ordering invariant lives here once).
DEFAULT_MODEL = DEFAULT_MODEL_GGUF

def qk_subprocess_env(extra: dict | None = None) -> dict:
  """Build the env for a spawned QK eval subprocess: AMD/JIT/PYTHONPATH/QK_MODEL (+ caller overrides). The single
  source for 'how to launch a QK child' so the env-ordering invariant + model default cannot drift across launchers.
  (Declaratively named: this builds the *QK subprocess* env; there is no longer a same-named twin in generate.py.)"""
  e = os.environ.copy()
  e.setdefault("DEV", "AMD"); e.setdefault("JIT", "1")
  e["PYTHONPATH"] = str(ROOT)
  e["QK_MODEL"] = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  for k, v in (extra or {}).items(): e[str(k)] = str(v)
  return e

def csv_ints(raw:str) -> tuple[int, ...]:
  vals = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
  if not vals: raise ValueError("expected at least one comma-separated integer")
  return vals

# The 13 fields a valid benchmark artifact must record (the principle doc's enumerated contract).
# This is a generic per-harness audit vocabulary, not a local promotion evaluator. BoltBeam owns candidate
# promotion policy; tinygrad harnesses only stamp workload/comparator/provenance evidence.
CONTRACT_FIELDS = (
  "workload_shape_and_context", "candidate_id_and_class", "comparator_id_and_why_winner",
  "exact_command_and_env", "git_commit_and_dirty", "hardware_and_clock_state", "warmup_compile_handling",
  "repeats_median_spread_band", "correctness_or_quality_gate", "local_vs_wd_timing_authority",
  "pass_fail_threshold", "final_verdict_and_stop_reason", "ledger_refutation_links",
)

# ---- provenance -----------------------------------------------------------------------------------------------
def _git(*a) -> str:
  try: return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"

def git_commit() -> str: return _git("rev-parse", "HEAD")
def git_short() -> str: return _git("rev-parse", "--short", "HEAD")
def dirty_tree() -> bool: return bool(_git("status", "--short"))

def perf_state() -> str:
  from extra.qk.clock_pin import read_perf_state  # the GPU perf-state boundary (single source for the sysfs path)
  return read_perf_state()

def hardware() -> str: return "RX 7900 XTX / gfx1100"

def provenance(env_keys=("DEV", "JIT", "PROFILE", "FLASH_VARIANT", "FLASH_L")) -> dict:
  """Capture exact command + env subset + git + hardware + perf-state + timestamp (contract fields 4,5,6)."""
  return {
    "command": "DEV=AMD JIT=1 python3 " + " ".join(sys.argv[0:1] + sys.argv[1:]),
    "env": {k: os.environ[k] for k in env_keys if k in os.environ},
    "git_commit": git_commit(), "git_short": git_short(), "dirty_tree": dirty_tree(),
    "hardware": hardware(), "perf_state": perf_state(), "timestamp": time.strftime("%Y%m%dT%H%M%S"),
  }

# ---- reproducibility band (contract field 8: repeats, median, spread, noise band) -------------------------------
def repro_band(samples) -> dict:
  """median + spread/noise band from repeated timings. The single most systemically-missing contract field --
  a bare median cannot tell a real 1.05x from host jitter. spread_pct = (max-min)/median*100."""
  xs = [float(x) for x in samples]
  if not xs: return {"n": 0, "median": None, "min": None, "max": None, "mean": None, "spread_pct": None, "mad": None}
  med = statistics.median(xs)
  return {"n": len(xs), "median": round(med, 3), "min": round(min(xs), 3), "max": round(max(xs), 3),
          "mean": round(statistics.fmean(xs), 3), "spread_pct": round((max(xs) - min(xs)) / med * 100, 2) if med else None,
          "mad": round(statistics.median([abs(x - med) for x in xs]), 3)}

# ---- the ONE per-call timing loop (do not clone this) -----------------------------------------------------------
def time_fn(fn, n: int = 200, warmup: int = 0, device: str = "AMD") -> list[float]:
  """Per-call wall times (us) for a synced GPU callable. Pair with repro_band() for the noise band, or
  statistics.median() for a point estimate. The ONE timing loop -- ~17 harnesses used to clone this
  synchronize()+perf_counter()+median shape; route through here instead. Returns the sample LIST (not a bare
  median) so it composes with repro_band(). The tinygrad import is LAZY so this module stays importable before
  tinygrad on the env-ordering-sensitive paths."""
  from tinygrad import Device                       # lazy: keep harness_contract importable pre-tinygrad
  dev = Device[device]
  for _ in range(warmup): fn(); dev.synchronize()
  dev.synchronize(); ts = []
  for _ in range(n):
    t0 = time.perf_counter(); fn(); dev.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
  return ts

# ---- contract self-audit (does THIS artifact capture the 13 fields?) --------------------------------------------
def _has(d: dict, *keys) -> bool:
  return any(k in d for k in keys)

def contract_audit(art: dict) -> dict:
  """Heuristically detect which of the 13 contract fields the artifact records. Used both as a self-check stamped
  into a harness artifact AND by the evaluator to flag a non-conforming child artifact it consumes."""
  hc = art.get("harness_contract", {})
  prov = hc.get("provenance", {}); comp = hc.get("comparator", {})
  checks = {
    "workload_shape_and_context": _has(art, "head_dim", "q_heads", "ctx", "ctx_fixed", "contexts", "workload", "L"),
    "candidate_id_and_class": _has(art, "candidate_id", "candidate", "primitive_class") and _has(art, "family", "primitive_class", "phase"),
    "comparator_id_and_why_winner": bool(comp.get("id") and comp.get("why_current_winner")) or (_has(art, "comparator") and _has(art, "comparator_why", "comparator_authority")),
    "exact_command_and_env": bool(prov.get("command")) or _has(art, "command", "exact_command"),
    "git_commit_and_dirty": bool(prov.get("git_commit")) or _has(art, "git_commit", "commit"),
    "hardware_and_clock_state": (bool(prov.get("hardware")) or _has(art, "hardware")) and (bool(prov.get("perf_state")) or _has(art, "clock_pin", "perf_state", "clock_state")),
    "warmup_compile_handling": _has(art, "warmups", "warmup", "compile_handling"),
    "repeats_median_spread_band": _has(art, "repro_band", "band_pct", "spread_pct") or _has(art, "repeats"),
    "correctness_or_quality_gate": _has(art, "correctness_rel_rmse", "rel_rmse", "dnll", "dNLL", "correctness", "quality_gate"),
    "local_vs_wd_timing_authority": bool(hc.get("timing_authority")) or _has(art, "timing_authority", "method"),
    "pass_fail_threshold": _has(art, "first_gate_pass", "threshold", "gate", "pass_fail_threshold"),
    "final_verdict_and_stop_reason": _has(art, "verdict", "decision") and _has(art, "stop_reason", "verdict", "decision"),
    "ledger_refutation_links": bool(hc.get("ledger_links")) or _has(art, "ledger_links", "linked_ledger_entry", "refutation"),
  }
  present = [k for k, v in checks.items() if v]; missing = [k for k, v in checks.items() if not v]
  n = len(present)
  conformance = "CONFORMS" if n == len(CONTRACT_FIELDS) else ("PARTIAL" if n >= 8 else "WEAK")
  return {"present": present, "missing": missing, "n_present": n, "n_total": len(CONTRACT_FIELDS), "conformance": conformance}

# ---- stamp: merge provenance + comparator + timing-authority + ledger + self-audit (additive, non-destructive) --
def stamp(art: dict, comparator_id: str, comparator_why: str, timing_authority: str,
          ledger_links=None, is_current_winner: bool = True, env_keys=None) -> dict:
  """Stamp the full contract envelope into `art` (additive: existing keys are preserved so downstream readers --
  still find best_speedup_vs_coop/first_gate_pass/results)."""
  prov = provenance(env_keys) if env_keys else provenance()
  art["harness_contract"] = {
    "contract_version": "harness_evaluator_contract_v1",
    "provenance": prov,
    "comparator": {"id": comparator_id, "why_current_winner": comparator_why, "is_current_winner": is_current_winner},
    "timing_authority": timing_authority,
    "ledger_links": list(ledger_links or []),
  }
  art["harness_contract"]["contract_audit"] = contract_audit(art)
  return art

if __name__ == "__main__":
  # self-test (no GPU): a thin artifact -> WEAK; a stamped full artifact -> CONFORMS
  thin = {"speedup": 1.2}
  print("thin:", contract_audit(thin)["conformance"], contract_audit(thin)["missing"][:3], "...")
  full = {"head_dim": 128, "ctx_fixed": 1024, "candidate_id": "x", "family": "attention_split",
          "warmups": 8, "repro_band": {1024: {}}, "correctness_rel_rmse": 5e-4, "first_gate_pass": False,
          "verdict": "FAIL_LOCAL_AB", "stop_reason": "local < 1.05"}
  full = stamp(full, DECODE_COMPARATOR, "shipped default decode winner", "local throughput proxy -- NOT W==D",
               ledger_links=["docs/x.md"])
  a = full["harness_contract"]["contract_audit"]
  print("full:", a["conformance"], f"{a['n_present']}/{a['n_total']}", "missing:", a["missing"])
  assert a["conformance"] == "CONFORMS", a["missing"]
  print("SELFTEST_PASS")
