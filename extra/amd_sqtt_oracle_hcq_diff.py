#!/usr/bin/env python3
from __future__ import annotations

import collections, json, pathlib, shutil, subprocess, sys, time
from typing import Any

from extra.amd_scheduler_tooling_backend_execute import OUTDIR, attempt_sqtt_decode, run_capture

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = OUTDIR / "att_decoder_binary_probe.json"
ORACLE_OUT = OUTDIR / "att_oracle_capture.json"
BASELINE_OUT = OUTDIR / "hcq_sqtt_baseline_capture.json"
DIFF_OUT = OUTDIR / "att_hcq_setup_diff.json"
PATCH_OUT = OUTDIR / "hcq_sqtt_oracle_patch_result.json"
ATTR_OUT = OUTDIR / "q8_body_attribution_smoke.json"
RESULT_OUT = OUTDIR / "sqtt_oracle_hcq_diff_result.json"

def read_json(path: pathlib.Path, default: Any = None) -> Any:
  return json.loads(path.read_text()) if path.exists() else default

def write_json(path: pathlib.Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, indent=2) + "\n")

def run(cmd: list[str], timeout: int = 360) -> dict[str, Any]:
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"cmd": cmd, "returncode": cp.returncode, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": cp.stdout.splitlines()[-80:], "stderr_tail": cp.stderr.splitlines()[-80:]}
  except subprocess.TimeoutExpired as e:
    return {"cmd": cmd, "timeout": True, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": (e.stdout or "").splitlines()[-80:] if isinstance(e.stdout, str) else [],
            "stderr_tail": (e.stderr or "").splitlines()[-80:] if isinstance(e.stderr, str) else []}

def find_existing_oracle_dir() -> pathlib.Path | None:
  probe = read_json(PROBE, {})
  out = probe.get("rocprof_att", {}).get("output_dir")
  if out and (ROOT / out).exists(): return ROOT / out
  candidates = sorted((OUTDIR / "att_decoder_binary_probe_work").glob("rocprof_att_*"))
  return candidates[-1] if candidates else None

def summarize_oracle_dir(base: pathlib.Path) -> dict[str, Any]:
  files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in sorted(base.rglob("*")) if p.is_file()]
  dispatch_rows = []
  total_wave_files = 0
  total_wave_instruction_records = 0
  code_instruction_rows = 0
  top_mnemonics: collections.Counter[str] = collections.Counter()
  traced_cus: collections.Counter[int] = collections.Counter()
  for disp in sorted(base.glob("ui_output_agent_*")):
    if not disp.is_dir(): continue
    code_rows = []
    code_path = disp / "code.json"
    if code_path.exists():
      code = read_json(code_path, {})
      code_rows = code.get("code", []) if isinstance(code, dict) else []
      code_instruction_rows += len(code_rows)
      for row in code_rows:
        if isinstance(row, list) and row and isinstance(row[0], str):
          top_mnemonics[row[0].split()[0]] += 1
    wave_files = sorted(disp.glob("se*_wv*.json"))
    wave_inst = 0
    for wp in wave_files:
      data = read_json(wp, {})
      wave = data.get("wave", {}) if isinstance(data, dict) else {}
      wave_inst += len(wave.get("instructions", []))
      if wave.get("cu") is not None: traced_cus[int(wave["cu"])] += 1
    total_wave_files += len(wave_files)
    total_wave_instruction_records += wave_inst
    dispatch_rows.append({"dir": str(disp.relative_to(ROOT)), "code_rows": len(code_rows), "wave_files": len(wave_files),
                          "wave_instruction_records": wave_inst,
                          "first_code_row": code_rows[0][0] if code_rows and isinstance(code_rows[0], list) else None})
  return {
    "source_dir": str(base.relative_to(ROOT)),
    "files": files,
    "file_counts": {
      "att": sum(1 for f in files if f["path"].endswith(".att")),
      "code_object_out": sum(1 for f in files if f["path"].endswith(".out")),
      "json": sum(1 for f in files if f["path"].endswith(".json")),
      "csv": sum(1 for f in files if f["path"].endswith(".csv")),
    },
    "dispatches": dispatch_rows,
    "total_wave_files": total_wave_files,
    "total_wave_instruction_records": total_wave_instruction_records,
    "code_instruction_rows": code_instruction_rows,
    "top_mnemonics": top_mnemonics.most_common(20),
    "traced_cus": traced_cus.most_common(),
    "gate_pass": total_wave_instruction_records > 0 and code_instruction_rows > 0,
  }

def run_baseline_proof() -> dict[str, Any]:
  res = run([sys.executable, "extra/amd_sqtt_t1_body_mapping_proof.py"], timeout=480)
  data = read_json(OUTDIR / "t1_body_mapping_proof.json", {})
  out = {"command": res, "proof": data, "gate_pass": data.get("verdict") == "PASS_BODY_MAPPING"}
  write_json(BASELINE_OUT, out)
  return out

def patch_probe_configs() -> list[dict[str, str]]:
  return [
    {"SQTT_ORACLE_TARGET_CU": "1"},
    {"SQTT_ORACLE_TARGET_CU": "1", "SQTT_SIMD_SEL": "1"},
    {"SQTT_ORACLE_TARGET_CU": "1", "SQTT_RAW_MASK": str(0x30013), "SQTT_RAW_TOKEN_MASK": str(0xc080683), "SQTT_RAW_CTRL": str(0xa0423941)},
    {"SQTT_ORACLE_TARGET_CU": "1", "SQTT_SIMD_SEL": "1", "SQTT_RAW_MASK": str(0x30013), "SQTT_RAW_TOKEN_MASK": str(0xc080683), "SQTT_RAW_CTRL": str(0xa0423941)},
  ]

def run_patch_trials() -> dict[str, Any]:
  rows = []
  for env in patch_probe_configs():
    capture = run_capture(env_extra=env, timeout_s=420)
    decode = attempt_sqtt_decode(capture.get("capture") if capture.get("ok") else None)
    rows.append({
      "env": env,
      "capture_ok": capture.get("ok", False),
      "returncode": capture.get("returncode"),
      "sqtt_events": 0 if not capture.get("capture") else len(capture["capture"].get("sqtt", [])),
      "sqtt_bytes": 0 if not capture.get("capture") else sum(x.get("blob_bytes", 0) for x in capture["capture"].get("sqtt", [])),
      "decode_ok_count": decode.get("decode_ok_count"),
      "raw_body_packet_events_top20": decode.get("raw_body_packet_events_top20"),
      "body_instruction_events": decode.get("body_instruction_events"),
      "gate_pass": decode.get("gate_pass", False),
      "itrace_packet_tops": [
        {
          "idx": r.get("idx"),
          "raw_packet_counts_top": r.get("raw_packet_counts_top"),
          "mapped_instruction_counts_top": r.get("instruction_counts_top"),
          "error": r.get("error"),
        } for r in decode.get("rows", []) if r.get("itrace")
      ][:4],
      "stderr_tail": capture.get("stderr_tail", [])[-20:],
    })
  result = {
    "phase": "O3_env_gated_patch_trials",
    "patch": "SQTT_ORACLE_TARGET_CU forces COMPUTE_STATIC_THREAD_MGMT_SE* CU mask to the ROCprofiler-observed CU.",
    "rows": rows,
    "passing_envs": [r["env"] for r in rows if r["gate_pass"]],
    "gate_pass": any(r["gate_pass"] for r in rows),
  }
  write_json(PATCH_OUT, result)
  return result

def build_diff(oracle: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
  proof = baseline.get("proof", {})
  configs = proof.get("configs", [])
  baseline_body = max([c.get("body_instruction_events") or 0 for c in configs] or [0])
  baseline_raw_body = max([c.get("raw_body_packet_events_top20") or 0 for c in configs] or [0])
  observed_cus = [cu for cu, _ in oracle.get("traced_cus", [])]
  diff_rows = [
    {
      "name": "decoder availability",
      "state": "closed",
      "evidence": "ROCprofiler ATT oracle emits decoded code/wave JSON.",
    },
    {
      "name": "instruction records",
      "state": "oracle_pass_hcq_fail",
      "evidence": f"oracle wave_instruction_records={oracle.get('total_wave_instruction_records')}; HCQ body_instruction_events={baseline_body}",
    },
    {
      "name": "target CU selection",
      "state": "observable_patchable" if observed_cus else "not_observable",
      "evidence": f"oracle traced CUs={oracle.get('traced_cus')}; tinygrad uses static-thread masks derived from global size and first-WGP assumptions",
      "patch": "SQTT_ORACLE_TARGET_CU=1",
    },
    {
      "name": "MASK/TOKEN/CTRL raw values",
      "state": "already_tested",
      "evidence": "T1b raw-register transplant changed bytes but kept raw/mapped body packets at zero.",
    },
    {
      "name": "full ROCprofiler command service",
      "state": "not_observable_from_ATT_JSON",
      "evidence": "ATT UI output proves decoded packets but does not serialize the full PM4/AQL command ordering used by rocprofv3.",
    },
  ]
  result = {
    "phase": "O2_command_setup_diff",
    "oracle_gate_pass": oracle.get("gate_pass", False),
    "hcq_baseline_verdict": proof.get("verdict"),
    "hcq_baseline_max_raw_body": baseline_raw_body,
    "hcq_baseline_max_body": baseline_body,
    "diff_rows": diff_rows,
    "new_patchable_differences": [r for r in diff_rows if r["state"] == "observable_patchable"],
    "gate_pass": any(r["state"] == "observable_patchable" for r in diff_rows),
  }
  write_json(DIFF_OUT, result)
  return result

def attribution_smoke(patch: dict[str, Any]) -> dict[str, Any]:
  result = {
    "phase": "O4_attribution_usability_check",
    "attempted": patch.get("gate_pass", False),
    "gate_pass": False,
    "verdict": "NOT_RUN_NO_BODY_PACKETS" if not patch.get("gate_pass") else "BODY_PACKETS_PRESENT_NEEDS_FEATURE_JOIN",
    "reason": "O3 did not produce body instruction packets." if not patch.get("gate_pass") else "body packets present; feature join deferred",
  }
  write_json(ATTR_OUT, result)
  return result

def main() -> int:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  oracle_dir = find_existing_oracle_dir()
  if oracle_dir is None:
    raise SystemExit("no existing ATT oracle output; rerun extra/amd_att_decoder_d0d1_binary_probe.py")
  oracle = summarize_oracle_dir(oracle_dir)
  write_json(ORACLE_OUT, oracle)
  baseline = run_baseline_proof()
  diff = build_diff(oracle, baseline)
  patch = run_patch_trials() if diff.get("gate_pass") else {"gate_pass": False, "rows": [], "skipped": "no patchable diff"}
  smoke = attribution_smoke(patch)
  if patch.get("gate_pass"): verdict = "PASS_BODY_ATTRIBUTION"
  elif not diff.get("gate_pass"): verdict = "KILL_NO_PATCHABLE_DIFF"
  else: verdict = "KILL_PATCH_NO_BODY"
  result = {
    "date": "2026-06-19",
    "phase": "O0_to_O5_sqtt_oracle_hcq_diff",
    "artifacts": {
      "oracle": str(ORACLE_OUT.relative_to(ROOT)),
      "baseline": str(BASELINE_OUT.relative_to(ROOT)),
      "diff": str(DIFF_OUT.relative_to(ROOT)),
      "patch": str(PATCH_OUT.relative_to(ROOT)),
      "attribution": str(ATTR_OUT.relative_to(ROOT)),
    },
    "gates": {
      "O0_oracle": oracle.get("gate_pass", False),
      "O1_baseline_reproduced": baseline.get("proof", {}).get("verdict") == "NO_LOCAL_REGISTER_KNOB_BODY_MAPPING",
      "O2_patchable_diff": diff.get("gate_pass", False),
      "O3_body_packets": patch.get("gate_pass", False),
      "O4_attribution": smoke.get("gate_pass", False),
    },
    "verdict": verdict,
    "decision": (
      "Use the passing env-gated SQTT setup for q8 attribution."
      if verdict == "PASS_BODY_ATTRIBUTION" else
      "The external oracle works, but the bounded HCQ patch did not produce body packets. Close Track T as a small tooling patch; "
      "further work is a broader ROCprofiler command-service integration project."
    ),
  }
  write_json(RESULT_OUT, result)
  print(json.dumps({"out": str(RESULT_OUT.relative_to(ROOT)), "verdict": verdict, "gates": result["gates"]}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
