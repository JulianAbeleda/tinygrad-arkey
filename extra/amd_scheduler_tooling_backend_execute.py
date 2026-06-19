#!/usr/bin/env python3
from __future__ import annotations

import base64, hashlib, json, os, pathlib, runpy, subprocess, sys, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
Q8_SCRIPT = ROOT / "extra/q8_ffn_asm_gateup_full.py"

def read_json(path: pathlib.Path, default: Any = None) -> Any:
  return json.loads(path.read_text()) if path.exists() else default

def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()

CHILD = r"""
import base64, json, pathlib, runpy, sys, traceback
from tinygrad.device import Buffer, Compiled, Device

out = pathlib.Path(sys.argv[1])
q8_script = pathlib.Path(sys.argv[2])
sys.argv = [str(q8_script), "--warmups", "0", "--iters", "1", "--out", str(out)]
status = {"ok": True, "error": None}
try:
  runpy.run_path(str(q8_script), run_name="__main__")
except SystemExit as e:
  status["ok"] = e.code in (0, None)
  status["error"] = None if status["ok"] else f"SystemExit({e.code})"
except Exception:
  status["ok"] = False
  status["error"] = traceback.format_exc(limit=12)

try:
  for d in list(Device._opened_devices): Device[d].synchronize()
  for d in list(Device._opened_devices): Device[d]._at_profile_finalize()
except Exception:
  status["profile_finalize_error"] = traceback.format_exc(limit=12)

events = Compiled.profile_events + Buffer.profile_events
dev_targets = {}
programs = []
pmc = []
sqtt = []
ranges = []
for idx, ev in enumerate(events):
  tn = type(ev).__name__
  if tn == "ProfileDeviceEvent" and getattr(ev, "props", None):
    dev_targets[ev.device] = "gfx%d" % (ev.props["gfx_target_version"] // 1000)
  elif tn == "ProfileProgramEvent":
    lib = ev.lib or b""
    programs.append({"idx": idx, "tag": ev.tag, "device": ev.device, "name": ev.name, "base": ev.base,
                     "lib_b64": base64.b64encode(lib).decode(), "lib_bytes": len(lib)})
  elif tn == "ProfileRangeEvent":
    ranges.append({"idx": idx, "device": ev.device, "name": str(ev.name), "st": str(ev.st),
                   "en": None if ev.en is None else str(ev.en)})
  elif tn == "ProfilePMCEvent":
    pmc.append({"idx": idx, "device": ev.device, "kern": ev.kern, "exec_tag": ev.exec_tag,
                "blob_b64": base64.b64encode(ev.blob).decode(), "blob_bytes": len(ev.blob),
                "sample_layout": [{"name": s.name, "block": s.block, "size": s.size, "off": s.off} for s in ev.sched]})
  elif tn == "ProfileSQTTEvent":
    sqtt.append({"idx": idx, "device": ev.device, "kern": ev.kern, "se": ev.se, "exec_tag": ev.exec_tag,
                 "itrace": ev.itrace, "blob_b64": base64.b64encode(ev.blob).decode(), "blob_bytes": len(ev.blob)})

print("CAPTURE_JSON=" + json.dumps({"status": status, "dev_targets": dev_targets, "programs": programs,
                                    "pmc": pmc, "sqtt": sqtt, "ranges": ranges}, sort_keys=True))
if not status["ok"]: sys.exit(2)
"""

def run_capture(timeout_s: int = 360) -> dict[str, Any]:
  env = os.environ.copy()
  env.update({
    "PYTHONPATH": str(ROOT), "PROFILE": "1", "PMC": "1", "SQTT": "1", "VIZ": "0", "DEBUG": "0",
    "SQTT_BUFFER_SIZE": "16", "SQTT_LIMIT_SE": "1", "SQTT_ITRACE_SE_MASK": "1", "SQTT_TOKEN_EXCLUDE": "0",
  })
  timing_out = OUTDIR / "t0_capture_q8_gateup_full.json"
  cmd = [sys.executable, "-c", CHILD, str(timing_out), str(Q8_SCRIPT)]
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=timeout_s, check=False)
    parsed = None
    for line in cp.stdout.splitlines():
      if line.startswith("CAPTURE_JSON="): parsed = json.loads(line.split("=", 1)[1])
    return {"ok": cp.returncode == 0 and parsed is not None, "returncode": cp.returncode,
            "elapsed_s": round(time.perf_counter() - t0, 3), "timing_out": str(timing_out.relative_to(ROOT)),
            "timing_exists": timing_out.exists(), "capture": parsed,
            "stdout_tail": cp.stdout.splitlines()[-30:], "stderr_tail": cp.stderr.splitlines()[-30:]}
  except subprocess.TimeoutExpired as e:
    return {"ok": False, "timeout": True, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": (e.stdout or "").splitlines()[-30:] if isinstance(e.stdout, str) else [],
            "stderr_tail": (e.stderr or "").splitlines()[-30:] if isinstance(e.stderr, str) else []}

def summarize_capture(capture: dict[str, Any] | None) -> dict[str, Any]:
  if capture is None: return {"available": False}
  programs = capture.get("programs", [])
  pmcs = capture.get("pmc", [])
  sqtts = capture.get("sqtt", [])
  return {
    "available": True,
    "program_count": len(programs),
    "pmc_count": len(pmcs),
    "sqtt_count": len(sqtts),
    "sqtt_itrace_count": sum(1 for e in sqtts if e.get("itrace")),
    "sqtt_total_blob_bytes": sum(e.get("blob_bytes", 0) for e in sqtts),
    "pmc_total_blob_bytes": sum(e.get("blob_bytes", 0) for e in pmcs),
    "programs": [{"tag": p.get("tag"), "name": p.get("name"), "device": p.get("device"),
                  "lib_bytes": p.get("lib_bytes"), "lib_sha256": sha256(base64.b64decode(p.get("lib_b64", "")))}
                 for p in programs],
  }

def attempt_sqtt_decode(capture: dict[str, Any] | None) -> dict[str, Any]:
  if capture is None: return {"attempted": False, "reason": "no capture"}
  from tinygrad.renderer.amd import sqtt as sqtt_decoder
  programs = {p["tag"]: p for p in capture.get("programs", []) if p.get("tag") is not None}
  rows = []
  for ev in capture.get("sqtt", []):
    blob = base64.b64decode(ev.get("blob_b64", ""))
    prg = programs.get(ev.get("kern"))
    row = {"idx": ev.get("idx"), "kern": ev.get("kern"), "se": ev.get("se"), "itrace": ev.get("itrace"),
           "blob_bytes": len(blob), "program": None if prg is None else prg.get("name")}
    if len(blob) == 0:
      row.update({"decode_ok": False, "error": "empty_blob"})
    else:
      try:
        pkt_counts: dict[str, int] = {}
        inst_counts: dict[str, int] = {}
        mapped = 0
        if prg is not None and ev.get("itrace"):
          lib = base64.b64decode(prg.get("lib_b64", ""))
          target = capture.get("dev_targets", {}).get(prg.get("device"), "")
          iterator = sqtt_decoder.map_insts(blob, lib, target)
        else:
          iterator = ((p, None) for p in sqtt_decoder.decode(blob))
        for pkt, inst in iterator:
          pkt_counts[type(pkt).__name__] = pkt_counts.get(type(pkt).__name__, 0) + 1
          if inst is not None:
            mapped += 1
            op = getattr(inst.inst, "op_name", type(inst.inst).__name__)
            inst_counts[op] = inst_counts.get(op, 0) + 1
        row.update({"decode_ok": True, "mapped_instruction_events": mapped,
                    "packet_counts_top": sorted(pkt_counts.items(), key=lambda kv: kv[1], reverse=True)[:20],
                    "instruction_counts_top": sorted(inst_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]})
      except Exception as exc:
        row.update({"decode_ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=4)})
    rows.append(row)
  ok_rows = [r for r in rows if r.get("decode_ok")]
  mapped = sum(r.get("mapped_instruction_events", 0) for r in ok_rows)
  body_mapped = 0
  for r in ok_rows:
    for name, count in r.get("instruction_counts_top", []):
      if name != "S_ENDPGM": body_mapped += count
  return {
    "attempted": True,
    "rows": rows,
    "decode_ok_count": len(ok_rows),
    "decode_fail_count": len(rows) - len(ok_rows),
    "mapped_instruction_events": mapped,
    "body_instruction_events": body_mapped,
    "structural_decode_ok": mapped > 0 and len(ok_rows) > 0,
    "gate_pass": body_mapped > 0,
    "gate_note": "T1 requires q8 body instruction mapping; S_ENDPGM-only mapping is structural decode, not attribution.",
  }

def parse_pmc(capture: dict[str, Any] | None) -> dict[str, Any]:
  if capture is None: return {"attempted": False, "reason": "no capture"}
  rows = []
  for ev in capture.get("pmc", []):
    blob = base64.b64decode(ev.get("blob_b64", ""))
    samples = []
    for s in ev.get("sample_layout", []):
      chunk = blob[s["off"]:s["off"] + s["size"]]
      words = [int.from_bytes(chunk[i:i+8], "little") for i in range(0, len(chunk) - len(chunk)%8, 8)]
      samples.append({"name": s["name"], "block": s["block"], "size": s["size"], "off": s["off"],
                      "nonzero_words": sum(1 for w in words if w != 0), "sum_u64": sum(words), "max_u64": max(words) if words else 0})
    by_name = {s["name"]: s for s in samples}
    gl2_hit, gl2_miss = by_name.get("GL2C_HIT", {}).get("sum_u64", 0), by_name.get("GL2C_MISS", {}).get("sum_u64", 0)
    rows.append({"idx": ev.get("idx"), "kern": ev.get("kern"), "exec_tag": ev.get("exec_tag"),
                 "blob_bytes": len(blob), "samples": samples,
                 "derived": {
                   "gl2_hit_rate": round(gl2_hit / (gl2_hit + gl2_miss), 6) if (gl2_hit + gl2_miss) else None,
                   "lds_bank_conflict_sum": by_name.get("SQC_LDS_BANK_CONFLICT", {}).get("sum_u64", 0),
                   "valu_to_salu_inst_ratio": round(by_name.get("SQ_INSTS_VALU", {}).get("sum_u64", 0) / max(1, by_name.get("SQ_INSTS_SALU", {}).get("sum_u64", 0)), 6),
                 }})
  return {"attempted": True, "rows": rows, "row_count": len(rows), "gate_pass": len(rows) > 0}

def join_timeline(capture: dict[str, Any] | None, pmc_summary: dict[str, Any], sqtt_summary: dict[str, Any]) -> dict[str, Any]:
  if capture is None: return {"attempted": False, "reason": "no capture"}
  programs = {p["tag"]: p for p in capture.get("programs", []) if p.get("tag") is not None}
  rows = []
  for ev in capture.get("pmc", []):
    prg = programs.get(ev.get("kern"), {})
    rows.append({"kind": "pmc", "program": prg.get("name"), "code_sha256": sha256(base64.b64decode(prg.get("lib_b64", ""))) if prg else None,
                 "kern": ev.get("kern"), "exec_tag": ev.get("exec_tag"), "blob_bytes": ev.get("blob_bytes")})
  for ev in capture.get("sqtt", []):
    prg = programs.get(ev.get("kern"), {})
    rows.append({"kind": "sqtt", "program": prg.get("name"), "code_sha256": sha256(base64.b64decode(prg.get("lib_b64", ""))) if prg else None,
                 "kern": ev.get("kern"), "exec_tag": ev.get("exec_tag"), "se": ev.get("se"), "itrace": ev.get("itrace"),
                 "blob_bytes": ev.get("blob_bytes")})
  return {"attempted": True, "row_count": len(rows), "rows": rows,
          "has_level4_pmc": pmc_summary.get("gate_pass", False),
          "has_decoded_sqtt": sqtt_summary.get("gate_pass", False)}

def build_b0_oracle_suite() -> dict[str, Any]:
  q8_contract = read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/oracle_contract.json", {})
  n1 = read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/n1_attribution.json", {})
  tensile_shape = read_json(ROOT / "bench/qk-tensile-extraction/shape_matrix.json", {})
  tensile_codegen = read_json(ROOT / "bench/qk-tensile-extraction/codegen_oracle.json", {})
  matrix = read_json(ROOT / "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json", {})
  rows_by_role = {r.get("role"): r for r in tensile_shape.get("rows", [])}
  return {
    "date": "2026-06-19",
    "phase": "B0_oracle_suite",
    "oracles": [
      {
        "name": "q8_decode_gate_up_consumer",
        "target": "hipcc/LLD q8 MMVQ gate/up",
        "tinygrad_baseline_us": q8_contract.get("known_timings_us", {}).get("tinygrad_asm_gateup_full"),
        "oracle_us": q8_contract.get("known_timings_us", {}).get("hipcc_lld_gateup_current_loader"),
        "quality_gate": "q8 proxy max_abs <=2e-3, W==D >=3%, dNLL <=0.01 if routed",
        "launch_contract": q8_contract.get("launch_contract"),
        "resource_metadata": q8_contract.get("resource_contract"),
        "schedule_features": q8_contract.get("instruction_contract"),
        "n1_verdict": n1.get("verdict"),
      },
      {
        "name": "prefill_tensile_ffn_gate_up",
        "target": "Tensile MT128x128x16 ffn_gate/up",
        "tinygrad_baseline_tflops": rows_by_role.get("ffn_gate_up", {}).get("tinygrad_tflops"),
        "oracle_tflops": rows_by_role.get("ffn_gate_up", {}).get("median_tflops"),
        "speedup_vs_tinygrad": rows_by_role.get("ffn_gate_up", {}).get("speedup_vs_tinygrad"),
        "launch_contract": {k: rows_by_role.get("ffn_gate_up", {}).get(k) for k in ("m", "n", "k", "global_size", "local_size", "kernarg_size", "workspace")},
        "oracle_source": "bench/qk-tensile-extraction/shape_matrix.json",
        "shape_matrix_row": rows_by_role.get("ffn_gate_up"),
        "codegen_oracle_summary": tensile_codegen,
        "quality_gate": "fp16 oracle rel_err <=1e-3, pp512 clock-controlled if routed",
      },
      {
        "name": "prefill_tensile_ffn_down",
        "target": "Tensile StreamK ffn_down",
        "tinygrad_baseline_tflops": rows_by_role.get("ffn_down", {}).get("tinygrad_tflops"),
        "oracle_tflops": rows_by_role.get("ffn_down", {}).get("median_tflops"),
        "speedup_vs_tinygrad": rows_by_role.get("ffn_down", {}).get("speedup_vs_tinygrad"),
        "launch_contract": {k: rows_by_role.get("ffn_down", {}).get(k) for k in ("m", "n", "k", "global_size", "local_size", "kernarg_size", "workspace", "streamk")},
        "oracle_source": "bench/qk-tensile-extraction/shape_matrix.json",
        "quality_gate": "fp16 oracle rel_err <=1e-3, pp512 clock-controlled if routed",
      },
      {
        "name": "small_smoke_kernel",
        "target": "HCQ attribution / comparable HIP control smoke",
        "oracle_source": "bench/qk-hcq-attribution/result.json + bench/qk-pmu-observability/result.json",
        "quality_gate": "correctness smoke plus visible program metadata",
      },
    ],
    "matrix_pointer": matrix.get("verdict", "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json"),
    "gate_pass": bool(q8_contract and tensile_shape and tensile_codegen),
  }

def main() -> int:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  existing = {
    "pmu_sqtt_evidence": read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json", {}),
    "n1_attribution": read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/n1_attribution.json", {}),
    "hcq_attribution": read_json(ROOT / "bench/qk-hcq-attribution/result.json", {}),
    "pmu_observability": read_json(ROOT / "bench/qk-pmu-observability/result.json", {}),
  }
  capture_run = run_capture()
  capture = capture_run.get("capture") if capture_run.get("ok") else None
  (OUTDIR / "t0_capture_blobs.json").write_text(json.dumps(capture_run, indent=2) + "\n")
  t0 = {
    "date": "2026-06-19",
    "phase": "T0_evidence_inventory",
    "existing_inputs_present": {k: bool(v) for k, v in existing.items()},
    "capture_summary": summarize_capture(capture),
    "gate_pass": capture is not None and len(capture.get("sqtt", [])) > 0 and len(capture.get("programs", [])) > 0,
    "stop": None if capture is not None else "capture failed; cannot replay blobs offline",
  }
  t1 = attempt_sqtt_decode(capture)
  t2 = parse_pmc(capture)
  t3 = join_timeline(capture, t2, t1)
  t4 = {
    "phase": "T4_attribution_verdict",
    "gate_pass": False,
    "verdict": "T_PARTIAL_PMC_STRUCTURAL_SQTT_NO_BODY_ATTRIBUTION" if not t1.get("gate_pass") else "T_PARTIAL_BODY_SQTT_NO_BOUNDED_FEATURE",
    "decision": "Do not claim scheduler/resource feature attribution yet; continue with B0 oracle suite only.",
    "reason": "SQTT replay is structurally decodable, but it does not map q8 body instructions; PMCs are parsed but not enough to assign a >=30us feature.",
  }
  b0 = build_b0_oracle_suite()
  t_ready = t0["gate_pass"] and t2.get("gate_pass") and t3.get("attempted")
  result = {
    "date": "2026-06-19",
    "verdict": "TRACK_T_PARTIAL_NO_FEATURE_B0_PASS" if t_ready and b0.get("gate_pass") else "TRACK_T_STARTED_B0_INCOMPLETE",
    "track_t": {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4},
    "track_b": {"b0": b0, "next": "Do not start B1/B2 until T4 attributes a feature or backend funding is explicitly accepted."},
  }
  (OUTDIR / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str((OUTDIR / "execution.json").relative_to(ROOT)), "verdict": result["verdict"],
                    "t0_gate": t0["gate_pass"], "t1_gate": t1.get("gate_pass"), "t2_gate": t2.get("gate_pass"),
                    "b0_gate": b0.get("gate_pass")}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
