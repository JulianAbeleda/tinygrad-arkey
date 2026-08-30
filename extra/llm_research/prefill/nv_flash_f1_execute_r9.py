#!/usr/bin/env python3
"""Fail-closed F1.2 exact-population replay driver.

The driver deliberately refuses to substitute the isolated full-tensor
fixture for graph-owned call slices or to relabel a missing installed route.
When an executable route adapter is supplied, it is invoked once per manifest
entry in control_0/candidate_1/control_2 order and all payloads are retained
outside timing.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, fcntl
from pathlib import Path

def digest(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("manifest")
  ap.add_argument("--out", required=True)
  ap.add_argument("--route-adapter", help="executable adapter accepting --arm --call-json")
  a = ap.parse_args()
  m = json.loads(Path(a.manifest).read_text())
  calls = m.get("calls", [])
  result = {"schema":"tinygrad.nv_prefill_flash_f1_2_r9.v1", "packet":"F1.2",
    "status":"STOP", "manifest":str(Path(a.manifest)), "manifest_sha256":digest(calls),
    "exact_population":36, "census":{"predicted":{"installed":36,"candidate":36},"observed":None},
    "arms":{"control_0":[],"candidate_1":[],"control_2":[]}, "full_outputs":[],
    "decision":"STOP: exact 36-call route adapter was not supplied; no installed or candidate launch is inferred",
    "next_packet":None}
  if len(calls) != 36:
    result["decision"] = f"STOP: manifest contains {len(calls)} calls; exact 36 required"
  elif a.route_adapter:
    lock_path = Path(a.out).with_suffix(".lock")
    with lock_path.open("w") as lock:
      fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
      for arm in ("control_0", "candidate_1", "control_2"):
        for i, call in enumerate(calls):
          payload = Path(a.out).with_suffix(f".{arm}.{i}.call.json")
          payload.write_text(json.dumps(call, sort_keys=True) + "\n")
          p = subprocess.run([a.route_adapter, "--arm", arm, "--call-json", str(payload)], capture_output=True, text=True)
          if p.returncode:
            result["decision"] = f"STOP: route adapter failed arm={arm} call={i}: {p.stderr[-500:]}"
            Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(result, indent=2)+"\n"); return 2
          result["arms"][arm].append(json.loads(p.stdout))
    result["census"]["observed"] = {"installed":36,"candidate":36}
    result["status"] = "PASS"
    result["decision"] = "PASS: adapter completed exact 36-call three-arm replay"
  Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__": sys.exit(main())
