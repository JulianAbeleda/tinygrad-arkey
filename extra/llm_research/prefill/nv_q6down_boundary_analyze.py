#!/usr/bin/env python3
"""D0.3 attribution for the Q6-down boundary capture (offline, fail closed)."""
import argparse, json
from pathlib import Path

BOUNDARIES = ("producer", "publication", "main", "residual")
CAPABILITIES = ("allocations", "copyin", "copyout", "graph_copy", "materializations")

def records(path):
  with path.open() as f:
    for n, line in enumerate(f, 1):
      if line.strip():
        try: yield json.loads(line)
        except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: {e}")

def unavailable(capability):
  return {"status": "UNAVAILABLE", "count": None, "bytes": None, "capability": capability}

def analyze(root):
  root = Path(root)
  issues, events, seen = [], [], set()
  arms = sorted(p for p in root.iterdir() if p.is_dir())
  for arm_dir in arms:
    result = json.loads((arm_dir / "result.json").read_text())
    arm = result.get("samples", [{}])[0].get("arm", "unknown")
    boundary = result.get("cut", {}).get("boundary", arm_dir.name.rsplit("-", 2)[0])
    temperature = result.get("samples", [{}])[0].get("temperature", "unknown")
    role_samples = {(s.get("sample"), s.get("temperature")): s for s in result.get("samples", [])}
    cuts = list(records(arm_dir / "profile.jsonl"))
    bufs = list(records(arm_dir / "buffer-events.jsonl"))
    if len(cuts) != len(bufs): issues.append(f"{arm_dir.name}: cut/buffer cardinality mismatch")
    for cut, buf in zip(cuts, bufs):
      sample = cut.get("sample")
      key = (arm_dir.name, sample)
      if key in seen: issues.append(f"duplicate invocation {key}")
      seen.add(key)
      if cut.get("packet") != "D0.2" or buf.get("packet") != "D0.2": issues.append(f"{key}: packet mismatch")
      cb = cut.get("boundary", boundary)
      if cb not in BOUNDARIES: issues.append(f"{key}: invalid boundary {cb}")
      sample_record = role_samples.get((sample, cut.get("temperature", temperature)))
      if sample_record is None: sample_record = next((s for s in result.get("samples", []) if s.get("sample") == sample), None)
      roles = sample_record.get("roles", []) if sample_record else []
      if not roles: issues.append(f"{key}: no roles")
      lifecycle = {}
      for cap in CAPABILITIES:
        src = buf.get(cap)
        if src is None and cap in ("copyin", "copyout", "graph_copy"):
          src = buf.get("copies")
        lifecycle[cap] = dict(src) if isinstance(src, dict) else unavailable(cap)
        if lifecycle[cap].get("status") == "UNAVAILABLE": lifecycle[cap] = unavailable(cap)
      # Segment-local readiness: use only explicit predecessor completion, never global time.
      pred = cut.get("predecessors", cut.get("segment_predecessors", []))
      completion = [p.get("end_ns") for p in pred if isinstance(p, dict) and isinstance(p.get("end_ns"), (int, float))]
      ready = max(completion) if completion else None
      readiness_status = "OK" if completion else "UNAVAILABLE"
      successors = cut.get("successors", cut.get("successor_fanout", []))
      event = {"invocation": {"arm": arm, "boundary": cb, "temperature": cut.get("temperature", temperature), "sample": sample},
               "roles": roles, "cut": cut, "lifecycle": lifecycle,
               "dependency_ready": {"status": readiness_status, "ns": ready, "predecessor_count": len(pred)},
               "successor_fanout": successors, "charge": {"selected": True, "charge_id": f"{arm_dir.name}:{sample}:{cb}"}}
      events.append(event)
  coverage = len(events) and sum(bool(e["roles"]) and all(e["invocation"].get(k) not in (None, "unknown") for k in ("arm", "boundary", "temperature", "sample")) for e in events) / len(events) or 0
  unknown = sum(1 for e in events for v in e["lifecycle"].values() if v.get("status") == "UNKNOWN")
  charges = [e["charge"]["charge_id"] for e in events]
  duplicate_charges = len(charges) - len(set(charges))
  status = "PASS" if events and not issues and coverage == 1 and unknown == 0 and duplicate_charges == 0 else "STOP"
  return {"schema": "tinygrad.nv_q6down_boundary_attribution.v1", "packet": "D0.3", "status": status,
          "source": str(root), "coverage": {"events": len(events), "fully_attributed": sum(bool(e["roles"]) for e in events), "ratio": coverage, "unknown": unknown, "duplicate_charges": duplicate_charges},
          "capabilities": {c: ("UNAVAILABLE" if not any(e["lifecycle"][c].get("status") == "OK" for e in events) else "OBSERVED") for c in CAPABILITIES},
          "events": events, "issues": issues, "decision": "Attribution-only; no performance or composition decision.", "next_packet": "D0.4" if status == "PASS" else None}

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--output", required=True); a = ap.parse_args()
  out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(analyze(a.root), indent=2) + "\n")
  print(json.loads(out.read_text())["status"])
if __name__ == "__main__": main()
