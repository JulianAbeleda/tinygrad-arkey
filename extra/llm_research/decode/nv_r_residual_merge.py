#!/usr/bin/env python3
"""Merge per-row fresh-process probe outputs into one adjudication artifact.

Each row of ``nv_r_residual_cache_dispatch_probe.py`` was run in its own
process (``--keys <row>``) to avoid cross-arm HCQ signal/QMD state.  This
tool reassembles those files in the probe's canonical row order, writes the
merged JSON, and records the SHA-256 of the merged artifact plus the cubins.

Measurement tooling only; no production code path is touched.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROW_ORDER = [
  "control_norm_8_128", "control_norm_32_128", "q_coop_4096", "q_g3_4096",
  "o_epi_4096", "flash_score", "kv_coop_1024", "kv_g3_1024",
]


def sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def main() -> int:
  src = pathlib.Path(sys.argv[1])
  out_dir = pathlib.Path(sys.argv[2])
  out_dir.mkdir(parents=True, exist_ok=True)

  parts = {k: json.loads((src / f"{k}.json").read_text()) for k in ROW_ORDER}
  assert len({p["commit"] for p in parts.values()}) == 1, "commit mismatch across rows"
  assert len({p["schema"] for p in parts.values()}) == 1, "schema mismatch across rows"
  assert len({p["n_per_arm"] for p in parts.values()}) == 1, "n mismatch across rows"

  head = parts[ROW_ORDER[0]]
  merged = {
    "schema": head["schema"],
    "commit": head["commit"],
    "method": head["method"],
    "flush_mib": head["flush_mib"],
    "n_per_arm": head["n_per_arm"],
    "row_order": ROW_ORDER,
    "run_policy": "one fresh process per row to avoid cross-arm HCQ signal/QMD state",
    "rows": [r for k in ROW_ORDER for r in parts[k]["rows"]],
  }

  merged_path = out_dir / "nv-r-residual-cache-dispatch-probe.json"
  merged_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")

  lines = [f"{sha256(merged_path)}  ./{merged_path.name}"]
  cubin_paths = sorted({pathlib.Path(r["cubin"]).resolve() for r in merged["rows"]})
  for cubin in cubin_paths:
    lines.append(f"{sha256(cubin)}  {cubin.name}")
  (out_dir / "sha256.txt").write_text("\n".join(lines) + "\n")

  print(json.dumps({"merged": str(merged_path), "rows": len(merged["rows"]),
                    "sha256": sha256(merged_path)}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
