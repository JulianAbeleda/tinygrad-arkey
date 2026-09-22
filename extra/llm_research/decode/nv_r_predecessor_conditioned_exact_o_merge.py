#!/usr/bin/env python3
"""Merge the final forward/reverse exact-O sessions and verify closure."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics

TARGET = "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"
ARMS = ("C0", "C2", "C3", "C4", "C5", "C6")
COUNT = 36
CLEAN_C_US = 7.6979
FROZEN_P_US = 9.184


def _load(path: pathlib.Path):
  return json.loads(path.read_text(encoding="utf-8"))


def _complete_replays(path: pathlib.Path):
  sizes_expected = (32, 64, 128, 256, 116)
  lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
  sizes = [len(x.get("entries", [])) for x in lines]
  out, i = [], 0
  while i + len(sizes_expected) <= len(lines):
    if tuple(sizes[i:i+len(sizes_expected)]) == sizes_expected:
      out.append([e for row in lines[i:i+len(sizes_expected)] for e in row.get("entries", [])])
      i += len(sizes_expected)
    else: i += 1
  return out


def _overlap_observation(paths: list[pathlib.Path], occurrence: int):
  rows, overlapping = [], []
  for path in paths:
    for ridx, replay in enumerate(_complete_replays(path)):
      targets = [x for x in replay if x.get("name") == TARGET]
      if len(targets) <= occurrence: continue
      target = targets[occurrence]
      st, en = float(target["start"]), float(target["end"])
      overlaps = [x["name"] for x in replay if x is not target and float(x["start"]) < en and float(x["end"]) > st]
      rows.append({"profile": path.name, "replay": ridx, "target_us": float(target["duration"]),
                   "overlap_names": overlaps})
      overlapping.extend(overlaps)
  return {"target_interval_count": len(rows), "overlapping_interval_count": len(overlapping),
          "overlapping_names": overlapping, "rows": rows}


def _sha(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
  return h.hexdigest()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--sessions", type=pathlib.Path, nargs=2, required=True)
  ap.add_argument("--profiles", type=pathlib.Path, nargs=2, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--manifest", type=pathlib.Path, required=True)
  args = ap.parse_args()
  sessions = [_load(x) for x in args.sessions]

  for row in sessions:
    assert row["verdict"] == "MEASURED_IDENTITY"
    assert row["identity"]["identity_residual_us"] == 0
    assert row["capture_identity"]["prefix_contiguous"] is True
    assert row["capture_identity"]["prefix_count"] == 14
    assert all(row["capture_identity"]["dependencies"].values())
    assert set(row["arm_medians_us"]) == set(ARMS)
    assert set(row["arm_output_checksums"].values()) == {row["production_output_checksum"]}
  assert len({x["token_sha256"] for x in sessions}) == 1
  assert len({x["production_output_checksum"] for x in sessions}) == 1
  assert sessions[0]["cubin_sha256"] == sessions[1]["cubin_sha256"]
  assert sessions[0]["capture_identity"]["prefix_names"] == sessions[1]["capture_identity"]["prefix_names"]

  arm = {name: round(statistics.median([float(x["arm_medians_us"][name]) for x in sessions]), 3) for name in ARMS}
  p = round(statistics.median([float(x["installed_P"]["median_us"]) for x in sessions]), 3)
  identity = {
    "P_minus_C0_us": round(p-arm["C0"], 3),
    "C2_minus_C0_us": round(arm["C2"]-arm["C0"], 3),
    "C3_minus_C2_us": round(arm["C3"]-arm["C2"], 3),
    "C4_minus_C3_us": round(arm["C4"]-arm["C3"], 3),
    "C5_minus_C4_us": round(arm["C5"]-arm["C4"], 3),
    "P_minus_C5_us": round(p-arm["C5"], 3),
  }
  identity["identity_residual_us"] = round(identity["P_minus_C0_us"] - sum(identity[k] for k in
    ("C2_minus_C0_us", "C3_minus_C2_us", "C4_minus_C3_us", "C5_minus_C4_us", "P_minus_C5_us")), 6)
  assert identity["identity_residual_us"] == 0

  clean_bridge = {
    "current_P_minus_clean_C_us_per_call": round(p-CLEAN_C_US, 4),
    "clean_C_to_C0_us_per_call": round(arm["C0"]-CLEAN_C_US, 4),
    "frozen_P_minus_clean_C_us_per_call": round(FROZEN_P_US-CLEAN_C_US, 4),
    "current_vs_frozen_P_drift_us_per_call": round(p-FROZEN_P_US, 4),
  }
  weighted = {k.replace("_us", "_us_per_token"): round(v*COUNT, 3) for k, v in identity.items() if k != "identity_residual_us"}
  weighted["clean_C_to_C0_us_per_token"] = round(clean_bridge["clean_C_to_C0_us_per_call"]*COUNT, 3)
  weighted["current_P_minus_clean_C_us_per_token"] = round(clean_bridge["current_P_minus_clean_C_us_per_call"]*COUNT, 3)
  weighted["projection_only"] = True

  result = {
    "schema": "tinygrad.nv_r_predecessor_conditioned_exact_o_merge.v1",
    "authority_sessions": [x.name for x in args.sessions], "authority_profiles": [x.name for x in args.profiles],
    "token_sha256": sessions[0]["token_sha256"],
    "production_output_checksum": sessions[0]["production_output_checksum"],
    "cubin_sha256": sessions[0]["cubin_sha256"],
    "arm_medians_us": arm, "installed_P_median_us": p, "identity": identity,
    "adjudication": {
      "position_only_C6_minus_C3_us": round(arm["C6"]-arm["C3"], 3),
      "real_prefix_minus_position_C5_minus_C6_us": round(arm["C5"]-arm["C6"], 3),
    },
    "clean_chain_bridge": clean_bridge, "count_weighted_projection": weighted,
    "profile_overlap": _overlap_observation(args.profiles, occurrence=0),
    "scope": {"measured_occurrence": 0, "family_count": COUNT,
              "all_occurrences_generalization": "UNMEASURED"},
    "verdict": "O_OCCURRENCE0_CLOSED",
  }
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")

  evidence_dir = args.out.parent
  paths = sorted(x for x in evidence_dir.iterdir() if x.is_file() and x.name != args.manifest.name)
  args.manifest.write_text("".join(f"{_sha(path)}  {path.name}\n" for path in paths), encoding="utf-8")
  print(json.dumps({"verdict": result["verdict"], "arm_medians_us": arm, "installed_P_median_us": p,
                    "identity": identity, "adjudication": result["adjudication"],
                    "profile_overlap_count": result["profile_overlap"]["overlapping_interval_count"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
