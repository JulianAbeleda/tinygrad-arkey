#!/usr/bin/env python3
"""Audit experimental WMMA A/B fragment reuse keys.

The renderer can emit `WMMA_FRAG_KEY_JSON` rows with
`PREFILL_WMMA_FRAG_KEY_DUMP=1`. This tool groups those rows so the resident
fragment experiment has an explicit proof gate instead of relying on raw dumps.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from collections import defaultdict
from typing import Any, Iterable

sys.path.insert(0, os.getcwd())

from extra.qk.prefill.hand_vs_generated_shape_matrix import DEFAULT_DBUF_ENV


REUSE_FLAGS = {
  "PREFILL_WMMA_AB_ADDR_KEY": "1",
  "PREFILL_WMMA_CHAIN_AB_RESIDENT": "1",
  "PREFILL_WMMA_FRAG_KEY_DUMP": "1",
  "PREFILL_WMMA_AB_PROOF_META": "1",
}

ACTIVE_SHAPES = "2,2;4,2;2,4"
PROOF_FIELDS = ("role", "lds_buffer_id", "dbuf_slot", "k_phase", "logical_row_or_col",
                "byte_start", "byte_len", "producer_epoch", "overwrite_epoch")


def _load_rows_from_text(text: str) -> list[dict[str, Any]]:
  rows = []
  for line in text.splitlines():
    if line.startswith("WMMA_FRAG_KEY_JSON "):
      rows.append(json.loads(line[len("WMMA_FRAG_KEY_JSON "):]))
  return rows


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
  out = []
  for item in raw.split(";"):
    if not item.strip(): continue
    a, b = item.split(",", 1)
    out.append((int(a), int(b)))
  return out


def _run_probe(m_up: int, timeout: int) -> tuple[list[dict[str, Any]], str, str]:
  env = {**os.environ, **DEFAULT_DBUF_ENV, **REUSE_FLAGS, "PYTHONPATH": os.getcwd()}
  cmd = [
    sys.executable, "extra/qk/prefill/native_isa_l4_stream_probe.py",
    "--prefill-dbuf", "1", "--targeted-waitcnt", "1", "--b128-frag", "1",
    "--m-up", str(m_up), "--indent", "0",
  ]
  p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
  return _load_rows_from_text(p.stdout), p.stdout, p.stderr


def _run_shape(shape: tuple[int, int], args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, str]:
  env = {**os.environ, **DEFAULT_DBUF_ENV, **REUSE_FLAGS, "PYTHONPATH": os.getcwd(), "WMMA_FRAG_KEY_AUDIT_WORKER": "1",
         "MM": str(args.m), "OUTF": str(args.n), "INF": str(args.k), "U0": str(shape[0]), "U1": str(shape[1]),
         "LOC": str(args.loc), "UNR": str(args.unr)}
  p = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=args.timeout)
  return _load_rows_from_text(p.stdout), p.stdout, p.stderr


def _audit_worker() -> None:
  from extra.qk.prefill_v2_schedule_search import _compile_native_program
  _compile_native_program(int(os.environ["MM"]), int(os.environ["OUTF"]), int(os.environ["INF"]),
                          int(os.environ["U0"]), int(os.environ["U1"]), int(os.environ["LOC"]), int(os.environ["UNR"]))


def _carrier_key(row: dict[str, Any]) -> str:
  return f"id:{row.get('fallback_id_key', row.get('carrier_id'))}"


def _address_key(row: dict[str, Any]) -> str:
  if row.get("reuse_key") is not None: return str(row["reuse_key"])
  parts = (row.get("ptr_key"), row.get("dyn_key"), row.get("const_byte_start"), row.get("const_byte_end"), row.get("byte_len"))
  if any(x is not None for x in parts): return repr(parts)
  return _carrier_key(row)


def _proof_key(row: dict[str, Any]) -> str | None:
  for field in ("proof_key", "frag_key"):
    if row.get(field) is not None: return str(row[field])
  proof = row.get("proof")
  if isinstance(proof, dict) and all(proof.get(k) is not None for k in PROOF_FIELDS): return repr(tuple(proof[k] for k in PROOF_FIELDS))
  return None


def _unprovable_reasons(row: dict[str, Any]) -> list[str]:
  reasons = list(row.get("missing_proof_fields") or [])
  if not _proof_key(row): reasons.append("missing_proof_key")
  if not row.get("contiguous", False): reasons.append(row.get("proof_key_status") or row.get("reason") or "not_contiguous")
  status = row.get("proof_key_status")
  if status and str(status).startswith("unprovable"): reasons.append(str(status))
  return sorted(set(str(x) for x in reasons))


def _group_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
  groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
  for row in rows:
    if mode == "carrier":
      key = _carrier_key(row)
    elif mode == "address":
      key = _address_key(row)
    elif mode == "proof":
      key = _proof_key(row)
      if key is None: key = "UNPROVABLE:" + ",".join(_unprovable_reasons(row))
    else:
      raise ValueError(mode)
    groups[(str(row.get("role")), str(key))].append(row)

  out = []
  for (role, key), grows in sorted(groups.items(), key=lambda kv: (kv[0][0], min(r.get("const_byte_start") or -1 for r in kv[1]), kv[0][1])):
    missing = sorted({m for r in grows for m in _unprovable_reasons(r)})
    statuses = sorted({str(r.get("proof_key_status")) for r in grows})
    consts = sorted({r.get("const_byte_start") for r in grows})
    carriers = sorted({r.get("carrier_id") for r in grows})
    tiles = sorted({r.get("tile") for r in grows})
    proof_keys = sorted({_proof_key(r) for r in grows if _proof_key(r) is not None})
    out.append({
      "role": role,
      "key": key,
      "consumer_count": len(grows),
      "carrier_count": len(carriers),
      "tiles": tiles,
      "const_byte_starts": consts,
      "byte_windows": sorted({(r.get("const_byte_start"), r.get("const_byte_end")) for r in grows}),
      "all_contiguous": all(bool(r.get("contiguous")) for r in grows),
      "missing_proof_fields": missing,
      "proof_key_statuses": statuses,
      "proof_key_count": len(proof_keys),
      "promotion_safe": mode == "proof" and len(proof_keys) == 1 and len(missing) == 0 and all(bool(r.get("contiguous")) for r in grows),
    })
  return out


def _unsafe_address_merges(address_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [{
    "role": g["role"], "address_key": g["key"], "consumers": g["consumer_count"], "carriers": g["carrier_count"],
    "tiles": g["tiles"], "byte_windows": g["byte_windows"], "unprovable_reasons": g["missing_proof_fields"],
  } for g in address_groups if g["consumer_count"] > 1 and not g["promotion_safe"]]


def _summary(rows: list[dict[str, Any]], shape: str | None=None) -> dict[str, Any]:
  by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in rows: by_role[str(row.get("role"))].append(row)
  carrier_groups = _group_rows(rows, "carrier")
  address_groups = _group_rows(rows, "address")
  proof_groups = _group_rows(rows, "proof")
  return {
    "shape": shape,
    "row_count": len(rows),
    "roles": {role: len(rrows) for role, rrows in sorted(by_role.items())},
    "provable_contiguous_rows": sum(1 for r in rows if r.get("provable") and r.get("contiguous")),
    "carrier_group_count": len(carrier_groups),
    "carrier_reused_group_count": sum(1 for g in carrier_groups if g["consumer_count"] > 1),
    "address_group_count": len(address_groups),
    "address_reused_group_count": sum(1 for g in address_groups if g["consumer_count"] > 1),
    "proof_group_count": len(proof_groups),
    "promotion_safe_group_count": sum(1 for g in proof_groups if g["promotion_safe"]),
    "groups_by_current_carrier": carrier_groups,
    "groups_by_address_only": address_groups,
    "groups_by_proof_key": proof_groups,
    "rejected_address_only_merges": _unsafe_address_merges(address_groups),
  }


def _print_table(payload: dict[str, Any]) -> None:
  if payload.get("shape"): print(f"shape={payload['shape']}")
  print(f"rows={payload['row_count']} roles={payload['roles']} carrier_groups={payload['carrier_group_count']} "
        f"carrier_reused_groups={payload['carrier_reused_group_count']} address_groups={payload['address_group_count']} "
        f"address_reused_groups={payload['address_reused_group_count']} "
        f"proof_groups={payload['proof_group_count']} promotion_safe_groups={payload['promotion_safe_group_count']}")
  print("| grouping | role | consumers | carriers | tiles | byte windows | safe | unprovable / missing proof |")
  print("|---|---|---:|---:|---|---|---:|---|")
  for label, groups in (("address", payload["groups_by_address_only"]), ("proof", payload["groups_by_proof_key"])):
    for g in groups:
      if g["consumer_count"] <= 1 and label == "address": continue
      if g["consumer_count"] <= 1 and g["promotion_safe"] is False: continue
      print(f"| {label} | {g['role']} | {g['consumer_count']} | {g['carrier_count']} | {g['tiles']} | "
            f"{g['byte_windows']} | {g['promotion_safe']} | {','.join(g['missing_proof_fields'])} |")
  if payload["rejected_address_only_merges"]:
    print("\nrejected address-only merges:")
  for g in payload["rejected_address_only_merges"]:
    if g["consumers"] <= 1: continue
    print(f"- role={g['role']} consumers={g['consumers']} carriers={g['carriers']} tiles={g['tiles']} "
          f"windows={g['byte_windows']} reasons={','.join(g['unprovable_reasons'])}")


def main(argv: Iterable[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--from-file", help="read an existing probe stdout dump instead of running the probe")
  ap.add_argument("--shapes", default=ACTIVE_SHAPES, help="semicolon-separated active generated shapes, default 2,2;4,2;2,4")
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=5120)
  ap.add_argument("--k", type=int, default=5120)
  ap.add_argument("--loc", type=int, default=2)
  ap.add_argument("--unr", type=int, default=2)
  ap.add_argument("--m-up", type=int, default=1, help="native_isa_l4_stream_probe UPCAST count; 1 is 2x2")
  ap.add_argument("--legacy-probe", action="store_true", help="use native_isa_l4_stream_probe --m-up instead of generated u0/u1 shapes")
  ap.add_argument("--timeout", type=int, default=180)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(list(argv) if argv is not None else None)

  if args.from_file:
    with open(args.from_file, "r", encoding="utf-8") as f: rows = _load_rows_from_text(f.read())
    payload: dict[str, Any] = _summary(rows)
  elif args.legacy_probe:
    rows, _stdout, stderr = _run_probe(args.m_up, args.timeout)
    if not rows and stderr: print(stderr, file=sys.stderr)
    payload = _summary(rows, f"legacy-m-up-{args.m_up}")
  else:
    reports = []
    for shape in _parse_shapes(args.shapes):
      rows, _stdout, stderr = _run_shape(shape, args)
      if not rows and stderr: print(f"shape={shape[0]}x{shape[1]} stderr:\n{stderr}", file=sys.stderr)
      reports.append(_summary(rows, f"{shape[0]}x{shape[1]}"))
    payload = {"active_shapes": [r["shape"] for r in reports], "reports": reports,
               "total_rows": sum(r["row_count"] for r in reports),
               "total_promotion_safe_groups": sum(r["promotion_safe_group_count"] for r in reports),
               "total_rejected_address_only_merges": sum(len(r["rejected_address_only_merges"]) for r in reports)}

  if args.json: print(json.dumps(payload, indent=2, sort_keys=True))
  elif "reports" in payload:
    print(f"active_shapes={payload['active_shapes']} total_rows={payload['total_rows']} "
          f"total_promotion_safe_groups={payload['total_promotion_safe_groups']} "
          f"total_rejected_address_only_merges={payload['total_rejected_address_only_merges']}")
    for report in payload["reports"]:
      _print_table(report)
      print()
  else: _print_table(payload)
  return 0 if payload.get("total_rows", payload.get("row_count", 0)) else 2


if __name__ == "__main__":
  if os.environ.get("WMMA_FRAG_KEY_AUDIT_WORKER") == "1":
    _audit_worker()
    raise SystemExit(0)
  raise SystemExit(main())
