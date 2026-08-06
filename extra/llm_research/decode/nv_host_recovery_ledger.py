#!/usr/bin/env python3
"""Hermetic CPU-only ledger for the native-NV d512 host workstream.

Encodes the outside-window partition constants from
``nv-decode-native-d512-host-partition-record-20260804.md`` and every host
recovery candidate enumerated by the exhaustive host scope.  A caller assumes
a subset of recoveries is realized on top of the composed same-session
baseline (5.3242440 ms/token, 187.82 tok/s) and the tool emits the remaining
host delta and the resulting tok/s, plus a per-item marginal tok/s and
promotion-gate break-even.

It fails closed: referencing a NO-GO route (e.g. packed greedy argmax),
claiming an already-banked item, claiming more than an item cap, unknown ids,
and negative claims all raise ``LedgerClaimError``.  A claim set whose total
plus the already-banked P1+P5 exceeds the outside-window delta is reported as
a warning (partition rows are not additive) rather than silently accepted.
No device is touched and no tinygrad default is changed.
"""
from __future__ import annotations

import argparse, json, pathlib
from typing import Mapping

SCHEMA = "tinygrad.nv_host_recovery_ledger.v1"

# --- composed same-session authority (nv-decode-final-composed-same-session-record-20260805.md) ---
COMPOSED_BASELINE_MS = 5.3242440
COMPOSED_BASELINE_TOK_S = 1000.0 / COMPOSED_BASELINE_MS

# --- outside-window authority (host-partition record; audit exact value) ---
OUTSIDE_DELTA_US = 239.804933
NATIVE_OUTSIDE_US = 321.784
LLAMA_OUTSIDE_US = 81.979
PARTITION_SUM_US = 338.458          # record-stated disjoint sum

# --- canonical partition rows (host-partition record) ---
PRE_FIRST_GRAPH_TOTAL_US = 247.557
SCALAR_COPYOUT_ITEM_US = 83.247
PYTHON_YIELD_TAIL_US = 3.096
REDUNDANT_SYNC_US = 4.358

# --- pre-first-graph sub-components (KEY FACTS / predispatch-breakdown record) ---
DEFENSIVE_COPY_REBIND_US = 121.380
JIT_SIGNATURE_RECONSTRUCTION_US = 77.927
STRUCTURAL_REWRITE_US = 49.284
INPUT_DISCOVERY_US = 15.479
VARIABLE_METADATA_US = 6.312
PRE_TINYJIT_US = 35.637
CACHE_LOOKUP_US = 1.112
PRE_FIRST_SUB_SUM_US = DEFENSIVE_COPY_REBIND_US + JIT_SIGNATURE_RECONSTRUCTION_US + PRE_TINYJIT_US + CACHE_LOOKUP_US  # 236.056

# --- banked inside the composed baseline (reconciled campaign ledger) ---
P1_BOOKED_US = 66.6620940           # descriptor cache + reusable input shadow, default-on
P5_BOOKED_US = 91.6365625           # two-capture feedback ping-pong, redirect-on midpoint
BANKED_AT_COMPOSED_TOTAL_US = P1_BOOKED_US + P5_BOOKED_US  # 158.2986565
# Per-bucket attribution used ONLY for per-item caps, never summed as credit.
P1_JIT_ATTRIBUTED_US = 50.003       # descriptor cache: prepare 78.095 -> 28.092 us
P1_COPY_ATTRIBUTED_US = 28.372      # reusable private shadow (B arm)
P5_COPY_ATTRIBUTED_US = 91.6365625  # ping-pong removes the pre-graph copy

HOST_PROMOTION_GATE_US = 25.0       # breakdown record wall A/B gate

HOST_ITEMS = [
  dict(id="predispatch_oracle_ab", label="full-logit predispatch oracle A/B",
       partition_us=69.166, sub={"structural_descriptor_cache_a": 65.536, "reusable_shadow_b": 28.372},
       attributed_us=69.166, status="BOOKED_VIA_P1", dev="GPU_ORACLE_DONE_CPU_LEDGER",
       note="P1 booked 66.662094 default-on with full-logit oracle; the 69.166 combined diagnostic arm is not additive",
       source="nv-decode-native-d512-predispatch-ab-record-20260805.md"),
  dict(id="jit_signature_reconstruction", label="JIT input/signature reconstruction",
       partition_us=JIT_SIGNATURE_RECONSTRUCTION_US,
       sub={"structural_rewrite": STRUCTURAL_REWRITE_US, "input_discovery": INPUT_DISCOVERY_US,
            "variable_metadata": VARIABLE_METADATA_US},
       attributed_us=P1_JIT_ATTRIBUTED_US, status="PARTIAL_BOOKED", dev="GPU_PARKED",
       note="P1 descriptor cache recovers the structural rewrite; remaining is discovery/metadata",
       source="nv-decode-native-d512-predispatch-breakdown-record-20260805.md"),
  dict(id="defensive_copy_rebind", label="defensive copy/rebind feedback",
       partition_us=DEFENSIVE_COPY_REBIND_US,
       sub={"fresh_alloc_uop_build": 16.892, "generic_copy_execution": 99.088},
       attributed_us=P1_COPY_ATTRIBUTED_US + P5_COPY_ATTRIBUTED_US, status="PARTIAL_BOOKED", dev="GPU_PARKED",
       note="P1 reusable shadow + P5 ping-pong; the copy is kept, never skipped",
       source="nv-decode-feedback-pingpong-record-20260805.md"),
  dict(id="pre_tinyjit", label="pre-TinyJit generator/model work",
       partition_us=PRE_TINYJIT_US, attributed_us=0.0, status="OPEN", dev="GPU_PARKED",
       note="variable bind, flash-route select, feedback slot, out.realize(); no mechanism named",
       source="nv-decode-native-d512-predispatch-breakdown-record-20260805.md"),
  dict(id="cache_lookup", label="settled graph-cache lookup",
       partition_us=CACHE_LOOKUP_US, attributed_us=0.0, status="OPEN", dev="GPU_PARKED",
       note="already cheap; below the 25 us host promotion gate",
       source="nv-decode-native-d512-predispatch-breakdown-record-20260805.md"),
  dict(id="scalar_copyout_item", label="scalar copyout + Tensor.item()",
       partition_us=SCALAR_COPYOUT_ITEM_US, attributed_us=0.0, status="OPEN", dev="GPU_PARKED",
       note="packed greedy argmax alternative was NO-GO; staged-copyout prototype is CPU-safe, wall credit GPU-parked",
       source="nv-decode-native-d512-host-partition-record-20260804.md"),
  dict(id="python_yield_tail", label="Python yield tail",
       partition_us=PYTHON_YIELD_TAIL_US, attributed_us=0.0, status="OPEN", dev="CPU_ANALYZED",
       note="generator contract; below the 25 us gate",
       source="nv-decode-native-d512-host-partition-record-20260804.md"),
  dict(id="redundant_sync", label="redundant synchronize after next()",
       partition_us=REDUNDANT_SYNC_US, attributed_us=REDUNDANT_SYNC_US,
       status="ALREADY_ABSENT_AT_COMPOSED", dev="CPU_ANALYZED",
       note="diagnostic harness per-token sync; the composed baseline syncs once per 32-token window",
       source="nv-decode-native-d512-host-partition-record-20260804.md"),
]

NO_GO_ROUTES = [
  dict(id="packed_argmax", label="packed greedy argmax", partition_us=0.0, status="NO_GO", dev="CLOSED",
       note="71.874 -> 142.647 us (+70.773); do not reopen without a different mechanism",
       source="nv-packed-argmax-microgate-record-20260805.md"),
  dict(id="p2b_owned_invocation_input", label="owned invocation-input copy removal", partition_us=0.0,
       status="NO_GO", dev="CLOSED",
       note="TOPOLOGY_NO_GO; 841 programs -> required <=804 gate failed; zero credit",
       source="nv-p2b-owned-invocation-input-record-20260805.md"),
]

ITEMS_BY_ID = {item["id"]: item for item in HOST_ITEMS}
NO_GO_BY_ID = {route["id"]: route for route in NO_GO_ROUTES}


class LedgerClaimError(ValueError):
  """A claim violates a record-pinned rule (NO-GO, cap, banked, unknown)."""


def max_claimable(item: dict) -> float:
  if item["status"] in ("BOOKED_VIA_P1", "ALREADY_ABSENT_AT_COMPOSED"): return 0.0
  return item["partition_us"] - item["attributed_us"]


def tok_s(ms: float) -> float:
  return 1000.0 / ms


def marginal_tok_s(partition_us: float) -> float:
  assert 0.0 <= partition_us < COMPOSED_BASELINE_MS * 1000.0, "partition exceeds the composed baseline"
  return tok_s(COMPOSED_BASELINE_MS - partition_us / 1000.0)


def _validate_claims(claims: Mapping[str, float]) -> dict[str, float]:
  out: dict[str, float] = {}
  for raw_id, amount in claims.items():
    rid = raw_id.strip()
    if rid in NO_GO_BY_ID:
      raise LedgerClaimError(
        f"{rid} is NO_GO per {NO_GO_BY_ID[rid]['source']} ({NO_GO_BY_ID[rid]['note']}); fail closed")
    if rid not in ITEMS_BY_ID:
      raise LedgerClaimError(f"unknown recovery id {rid!r}; known ids: {', '.join(ITEMS_BY_ID)}")
    item = ITEMS_BY_ID[rid]
    if amount < 0: raise LedgerClaimError(f"{rid}: negative claim {amount}")
    cap = max_claimable(item)
    if amount > cap + 1e-9:
      if item["status"] == "BOOKED_VIA_P1":
        raise LedgerClaimError(
          f"{rid}: already banked at the composed baseline via P1 (66.662094 us); additional credit is 0")
      if item["status"] == "ALREADY_ABSENT_AT_COMPOSED":
        raise LedgerClaimError(
          f"{rid}: already absent at the composed baseline (per-token sync not in the 32-token-window timer); "
          "additional credit is 0")
      raise LedgerClaimError(
        f"{rid}: claim {amount} us exceeds max claimable {cap} us "
        f"(partition {item['partition_us']} minus already banked {item['attributed_us']})")
    out[rid] = amount
  return out


def evaluate(assume_recovered: Mapping[str, float] | None = None) -> dict:
  """Return the full audit payload for the assumed recovery set (marginal frame)."""
  claims = dict(assume_recovered or {})
  _validate_claims(claims)
  recovered_total = sum(claims.values())
  remaining_delta = OUTSIDE_DELTA_US - recovered_total
  remaining_composed = remaining_delta - BANKED_AT_COMPOSED_TOTAL_US
  resulting_tok_s = tok_s(COMPOSED_BASELINE_MS - recovered_total / 1000.0)
  banked_overreach = BANKED_AT_COMPOSED_TOTAL_US + recovered_total - OUTSIDE_DELTA_US
  warnings = []
  if banked_overreach > 1e-9:
    warnings.append(
      f"recovered {recovered_total} us plus already-banked {BANKED_AT_COMPOSED_TOTAL_US} us exceeds the "
      f"outside-window delta by {banked_overreach:.3f} us; partition rows are not additive, treat the "
      "composed-baseline remainder as a lower bound, not a booking")
  items = []
  for item in HOST_ITEMS:
    cap = max_claimable(item)
    marginal = marginal_tok_s(item["partition_us"]) if item["partition_us"] > 0 else None
    items.append({
      "id": item["id"], "label": item["label"], "partition_us": item["partition_us"],
      "sub_components_us": item.get("sub"), "already_banked_us": item["attributed_us"],
      "max_claimable_us": round(cap, 6), "status": item["status"], "dev": item["dev"],
      "source": item["source"],
      "marginal_tok_s_if_only_this_partition_recovered": round(marginal, 3) if marginal is not None else None,
      "tok_s_gain": round(marginal - COMPOSED_BASELINE_TOK_S, 2) if marginal is not None else None,
      "promotion_gate": ("ABOVE_HOST_PROMOTION_GATE" if item["partition_us"] >= HOST_PROMOTION_GATE_US
                         else "BELOW_GATE_NOISE"),
    })
  return {
    "schema": SCHEMA,
    "frame": {
      "composed_baseline_ms_per_token": COMPOSED_BASELINE_MS,
      "composed_baseline_tok_s": round(COMPOSED_BASELINE_TOK_S, 3),
      "outside_delta_us": OUTSIDE_DELTA_US,
      "native_outside_us": NATIVE_OUTSIDE_US,
      "llama_outside_us": LLAMA_OUTSIDE_US,
      "partition_sum_stated_us": PARTITION_SUM_US,
      "partition_rows_arithmetic_us": round(
        PRE_FIRST_GRAPH_TOTAL_US + SCALAR_COPYOUT_ITEM_US + PYTHON_YIELD_TAIL_US + REDUNDANT_SYNC_US, 3),
      "banked_at_composed_total_us": round(BANKED_AT_COMPOSED_TOTAL_US, 6),
      "banked_p1_us": P1_BOOKED_US,
      "banked_p5_us": P5_BOOKED_US,
      "host_promotion_gate_us": HOST_PROMOTION_GATE_US,
    },
    "partition": {
      "pre_first_graph_us": PRE_FIRST_GRAPH_TOTAL_US,
      "pre_first_sub_components_us": {
        "defensive_copy_rebind": DEFENSIVE_COPY_REBIND_US,
        "jit_signature_reconstruction": JIT_SIGNATURE_RECONSTRUCTION_US,
        "pre_tinyjit": PRE_TINYJIT_US,
        "cache_lookup": CACHE_LOOKUP_US,
      },
      "pre_first_sub_sum_us": round(PRE_FIRST_SUB_SUM_US, 3),
      "scalar_copyout_item_us": SCALAR_COPYOUT_ITEM_US,
      "python_yield_tail_us": PYTHON_YIELD_TAIL_US,
      "redundant_sync_us": REDUNDANT_SYNC_US,
    },
    "items": items,
    "no_go_routes": NO_GO_ROUTES,
    "assumed_recovered_us": {key: claims[key] for key in sorted(claims)},
    "recovered_total_us": round(recovered_total, 6),
    "remaining_host_delta_us": round(remaining_delta, 6),
    "remaining_host_delta_at_composed_baseline_us": round(remaining_composed, 6),
    "resulting_tok_s": round(resulting_tok_s, 3),
    "tok_s_delta_vs_composed": round(resulting_tok_s - COMPOSED_BASELINE_TOK_S, 2),
    "host_parity_tok_s": round(marginal_tok_s(OUTSIDE_DELTA_US), 3),
    "warnings": warnings,
  }


def parse_assume_recovered(text: str | None) -> dict[str, float]:
  """Parse the CLI comma list: ``id`` (full remaining cap) or ``id=us``."""
  if text is None or not text.strip(): return {}
  out: dict[str, float] = {}
  for token in text.split(","):
    token = token.strip()
    if not token: continue
    if "=" in token:
      rid, _, amount_s = token.partition("=")
      rid, amount_s = rid.strip(), amount_s.strip()
      try: amount = float(amount_s)
      except ValueError: raise LedgerClaimError(f"invalid amount {amount_s!r} for {rid}")
    else:
      rid, amount = token.strip(), None
    if rid in NO_GO_BY_ID:
      raise LedgerClaimError(f"{rid} is NO_GO per {NO_GO_BY_ID[rid]['source']}; fail closed on any claim")
    if rid not in ITEMS_BY_ID: raise LedgerClaimError(f"unknown recovery id {rid!r}")
    if amount is None:
      amount = max_claimable(ITEMS_BY_ID[rid])
      if amount == 0.0:
        raise LedgerClaimError(f"{rid}: no claimable credit remains ({ITEMS_BY_ID[rid]['status']})")
    out[rid] = amount
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--assume-recovered", default=None,
                  help="comma list of 'id' (full remaining cap) or 'id=us'")
  ap.add_argument("--out", help="write the JSON payload to this path")
  args = ap.parse_args()
  payload = evaluate(parse_assume_recovered(args.assume_recovered))
  text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if args.out:
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
  print(text, end="")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
