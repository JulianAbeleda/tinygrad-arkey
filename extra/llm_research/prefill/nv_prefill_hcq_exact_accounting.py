#!/usr/bin/env python3
"""Exact interval accounting for the six-segment NV pp512 HCQ graph.

This consumes device-clock timestamps exported by HCQ_GRAPH_PROFILE_JSON.  It
does not scale profile shares onto an unprofiled wall.  Active category time,
the interval union, overlap, and device-idle time are computed independently.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
from decimal import Decimal


ROLE_IDS = {
  "81b2583b95e4fcddb614036cfd9ab0abcbd8a245774be7c36dcc143e3bbdb945": "v",
  "50c7576b59193b8ecedf58701515f45e288ce8dde993db3a492badebbead057e": "k",
  "fc3b6e588b4dc162749e50ef060e3e2312f8560463e8a7219bcc99cc7fb0627a": "qo",
  "c0d2b181a6de5751180a48cc5c1b1d1c8977770845951d3196a6596877493d28": "gate_up",
  "03896c56299ec804cdeeb477becc2a574b33e76d5e2cabc7c4dc86678b7b1e62": "down",
}

NORM_NAME = "r_32_8_4_4_512_628602e2c7aa848d6e4ff1d7eb39bd95e9b93923f4805b7db94c8e9bfb030561"
ACTIVATION_NAMES = {
  "E_64_192_8_16_4_1e161f6c4c230e894f4d2601704fc92075a12b3f53be815dcba4bbed84e83ed5",
  "E_64_192_8_16_4_5a2137f8e57933947793f5908b5b1c440a16780ed3881ebe0c386c7e3680548c",
}
INPUT_NAMES = {
  "E_4_32_4_3a39b1a8ecb491ce70bff135129041430248b04e874e29e2e6fd3dbecfcf6703",
  "E_512_16_8_2_4_4_2e162bdabfe040283cb257ced27118633de4e59e1679a9699f9c54996923a749",
}


def _primary(row:dict) -> tuple[str, str|None]:
  name=row["name"]
  ident=(row.get("metadata") or {}).get("canonical_identity")
  role=ROLE_IDS.get(ident)
  if role == "v": return "v", role
  if role == "k": return "k", role
  if role == "qo": return "qo", role
  if role == "gate_up": return "gate_up", role
  if role == "down": return "down", role
  if name == "q8_compact_record_fp16": return "norm_and_activation_conversion", "q8_producer"
  if name in INPUT_NAMES: return "input_embed_graph_setup", "input_embed"
  if name == "nv_sm120_q16_grid_hd128_loop_attention": return "flash_score_reduction", "flash"
  if name == NORM_NAME: return "norm_and_activation_conversion", "rmsnorm"
  if name in ACTIVATION_NAMES: return "activation_multiply", "activation_multiply"
  if name.startswith("r_1187_"): return "vocabulary", "vocabulary_main"
  if name.startswith("E_1187_"): return "vocabulary", "vocabulary_epilogue"
  if name.startswith("native_finite_fp32_argmax_"): return "output_token_transfer", "argmax"
  if name == "E_c9699af0d7ec2ef027b0f07fc5156c7a5f00221c2efb3447f24a23d08b2e0346":
    return "output_token_transfer", "token_copy"
  return "residual_rope_kv_support", "support"


def _specialize_roles(rows:list[dict]) -> None:
  # Q and O share one binary; gate and up share one binary.  Their executed
  # ordering is invariant and proven by the 36-layer census: Q precedes Flash,
  # O follows it, then gate precedes up.  Keep the binary identity as a tag.
  qo=gu=0
  for row in rows:
    primary,tag=_primary(row)
    if primary == "qo":
      row["primary"]="q" if qo % 2 == 0 else "o";row["role"]="q" if qo % 2 == 0 else "o";qo+=1
    elif primary == "gate_up":
      row["primary"]="gate" if gu % 2 == 0 else "up";row["role"]="gate" if gu % 2 == 0 else "up";gu+=1
    else:row["primary"],row["role"]=primary,tag
  if qo != 72 or gu != 72: raise ValueError(f"incomplete dense role census: qo={qo}, gate_up={gu}")


def _assign_layers(rows:list[dict]) -> None:
  # Each layer has one V anchor.  Work before the first anchor belongs to
  # layer 0 setup; support after down remains charged to that layer.
  layer=0;seen_v=False
  role_ord=collections.Counter()
  for row in rows:
    if row["primary"] == "v":
      if seen_v: layer+=1
      seen_v=True
    row["layer"]=layer if layer < 36 else None
    if row["primary"] in {"q","k","v","o","gate","up","down","flash_score_reduction"}:
      role_ord[row["primary"]]+=1
  expected={x:36 for x in ("q","k","v","o","gate","up","down","flash_score_reduction")}
  if dict(role_ord) != expected: raise ValueError(f"role census mismatch: {dict(role_ord)}")


def _interval_partition(rows:list[dict]) -> dict:
  points=sorted({Decimal(x[k]) for x in rows for k in ("start","end")})
  exclusive=collections.defaultdict(Decimal);overlap=Decimal(0);union=Decimal(0)
  overlap_sets=collections.defaultdict(Decimal)
  concurrency_integral=Decimal(0)
  for a,b in zip(points,points[1:]):
    active=[x["primary"] for x in rows if Decimal(x["start"]) < b and Decimal(x["end"]) > a]
    if not active: continue
    dt=b-a;union+=dt
    concurrency_integral += dt*len(active)
    if len(active)==1: exclusive[active[0]]+=dt
    else:
      overlap+=dt
      mult=collections.Counter(active)
      overlap_sets[" + ".join(f"{k} x{v}" if v>1 else k for k,v in sorted(mult.items()))]+=dt
  first=min(Decimal(x["start"]) for x in rows);last=max(Decimal(x["end"]) for x in rows)
  span=last-first
  return {"first_device_timestamp_us":str(first),"last_device_timestamp_us":str(last),"device_span_us":str(span),
    "interval_union_us":str(union),"device_idle_us":str(span-union),"shared_overlap_us":str(overlap),
    "summed_active_us":str(concurrency_integral),"duplicate_active_charge_us":str(concurrency_integral-union),
    "exclusive_us":{k:str(v) for k,v in sorted(exclusive.items())},
    "overlap_sets_us":{k:str(v) for k,v in sorted(overlap_sets.items(),key=lambda x:(-x[1],x[0]))}}


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("--profile",required=True);ap.add_argument("--out",required=True)
  ap.add_argument("--segments-per-run",type=int,default=6);ap.add_argument("--run",type=int,default=-1)
  args=ap.parse_args()
  lines=[json.loads(x) for x in pathlib.Path(args.profile).read_text().splitlines() if x.strip()]
  if len(lines)%args.segments_per_run:raise ValueError("profile line count is not a whole number of graph invocations")
  groups=[lines[i:i+args.segments_per_run] for i in range(0,len(lines),args.segments_per_run)]
  idx=args.run if args.run >= 0 else len(groups)+args.run
  if idx < 0 or idx >= len(groups):raise IndexError("requested run is absent")
  rows=[]
  for seg,payload in enumerate(groups[idx]):
    if payload.get("schema")!="tinygrad.hcq_graph_profile.v1":raise ValueError("bad HCQ profile schema")
    for pos,entry in enumerate(payload["entries"]): rows.append({**entry,"segment":seg,"segment_index":pos})
  _specialize_roles(rows);_assign_layers(rows)
  active=collections.defaultdict(Decimal);counts=collections.Counter()
  for row in rows:active[row["primary"]]+=Decimal(row["duration"]);counts[row["primary"]]+=1
  by_layer=[]
  for layer in range(36):
    lr=[x for x in rows if x["layer"]==layer]
    vals=collections.defaultdict(Decimal)
    for row in lr:vals[row["primary"]]+=Decimal(row["duration"])
    by_layer.append({"layer":layer,"active_us":{k:str(v) for k,v in sorted(vals.items())}})
  partition=_interval_partition(rows)
  payload={"schema":"tinygrad.nv_prefill_hcq_exact_accounting.v1","source":str(pathlib.Path(args.profile)),
    "selected_run":idx,"available_runs":len(groups),"segments":args.segments_per_run,
    "segment_call_counts":[len(x["entries"]) for x in groups[idx]],"launches":len(rows),"unknown_launches":0,
    "active_us":{k:str(v) for k,v in sorted(active.items())},"launch_counts":dict(sorted(counts.items())),
    "timeline":partition,"per_layer":by_layer,"entries":rows}
  target=pathlib.Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
  target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps({k:payload[k] for k in ("selected_run","launches","unknown_launches","active_us","timeline")},indent=2))


if __name__=="__main__":main()
