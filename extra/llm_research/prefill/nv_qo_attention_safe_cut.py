#!/usr/bin/env python3
"""Build a research-only NV queue cut that keeps attention's direct inputs local."""
from __future__ import annotations
import argparse, hashlib, json, pathlib


def identity(name:str)->str:
  stem,sep,suffix=name.rpartition("_")
  return stem if sep and len(suffix)==64 and all(c in "0123456789abcdef" for c in suffix) else name


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--profile",required=True);ap.add_argument("--census",required=True)
  ap.add_argument("--out",required=True)
  ap.add_argument("--pin-dep-position",type=int,action="append",default=[],
    help="pin only these zero-based positions in each attention dependency list; default pins every direct dependency")
  args=ap.parse_args()
  profiles=[json.loads(x) for x in pathlib.Path(args.profile).read_text().splitlines()[:6]]
  censuses=[json.loads(x) for x in pathlib.Path(args.census).read_text().splitlines()[:6]]
  rows=[];identity_lists=[]
  for profile,census in zip(profiles,censuses):
    names=[str(x["name"]) for x in profile["entries"]]
    identities=list(map(identity,names));identity_lists.append(identities)
    unsafe={dep for i,entry in enumerate(profile["entries"]) if entry["name"]=="nv_sm120_q16_grid_hd128_loop_attention"
            for pos,dep in enumerate(profile["deps"][i]) if not args.pin_dep_position or pos in args.pin_dep_position}
    selected=[int(x["graph_idx"]) for x in census["assignments"] if int(x["queue"])==1 and int(x["graph_idx"]) not in unsafe]
    rows.append({"prefix_count":len(names),"prefix_name_digest":hashlib.sha256("\n".join(identities).encode()).hexdigest(),
      "selected":[{"index":i,"identity":identity(names[i])} for i in selected],
      "diagnostic":{"default_aux":sum(int(x["queue"])==1 for x in census["assignments"]),
                    "attention_direct_inputs_pinned":len(unsafe),"pin_dep_positions":args.pin_dep_position or "all",
                    "selected_aux":len(selected)}})
  # Runtime policies deliberately match prefixes so a suffix-only graph change
  # fails closed.  A captured graph can therefore match both its own row and a
  # shorter graph row when the latter is an exact prefix.  Keep the longer,
  # more-specific row and omit the shorter one; the shorter graph then runs on
  # the primary queue, which is conservative and unambiguous.
  omitted={i for i,names in enumerate(identity_lists) for j,other in enumerate(identity_lists)
           if i!=j and len(names)<len(other) and other[:len(names)]==names}
  rows=[row for i,row in enumerate(rows) if i not in omitted]
  payload={"schema":"tinygrad.nv_multi_queue_cut_policy.v1","graphs":rows,
    "diagnostic":{"omitted_ambiguous_prefix_graphs":sorted(omitted)}}
  out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n")
  print(json.dumps({"graphs":len(rows),"omitted_ambiguous_prefix_graphs":sorted(omitted),
    "diagnostic":[r["diagnostic"] for r in rows]},sort_keys=True))

if __name__=="__main__":main()
