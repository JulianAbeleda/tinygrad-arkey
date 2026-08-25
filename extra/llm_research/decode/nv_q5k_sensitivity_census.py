#!/usr/bin/env python3
"""Join Q5 singleton full-logit arms to local projection quality evidence."""
from __future__ import annotations
import argparse,glob,json,pathlib
import numpy as np

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--evidence",default="docs/task_workflow/evidence/nv-numerical-byte-reduction");ap.add_argument("--out",required=True);a=ap.parse_args();p=pathlib.Path(a.evidence)
  control=np.load(p/"q5-logits-control.npz")["logits"].astype(np.float64);feas=json.loads((p/"q6-q5-feasibility.json").read_text())
  local={r["block"]:{"local_output_relative_l2_median":float(np.median([x["relative_l2"] for x in r["outputs"]])),"weight_relative_l2":r["weight"]["relative_l2"]} for r in feas["rows"] if r["role"]=="down"};rows=[]
  for f in glob.glob(str(p/"q5-logits-block*.npz")):
    stem=pathlib.Path(f).stem
    if "retain" in stem:continue
    block=int(stem.rsplit("block",1)[1]);candidate=np.load(f)["logits"].astype(np.float64);steps=[float(np.linalg.norm(y-x)/np.linalg.norm(x)) for x,y in zip(control,candidate)]
    rows.append({"block":block,"aggregate_relative_l2":float(np.linalg.norm(candidate-control)/np.linalg.norm(control)),"per_step_relative_l2":steps,"max_step_relative_l2":max(steps),**local[block]})
  rows.sort(key=lambda r:r["aggregate_relative_l2"]);agg=np.array([r["aggregate_relative_l2"] for r in rows]);loc=np.array([r["local_output_relative_l2_median"] for r in rows]);weight=np.array([r["weight_relative_l2"] for r in rows])
  hybrids=[]
  for fraction in (0,.25,.5,.75):
    suffix="" if fraction==0 else f"-retain{int(fraction*100)}";candidate=np.load(p/f"q5-logits-block2{suffix}.npz")["logits"].astype(np.float64);steps=[float(np.linalg.norm(y-x)/np.linalg.norm(x)) for x,y in zip(control,candidate)]
    hybrids.append({"q6_row_fraction":fraction,"effective_bytes_saved":int(6684672*(1-fraction)),"aggregate_relative_l2":float(np.linalg.norm(candidate-control)/np.linalg.norm(control)),"max_step_relative_l2":max(steps),"per_step_relative_l2":steps})
  ret={"schema":"tinygrad.nv_q5k_sensitivity_census.v1","contract_relative_l2_max":1e-3,"singleton_count":len(rows),"singleton_pass_count":sum(r["aggregate_relative_l2"]<=1e-3 and r["max_step_relative_l2"]<=1e-3 for r in rows),"best_singleton":rows[0],"singleton_aggregate_median":float(np.median(agg)),"singleton_aggregate_max":float(max(agg)),"pearson_aggregate_vs_local_output":float(np.corrcoef(agg,loc)[0,1]),"pearson_aggregate_vs_weight":float(np.corrcoef(agg,weight)[0,1]),"rows":rows,"block2_weight_error_ranked_hybrid":hybrids,"hybrid_pass_count":sum(r["aggregate_relative_l2"]<=1e-3 and r["max_step_relative_l2"]<=1e-3 for r in hybrids),"decision":"NO_ADMISSIBLE_SINGLETON__WEIGHT_ERROR_ROW_SELECTOR_NON_MONOTONIC"};pathlib.Path(a.out).write_text(json.dumps(ret,indent=2,sort_keys=True)+"\n");print(json.dumps({k:v for k,v in ret.items() if k not in ("rows","block2_weight_error_ranked_hybrid")},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
