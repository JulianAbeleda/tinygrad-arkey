"""Serialize canonical BubbleBeam dimensions for BoltBeam's semantic-campaign CLI."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import argparse, json, pathlib
from extra.llm_research.bubblebeam_futuresight import dimension_mapping, propose_legal_dimensions, classify_coupled_rows
def split_coupled_rows(rows, compiler_facts, baseline, workload_facts):
  return classify_coupled_rows(baseline, rows, workload_facts, compiler_facts)

def export_request(*, semantic_workload:Mapping[str,Any], schedule:Mapping[str,Any], compiler_facts:Mapping[str,Any],
                   axis_choices:Mapping[str,Sequence[Any]], legal_coupled_rows:Sequence[Mapping[str,Any]], **request:Any)->dict[str,Any]:
  facts={"shape":dict(semantic_workload["shape"]),"operands":dict(semantic_workload["operands"])}
  dimensions=dimension_mapping(propose_legal_dimensions(facts,compiler_facts,axis_choices)); legal,rejected=split_coupled_rows(legal_coupled_rows,compiler_facts,{"schedule":dict(schedule)},facts)
  return {"semantic_workload":dict(semantic_workload),"schedule":dict(schedule),"dimensions":dimensions,
          "legal_coupled_rows":[dict(row) for row in legal],"rejected_coupled_rows":rejected,"futuresight_evidence":request.pop("futuresight_evidence",{"assessments":[],"rejections":[]}),"compiler_facts":dict(compiler_facts),**request}

def main(argv=None):
  parser=argparse.ArgumentParser(description="export authoritative semantic campaign request from Tinygrad raw spec")
  parser.add_argument("spec",type=pathlib.Path); parser.add_argument("--out",type=pathlib.Path,required=True); args=parser.parse_args(argv)
  args.out.write_text(json.dumps(export_request(**json.loads(args.spec.read_text())),sort_keys=True,indent=2)+"\n"); return 0
if __name__=="__main__": raise SystemExit(main())
