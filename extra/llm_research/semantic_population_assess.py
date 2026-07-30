"""Assess a BoltBeam population with FutureSight static legality/priority only."""
from __future__ import annotations
import json, pathlib
from typing import Any, Mapping
from extra.llm_research.bubblebeam_futuresight import candidate_report, build_static_legality, build_static_priority
def assess_population(population:Mapping[str,Any])->dict[str,Any]:
  candidates=[row["candidate"]|{"candidate_hash":row["candidate_hash"]} for row in population["candidates"]]
  legality=build_static_legality(population["workload_facts"],population["compiler_facts"])
  preferences=population["compiler_facts"].get("static_preferences",{})
  return candidate_report(candidates,(legality,),build_static_priority(preferences)) | {"population_hash":population["population_hash"],"rejected_coupled_rows":population.get("rejected_coupled_rows",[])}
def main(argv=None):
  import argparse
  p=argparse.ArgumentParser(description="emit FutureSight static evidence for BoltBeam population JSON");p.add_argument("population",type=pathlib.Path);p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args(argv)
  a.out.write_text(json.dumps(assess_population(json.loads(a.population.read_text())),sort_keys=True,indent=2)+"\n")
  return 0
if __name__=="__main__": raise SystemExit(main())
