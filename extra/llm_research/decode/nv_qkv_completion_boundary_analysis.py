#!/usr/bin/env python3
"""Measure O44 producer/consumer readiness from raw HCQ graph profiles.

This is deliberately a profile discriminator, not a wall benchmark.  It pairs
each installed Q + K/V producer or full-QKV producer with its Q norm, K norm,
cache store, flash score, and attention-output consumer in the same graph.
All offsets are relative to the common input RMSNorm completion.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics

Q = "q4k_g3_lanemap_gemv_vec_4096_4096"
PAIR = "q4k_g3_lanemap_gemv_pair_vec_1024_4096"
FULL = "q4k_g3_lanemap_gemv_qkv_full_4096_1024_4096"


def _f(entry:dict, key:str) -> float: return float(entry[key])


def _following(entries:list[dict], start:int, prefix:str) -> tuple[int,dict]:
  for index in range(start+1, min(len(entries), start+10)):
    if entries[index]["name"].startswith(prefix): return index, entries[index]
  raise ValueError(f"missing {prefix!r} after entry {start}")


def _samples(path:pathlib.Path, candidate:bool) -> list[dict]:
  rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  # The final nine occurrences of each graph size are the settled replay set.
  selected=[]
  for size in (32,64,128,256,10):
    matches=[row for row in rows if len(row.get("entries",())) == size]
    selected.extend(matches[-9:])
  out=[]
  for graph_index,row in enumerate(selected):
    entries=row["entries"]
    for index,producer in enumerate(entries):
      if candidate:
        if producer["name"] != FULL: continue
        pred=entries[index-1]
        if pred["name"] != "reduce_output_rmsnorm_1_4096": raise ValueError("full QKV predecessor mismatch")
        qnorm_i,qnorm=_following(entries,index,"reduce_output_rmsnorm_rope_32_128")
        knorm_i,knorm=_following(entries,index,"reduce_output_rmsnorm_rope_8_128")
        store_i,store=_following(entries,knorm_i,"E_8_8_16_2_")
        _flash_i,flash=_following(entries,store_i,"flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128")
        _out_i,attn_out=_following(entries,_flash_i,"flash_fused_gmax_combine_f16_32_128")
        _out_i,attn_out=_following(entries,_out_i,"q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096")
        base=_f(pred,"end")
        out.append({"graph_index":graph_index,"entry_index":index,
          "q_ready_us":_f(producer,"end")-base,"kv_ready_us":_f(producer,"end")-base,
          "q_norm_start_us":_f(qnorm,"start")-base,"k_norm_start_us":_f(knorm,"start")-base,
          "cache_ready_us":_f(store,"end")-base,"flash_start_us":_f(flash,"start")-base,
          "attention_output_ready_us":_f(attn_out,"end")-base,"producer_span_us":_f(producer,"duration")})
      else:
        if producer["name"] != PAIR or index == 0 or entries[index-1]["name"] != Q: continue
        q=entries[index-1]; pred=entries[index-2]
        if pred["name"] != "reduce_output_rmsnorm_1_4096": raise ValueError("Q/pair predecessor mismatch")
        _qnorm_i,qnorm=_following(entries,index-1,"reduce_output_rmsnorm_rope_32_128")
        knorm_i,knorm=_following(entries,index,"reduce_output_rmsnorm_rope_8_128")
        store_i,store=_following(entries,knorm_i,"E_8_8_16_2_")
        flash_i,flash=_following(entries,store_i,"flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128")
        combine_i,_combine=_following(entries,flash_i,"flash_fused_gmax_combine_f16_32_128")
        _out_i,attn_out=_following(entries,combine_i,"q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096")
        base=_f(pred,"end")
        out.append({"graph_index":graph_index,"entry_index":index,
          "q_ready_us":_f(q,"end")-base,"kv_ready_us":_f(producer,"end")-base,
          "q_norm_start_us":_f(qnorm,"start")-base,"k_norm_start_us":_f(knorm,"start")-base,
          "cache_ready_us":_f(store,"end")-base,"flash_start_us":_f(flash,"start")-base,
          "attention_output_ready_us":_f(attn_out,"end")-base,
          "producer_span_us":max(_f(q,"end"),_f(producer,"end"))-min(_f(q,"start"),_f(producer,"start"))})
  # Nine final rows per graph size do not necessarily form nine complete token
  # cycles when the capture ends between graph groups.  Retain the bounded raw
  # sample and prove coverage by requiring an integral number of 9-block cycles.
  if len(out) < 72 or len(out) % 9: raise ValueError(f"expected complete 9-block samples, found {len(out)} in {path}")
  return out


def _medians(rows:list[dict]) -> dict:
  keys=[key for key in rows[0] if key.endswith("_us")]
  return {key:round(statistics.median(row[key] for row in rows),3) for key in keys}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--control",type=pathlib.Path,required=True)
  ap.add_argument("--candidate",type=pathlib.Path,required=True); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args(); control=_samples(args.control,False); candidate=_samples(args.candidate,True)
  cm,dm=_medians(control),_medians(candidate)
  result={"schema":"tinygrad.nv_qkv_completion_boundary_analysis.v1","sample_count":{"control":len(control),"candidate":len(candidate)},
    "population_blocks":9,"bounded_steady_replays":{"control":len(control)//9,"candidate":len(candidate)//9},
    "control_median_us":cm,"candidate_median_us":dm,
    "candidate_minus_control_us":{key:round(dm[key]-cm[key],3) for key in cm},
    "decision":("DOWNSTREAM_READINESS_NOT_CAUSAL" if dm["attention_output_ready_us"] < cm["attention_output_ready_us"]
                and dm["flash_start_us"] < cm["flash_start_us"] else "READINESS_WALL_REMAINS"),
    "interpretation":"The full producer delays Q-only readiness but advances the cache join, flash start, and attention-output completion; re-bracket the contradictory unprofiled wall before changing CTA geometry.",
    "sources":{"control":str(args.control),"candidate":str(args.candidate)}}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
