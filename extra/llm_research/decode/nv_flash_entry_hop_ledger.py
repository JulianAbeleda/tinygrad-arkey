#!/usr/bin/env python3
"""Locate the first production hop that makes the installed wide Flash score cold.

The harness captures one exact common-layer program sequence and live buffers,
then replays cumulative prefixes between a score reheat and the timed score:

  previous gate/up -> previous down -> shared provider -> paired Q -> paired K/V
  -> Q completion -> KV completion -> score

Only the final score is timestamped. The captured cubins and exact launch ABI
are retained so the same checkpoint sequence can be replayed through CUDA/NCU.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_r_residual_cache_dispatch_probe import _make_queue

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
TARGET = os.environ.get("NV_FLASH_ENTRY_TARGET", "flash_vec_llama_score_pv_32_128_8_widekv16")
GATE = "q4k_gate_up_four_warp_vec_fp16_12288_4096"
DOWN = "q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd" + \
       ("_splitstream" if os.environ.get("NV_Q6_FFN_DOWN_SPLIT_WEIGHT_STREAM", "0") == "1" else "")
PROVIDER = "rmsnorm_q8_1_llama_provider_4096"
QPAIR = "q4k_warp_coop_q8_dp4a_direct_4096_4096"
KVPAIR = "q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096"
QDONE = "reduce_output_rmsnorm_rope_32_128"
KVDONE = "reduce_output_rmsnorm_rope_kv_cache_8_128"
CHAIN = (GATE, DOWN, PROVIDER, QPAIR, KVPAIR, QDONE, KVDONE, TARGET)
SELECTED = frozenset(CHAIN)


def _buf_meta(buf) -> dict:
  inner = getattr(buf, "_buf", None)
  va = getattr(buf, "va_addr", getattr(inner, "va_addr", None))
  size = getattr(buf, "size", getattr(inner, "size", None))
  return {"va_addr":int(va) if va is not None else None, "size":int(size) if size is not None else None}


def _capture(depth:int, max_context:int, tokens:int, cubin_dir:pathlib.Path) -> tuple[list[dict], dict]:
  import tinygrad.runtime.ops_nv as ops_nv
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

  trace:list[dict] = []
  cubins:dict[str,dict] = {}
  orig_init, orig_call = ops_nv.NVProgram.__init__, ops_nv.NVProgram.__call__

  def patched_init(self, dev, name, lib, **kwargs):
    orig_init(self, dev, name, lib, **kwargs)
    if name in SELECTED and name not in cubins:
      blob = bytes(lib); cubin_dir.mkdir(parents=True, exist_ok=True)
      path = cubin_dir/f"{name}.cubin"; path.write_bytes(blob)
      cubins[name] = {"path":str(path), "sha256":hashlib.sha256(blob).hexdigest(),
        "regs_usage":getattr(self,"regs_usage",None), "shmem_usage":getattr(self,"shmem_usage",None)}

  def patched_call(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, timeout=None):
    if self.name in SELECTED:
      trace.append({"name":self.name, "program":self, "bufs":tuple(bufs), "vals":tuple(vals),
        "global_size":tuple(global_size), "local_size":tuple(local_size), "buf_meta":[_buf_meta(x) for x in bufs]})
    return orig_call(self,*bufs,global_size=global_size,local_size=local_size,vals=vals,wait=wait,timeout=timeout)

  ops_nv.NVProgram.__init__, ops_nv.NVProgram.__call__ = patched_init, patched_call
  try:
    model = _load(MODEL,max_context); model._decode_direct_greedy_promoted=True; model._decode_feedback_pingpong_promoted=True
    gen = model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
    try: produced=[int(next(gen)) for _ in range(tokens)]; Device["NV"].synchronize()
    finally: gen.close()
  finally: ops_nv.NVProgram.__init__, ops_nv.NVProgram.__call__ = orig_init, orig_call

  names=[x["name"] for x in trace]; found=[]
  for i in range(len(trace)-len(CHAIN)+1):
    if tuple(names[i:i+len(CHAIN)]) == CHAIN: found.append(trace[i:i+len(CHAIN)])
  if not found:
    raise RuntimeError(f"exact common-layer chain not found; selected trace tail={names[-100:]}")
  return found[-1], {"tokens":produced,"selected_call_count":len(trace),"matching_chain_count":len(found),"cubins":cubins}


def _exec(queue, rec:dict) -> None:
  queue.exec(rec["program"],rec["program"].fill_kernargs(rec["bufs"],vals=rec["vals"]),rec["global_size"],rec["local_size"])


def _measure(dev,target:dict,prefix:list[dict],n:int,timeout_s:float) -> list[float]:
  samples=[]
  for _ in range(n):
    q=_make_queue(dev); _exec(q,target)
    for rec in prefix: _exec(q,rec)
    start,end=dev.new_signal(),dev.new_signal(); q.timestamp(start); _exec(q,target); q.timestamp(end)
    q.signal(dev.timeline_signal,dev.next_timeline()).submit(dev); dev.synchronize(timeout=int(timeout_s*1000))
    samples.append(float(end.timestamp-start.timestamp))
  return samples


def _summary(xs:list[float],warmup:int) -> dict:
  ys=xs[warmup:]
  return {"median_us":round(statistics.median(ys),3),"mean_us":round(statistics.mean(ys),3),
    "min_us":round(min(ys),3),"max_us":round(max(ys),3),"samples_us":[round(x,3) for x in ys]}


def _public(rec:dict,cubin:dict) -> dict:
  return {"name":rec["name"],"cubin":cubin,"buf_meta":rec["buf_meta"],"vals":list(rec["vals"]),
    "global_size":list(rec["global_size"]),"local_size":list(rec["local_size"])}


def main() -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--capture-tokens",type=int,default=8); ap.add_argument("--n",type=int,default=48)
  ap.add_argument("--warmup",type=int,default=8); ap.add_argument("--timeout-s",type=float,default=180.0)
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  os.environ.update(DEV="NV",PROFILE="0")
  cubin_dir=pathlib.Path(str(args.out).removesuffix(".json"))/"cubins"
  chain, capture = _capture(args.depth,args.max_context,args.capture_tokens,cubin_dir)
  by_name={x["name"]:x for x in chain}; target=by_name[TARGET]
  from tinygrad import Device
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _gpu_state
  dev=Device["NV"]

  ordered=[by_name[x] for x in CHAIN[:-1]]
  definitions={
    "hot_a":[],
    "previous_gate_up":[by_name[GATE]],
    "previous_ffn":[by_name[GATE],by_name[DOWN]],
    "previous_ffn_provider":ordered[:3],
    "through_q_projection":ordered[:4],
    "through_kv_projection":ordered[:5],
    "through_q_completion":ordered[:6],
    "full_entry_chain":ordered[:7],
    "provider_only":[by_name[PROVIDER]],
    "qkv_only":[by_name[QPAIR],by_name[KVPAIR]],
    "completions_only":[by_name[QDONE],by_name[KVDONE]],
    "hot_c":[],
  }
  rows={}
  for name,prefix in definitions.items():
    rows[name]=_summary(_measure(dev,target,prefix,args.n,args.timeout_s),args.warmup)
    print(json.dumps({name:rows[name]},sort_keys=True),flush=True)
  hot=statistics.median((rows["hot_a"]["median_us"],rows["hot_c"]["median_us"]))
  cumulative=("previous_gate_up","previous_ffn","previous_ffn_provider","through_q_projection",
              "through_kv_projection","through_q_completion","full_entry_chain")
  prior=hot; hops={}
  for name in cumulative:
    current=rows[name]["median_us"]
    hops[name]={"score_minus_hot_us":round(current-hot,3),"increment_from_prior_checkpoint_us":round(current-prior,3)}
    prior=current
  public_chain=[_public(x,capture["cubins"][x["name"]]) for x in chain]
  payload={"schema":"tinygrad.nv_flash_entry_hop_ledger.v1",
    "commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
    "gpu_state":_gpu_state(),"shape":{"depth":args.depth,"max_context":args.max_context},
    "method":"exact common-layer captured programs/live buffers; target reheat; cumulative prefix; only final score timestamped",
    "chain_names":list(CHAIN),"capture":{k:v for k,v in capture.items() if k != "cubins"},"launch_manifest":public_chain,
    "n_per_arm":args.n-args.warmup,"rows":rows,
    "reconciliation":{"hot_midpoint_us":round(hot,3),"cumulative_hops":hops,
      "first_checkpoint_over_hot_0p25us":next((x for x in cumulative if rows[x]["median_us"]-hot>=0.25),None),
      "full_entry_minus_hot_us":round(rows["full_entry_chain"]["median_us"]-hot,3)}}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps(payload["reconciliation"],indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
