#!/usr/bin/env python3
"""Matched pp512 attention-core census for the retained Qwen3 NV route.

The forward is warmed and then profiled asynchronously.  No synchronize is
performed inside the profiled forward; the final device sync only closes the
profile window.  Role classes intentionally mirror llama_prefill_exact_trace.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re
from collections import defaultdict

os.environ.setdefault("DEV", "NV")
from tinygrad import Tensor
from tinygrad.device import Compiled
from tinygrad.llm.model import Transformer
from tinygrad.llm.runtime_state import SimpleTokenizer
from tinygrad.device import Device

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
ROLES = {
  "Q": ("q8", "q_proj", "q_"), "K": ("k8", "k_proj", "k_"),
  "V": ("v8", "v_proj", "v_"), "O": ("o8", "o_proj", "o_"),
  "RoPE/norm/KV support": ("rope", "rms_norm", "norm", "kv", "set_rows"),
  "Flash score/reduction": ("flash", "attn", "attention"),
}

def name_of(e):
  return str(getattr(e, "key", None) or getattr(e, "name", None) or getattr(e, "prg", None) or e)

def classify(name):
  low = name.lower()
  if "flash" in low or "attention" in low or "attn" in low: return "Flash score/reduction"
  if "rope" in low or "norm" in low or "kv" in low or "set_rows" in low: return "RoPE/norm/KV support"
  for role, needles in (("Q", ROLES["Q"]), ("K", ROLES["K"]), ("V", ROLES["V"]), ("O", ROLES["O"])):
    if any(n in low for n in needles): return role
  return None

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--model", default=MODEL); ap.add_argument("--out", type=pathlib.Path)
  a = ap.parse_args()
  model, kv = Transformer.from_gguf(a.model, 2048)
  SimpleTokenizer.from_gguf_kv(kv)
  prompt = Tensor([[(i % 1000) + 1 for i in range(512)]], dtype="int32").contiguous()
  temp = Tensor([0.0])
  for _ in range(2): model(prompt, 0, temp, use_flash=True).realize()
  dev = Device[Device.DEFAULT]; dev.synchronize()
  start = len(Compiled.profile_events)
  # This region contains only asynchronous production launches.
  model(prompt, 0, temp, use_flash=True).realize()
  events = list(Compiled.profile_events[start:])
  dev.synchronize()
  rows = defaultdict(lambda: {"events": 0, "us": 0.0, "names": []})
  unknown = {"events": 0, "us": 0.0, "names": []}
  for e in events:
    if getattr(e, "en", None) is None: continue
    name = name_of(e); role = classify(name)
    t0, t1 = getattr(e, "st", None), getattr(e, "en", None)
    us = (float(t1) - float(t0)) if t0 is not None else 0.0
    dest = rows[role] if role else unknown
    dest["events"] += 1; dest["us"] += us
    if len(dest["names"]) < 12: dest["names"].append(name)
  total = sum(x["us"] for x in rows.values()) + unknown["us"]
  result = {"schema":"tinygrad.qwen3_attention_core_census.v1", "workload":{"model":a.model,"tokens":512,"start_pos":0,"route_env":{k:os.environ.get(k) for k in os.environ if k.startswith("NV_LLAMA_PACKED")}}, "async_profile":{"event_count":sum(x["events"] for x in rows.values())+unknown["events"],"profile_us":total,"host_sync_inside_region":False}, "roles":{k:{**v,"share_pct":100*v["us"]/total if total else 0} for k,v in rows.items()}, "unknown":unknown, "llama_trace_classes":["Q","K","V","Flash score/reduction","O","RoPE/norm/KV support"], "largest_nonoverlap":max(((k,v["us"]) for k,v in rows.items()),key=lambda x:x[1],default=(None,0))}
  text=json.dumps(result,sort_keys=True)
  if a.out: a.out.write_text(text+"\n")
  print(text)
if __name__ == "__main__": main()
