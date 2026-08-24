#!/usr/bin/env python3
"""Live-buffer HCQ gate for cache-cold Q || paired-K/V producer overlap.

Capture one installed shared-Q8 Q4/Q4 attention group from a real decode,
retain its exact cubins, immutable weight allocations, launch geometry, and
small-buffer contents, then replay two equivalent arms:

  serial:     provider -> Q -> paired K/V
  concurrent: provider -> { Q on GPFIFO 0 || paired K/V on GPFIFO 1 } -> join

Every timed sample first walks 128 MiB through a separate flush kernel.  The
flush completion is the start timestamp, so eviction is required but not
charged.  This tool mutates no production scheduler or route policy.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import (  # noqa: E402
  FLUSH_BLOCK, FLUSH_GRID, FLUSH_MIB, _alloc, _compile_flush)

SCHEMA = "tinygrad.nv_qkv_live_hcq_concurrency.v1"
PROVIDER = "rmsnorm_q8_1_llama_provider_4096"
QPROJ = "q4k_warp_coop_q8_dp4a_direct_4096_4096"
KVPAIR = "q4k_warp_coop_q8_dp4a_pair_direct_1024_4096"
SELECTED = (PROVIDER, QPROJ, KVPAIR)
WEIGHT_BYTES_MIN = 1 << 20
ENDPOINT_US = 4515.39571875

P_OUT, P_IN, P_W = 0, 1, 2
Q_OUT, Q_W, Q_IN = 0, 1, 2
KV_KOUT, KV_VOUT, KV_KW, KV_VW, KV_IN = 0, 1, 2, 3, 4


def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _va(buf) -> int:
  va = getattr(buf, "va_addr", None)
  if va is None: va = getattr(getattr(buf, "_buf", None), "va_addr", None)
  return int(va)


def _size(buf) -> int:
  size = getattr(buf, "size", None)
  if size is None: size = getattr(getattr(buf, "_buf", None), "size", None)
  return int(size)


def _copy(dev, buf) -> bytes:
  host = memoryview(bytearray(_size(buf)))
  dev.allocator._copyout(host, buf)
  return bytes(host)


def _clone(dev, buf):
  data = _copy(dev, buf)
  out = _alloc(dev, len(data))
  dev.allocator._copyin(out, memoryview(bytearray(data)))
  return out


def _zero(dev, size: int):
  out = _alloc(dev, size)
  dev.allocator._copyin(out, memoryview(bytearray(size)))
  return out


def _checksum(dev, buf) -> str: return _sha(_copy(dev, buf))


def _record(prg, bufs, vals, global_size, local_size, seq: int) -> dict[str, Any]:
  return {"name": prg.name, "program": prg, "live": tuple(bufs), "vals": tuple(vals),
          "global_size": tuple(global_size), "local_size": tuple(local_size), "seq": seq,
          "buf_va": [_va(x) for x in bufs], "buf_size": [_size(x) for x in bufs],
          "cubin_sha": _sha(bytes(prg.lib))}


def _freeze(dev, rec: dict[str, Any]) -> dict[str, Any]:
  out = dict(rec)
  out["snapshots"] = tuple(_clone(dev, b) if _size(b) < WEIGHT_BYTES_MIN else None for b in rec["live"])
  return out


def install_hook(cap: dict[str, Any], target_occurrence: int):
  import tinygrad.runtime.ops_nv as ops_nv
  original = ops_nv.NVProgram.__call__
  recent: list[dict[str, Any]] = []
  pair_seen, seq = 0, 0

  def patched(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, timeout=None):
    nonlocal pair_seen, seq
    name, my_seq = self.name, seq
    seq += 1
    selected_pair = name == KVPAIR and pair_seen == target_occurrence
    ret = original(self, *bufs, global_size=global_size, local_size=local_size, vals=vals,
                   wait=(wait or selected_pair), timeout=timeout)
    if name in SELECTED:
      rec = _record(self, bufs, vals, global_size, local_size, my_seq)
      if selected_pair:
        xp_va = rec["buf_va"][KV_IN]
        q = next((x for x in reversed(recent) if x["name"] == QPROJ and x["buf_va"][Q_IN] == xp_va), None)
        p = next((x for x in reversed(recent) if x["name"] == PROVIDER and x["buf_va"][P_OUT] == xp_va), None)
        if q is None or p is None or not (p["seq"] < q["seq"] < rec["seq"]):
          raise RuntimeError(f"pair occurrence {target_occurrence} has no exact provider/Q prefix: "
                             f"xp={xp_va:#x} recent={[x['name'] for x in recent[-12:]]}")
        dev = self.dev
        cap[PROVIDER], cap[QPROJ], cap[KVPAIR] = _freeze(dev, p), _freeze(dev, q), _freeze(dev, rec)
      recent.append(rec)
      if len(recent) > 64: del recent[0]
    if name == KVPAIR: pair_seen += 1
    return ret

  ops_nv.NVProgram.__call__ = patched
  return original


def restore_hook(original) -> None:
  import tinygrad.runtime.ops_nv as ops_nv
  ops_nv.NVProgram.__call__ = original


def run_decode(cap: dict[str, Any], occurrence: int, model: str, decode_tokens: int) -> list[int]:
  from tinygrad import Device
  from tinygrad.llm.generate import load_model_and_tokenizer
  original = install_hook(cap, occurrence)
  tokens: list[int] = []
  try:
    dev = Device[Device.DEFAULT]
    mdl, tok = load_model_and_tokenizer(model, 4608, seed=20260617)
    base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
    gen = mdl.generate(base[:512].copy(), chunk_size=32, temperature=0.0)
    try:
      for _ in range(decode_tokens): tokens.append(int(next(gen)))
      dev.synchronize()
    finally:
      gen.close()
  finally:
    restore_hook(original)
  return tokens


def _assert_capture(cap: dict[str, Any], occurrence: int) -> dict[str, Any]:
  missing = [x for x in SELECTED if x not in cap]
  if missing: raise RuntimeError(f"capture missing {missing}; got={sorted(cap)}")
  p, q, kv = (cap[x] for x in SELECTED)
  expected = {PROVIDER: [8192, 16384, 8192], QPROJ: [16384, 9437184, 8192],
              KVPAIR: [4096, 4096, 2359296, 2359296, 8192]}
  for name, rec in ((PROVIDER,p),(QPROJ,q),(KVPAIR,kv)):
    if rec["buf_size"] != expected[name]: raise RuntimeError(f"layout mismatch {name}: {rec['buf_size']} != {expected[name]}")
  deps = {"provider_to_q": p["buf_va"][P_OUT] == q["buf_va"][Q_IN],
          "provider_to_kv": p["buf_va"][P_OUT] == kv["buf_va"][KV_IN]}
  if not all(deps.values()) or not p["seq"] < q["seq"] < kv["seq"]:
    raise RuntimeError(f"not an exact producer chain: deps={deps} seq={[p['seq'],q['seq'],kv['seq']]}")
  return {"target_pair_occurrence": occurrence, "dependencies": deps,
          "sequence": {name: cap[name]["seq"] for name in SELECTED}}


def _input(rec: dict[str, Any], idx: int):
  snap = rec["snapshots"][idx]
  return snap if snap is not None else rec["live"][idx]


def _launch(rec: dict[str, Any], bufs) -> tuple[Any, tuple, tuple, tuple, tuple]:
  return rec["program"], tuple(bufs), rec["vals"], rec["global_size"], rec["local_size"]


def _build_arm(dev, cap: dict[str, Any]) -> dict[str, Any]:
  p, q, kv = (cap[x] for x in SELECTED)
  xp, qo, ko, vo = _zero(dev, p["buf_size"][P_OUT]), _zero(dev, q["buf_size"][Q_OUT]), \
                       _zero(dev, kv["buf_size"][KV_KOUT]), _zero(dev, kv["buf_size"][KV_VOUT])
  pbufs = [xp, _input(p, P_IN), _input(p, P_W)]
  qbufs = [qo, _input(q, Q_W), xp]
  kvbufs = [ko, vo, _input(kv, KV_KW), _input(kv, KV_VW), xp]
  return {"provider": _launch(p,pbufs), "q": _launch(q,qbufs), "kv": _launch(kv,kvbufs),
          "outputs": (xp,qo,ko,vo)}


def _queue(dev, idx: int):
  from tinygrad.runtime.ops_nv import NVComputeQueue
  q = NVComputeQueue(queue_idx=idx)
  q.setup(compute_class=dev.iface.compute_class, local_mem_window=dev.local_mem_window,
          shared_mem_window=dev.shared_mem_window)
  q.wait(dev.timeline_signal, dev.timeline_value-1).memory_barrier()
  return q


def _exec(q, launch) -> None:
  prg, bufs, vals, grid, block = launch
  q.exec(prg, prg.fill_kernargs(bufs, vals=vals), grid, block)


def _run_control(dev, flush_prg, flush_buf, arm: dict[str, Any], timeout_s: float) -> float:
  q = _queue(dev, 0)
  _exec(q, (flush_prg, (flush_buf,), (), FLUSH_GRID, (FLUSH_BLOCK,1,1)))
  st, en = dev.new_signal(), dev.new_signal()
  q.timestamp(st)
  for name in ("provider","q","kv"): _exec(q, arm[name])
  q.timestamp(en)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize(timeout=int(timeout_s*1000))
  return float(en.timestamp-st.timestamp)


def _run_concurrent(dev, flush_prg, flush_buf, arm: dict[str, Any], timeout_s: float) -> float:
  if len(dev.compute_gpfifos) < 2: raise RuntimeError("live HCQ concurrency requires HCQ_NUM_COMPUTE>=2 at device construction")
  parent, sibling = _queue(dev,0), _queue(dev,1)
  _exec(parent, (flush_prg,(flush_buf,),(),FLUSH_GRID,(FLUSH_BLOCK,1,1)))
  st, en, ready, qdone, kvdone = (dev.new_signal() for _ in range(5))
  parent.timestamp(st)
  _exec(parent,arm["provider"]); parent.signal(ready,1)
  _exec(parent,arm["q"]); parent.signal(qdone,1)
  parent.wait(qdone,1).wait(kvdone,1).timestamp(en)
  sibling.wait(ready,1); _exec(sibling,arm["kv"]); sibling.signal(kvdone,1)
  target = dev.next_timeline()
  parent.signal(dev.timeline_signal,target)
  sibling.submit(dev)
  parent.submit(dev)
  dev.synchronize(timeout=int(timeout_s*1000))
  return float(en.timestamp-st.timestamp)


def _gpu_state() -> dict[str,str]:
  fields=("name","driver_version","persistence_mode","pstate","clocks.sm","clocks.mem","power.draw","temperature.gpu")
  vals=[x.strip() for x in subprocess.check_output(["nvidia-smi",f"--query-gpu={','.join(fields)}","--format=csv,noheader,nounits"],
                                                  text=True).strip().split(",")]
  return dict(zip(fields,vals))


def main() -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--occurrence",type=int,default=0)
  ap.add_argument("--n",type=int,default=28)
  ap.add_argument("--warmup",type=int,default=4)
  ap.add_argument("--order",default="control,candidate,candidate,control")
  ap.add_argument("--decode-tokens",type=int,default=4)
  ap.add_argument("--timeout-s",type=float,default=180.0)
  ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  order=[x.strip() for x in args.order.split(",") if x.strip()]
  if set(order)!={"control","candidate"}: raise SystemExit(f"order must include control and candidate: {order}")
  from tinygrad import Device
  before=_gpu_state(); cap:dict[str,Any]={}
  tokens=run_decode(cap,args.occurrence,args.model,args.decode_tokens)
  dev=Device["NV"]
  identity=_assert_capture(cap,args.occurrence)
  flush_blob=_compile_flush(dev); flush_prg=__import__("tinygrad.runtime.ops_nv",fromlist=["NVProgram"]).NVProgram(
    dev,f"nv_r_flush_{FLUSH_MIB}mib",flush_blob)
  flush_buf=_alloc(dev,FLUSH_MIB*1024*1024)
  dev.allocator._copyin(flush_buf,memoryview(bytearray(flush_buf.size))); dev.synchronize()
  control_arm,candidate_arm=_build_arm(dev,cap),_build_arm(dev,cap)

  _run_control(dev,flush_prg,flush_buf,control_arm,args.timeout_s)
  expected=tuple(_checksum(dev,x) for x in control_arm["outputs"])
  _run_concurrent(dev,flush_prg,flush_buf,candidate_arm,args.timeout_s)
  got=tuple(_checksum(dev,x) for x in candidate_arm["outputs"])
  if got!=expected: raise RuntimeError(f"concurrent output mismatch: got={got} expected={expected}")

  runs=[]; by={"control":[],"candidate":[]}
  for name in order:
    fn=_run_control if name=="control" else _run_concurrent
    vals=[fn(dev,flush_prg,flush_buf,control_arm if name=="control" else candidate_arm,args.timeout_s) for _ in range(args.n)]
    vals=vals[args.warmup:]; by[name].extend(vals)
    runs.append({"arm":name,"durations_us":[round(x,3) for x in vals],"median_us":round(statistics.median(vals),3)})
  cm,pm=statistics.median(by["control"]),statistics.median(by["candidate"]); recovery=cm-pm
  route_recovery,global_ceiling=recovery*9,recovery*36
  result={"schema":SCHEMA,"commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
    "gpu_state_before":before,"gpu_state_after":_gpu_state(),"capture_identity":identity,
    "token_ids":tokens,"token_sha256":_sha(",".join(map(str,tokens)).encode()),
    "cubin_sha256":{name:cap[name]["cubin_sha"] for name in SELECTED},
    "buffer_sizes":{name:cap[name]["buf_size"] for name in SELECTED},
    "buffer_vas":{name:cap[name]["buf_va"] for name in SELECTED},
    "output_checksums":{"control":expected,"candidate":got,"bitwise_identical":got==expected},
    "order":order,"n_per_block":args.n,"warmup_per_block":args.warmup,"runs":runs,
    "timing":{"unit":"us_per_cache_cold_provider_to_join_span","control_median":round(cm,3),
              "candidate_median":round(pm,3),"recovery_us_per_layer":round(recovery,3),"candidate_over_control":pm/cm},
    "projection":{"endpoint_us":ENDPOINT_US,"endpoint_tok_s":1e6/ENDPOINT_US,
                  "installed_shared_q4q4_layers":9,"installed_route_recovery_us":round(route_recovery,3),
                  "installed_route_projected_tok_s":1e6/(ENDPOINT_US-route_recovery),
                  "all_36_layers_unvalidated_ceiling_us":round(global_ceiling,3),
                  "all_36_layers_unvalidated_ceiling_tok_s":1e6/(ENDPOINT_US-global_ceiling)},
    "verdict":"ADVANCE_TO_SCHEDULER_SCOPE" if got==expected and recovery>0.5 else "NO_GO"}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
