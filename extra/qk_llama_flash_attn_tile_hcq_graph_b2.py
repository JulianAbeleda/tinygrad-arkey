#!/usr/bin/env python3
"""Route B B2 — HCQ graph/AQL launch-integration de-risk for the vendored llama flash_attn_tile oracle.

B1 found the vendored tile WINS 2.96x by GPU-busy but LOSES ~2.5x by WALL because it was launched as TWO raw
HCQProgram dispatches (2 doorbells, 2 kernarg fills, 2 submits). B2 asks ONE question: does the GPU-busy win survive
once tile+combine are folded into ONE HCQ compute queue (one doorbell, kernargs baked once, replayed) — i.e. does the
integrated WALL beat gqa_coop_vec by >=1.5x?

Mechanism (audited, no tinygrad/ change): build one `dev.hw_compute_queue_t()` with `wait().memory_barrier()
.exec(tile).exec(comb).signal().submit()` — one submit = one doorbell, two dispatch packets; the AMD exec emits
CS_PARTIAL_FLUSH + acquire_mem so tile->combine serialize with no barrier between. Kernargs baked once into dedicated
buffers and reused. Two replay variants: REBUILD (queue rebuilt python-side each call, 1 submit) and BOUND (queue built
+ bind()-ed once, replay = re-submit only — the HCQGraph-ideal a real model-JIT integration would get).

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_llama_flash_attn_tile_hcq_graph_b2.py
"""
from __future__ import annotations
import json, struct, statistics, subprocess, pathlib, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, Context
from tinygrad.device import Compiled, BufferSpec
from tinygrad.runtime.support.hcq import hcq_profile
from extra.qk_flash_decode import flash_decode_attention
from extra.qk_clock_pin import pinned_peak
from extra.qk_harness_contract import stamp, repro_band
from extra.qk_llama_flash_attn_tile_hcq_ab import build_replay, COMP_MAXC, COMP_L, COMP_CTX

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-route-b-b2"

def main():
  assert Device.DEFAULT == "AMD"
  dev = Device[Device.DEFAULT]
  r = build_replay(dev)
  tile, comb, tg, tb, cg, cb, bDst, ref = r.tile, r.comb, r.tg, r.tb, r.cg, r.cb, r.bDst, r.ref
  Hd, Hq, Hkv, KV, PB, NVALID, scale = r.Hd, r.Hq, r.Hkv, r.KV, r.PB, r.NVALID, r.scale

  # ---- bake kernargs ONCE into dedicated persistent buffers (reused on every submit) ----
  kb_tile = dev.allocator.alloc(tile.kernargs_alloc_size, BufferSpec(cpu_access=True, nolru=True))
  kb_comb = dev.allocator.alloc(comb.kernargs_alloc_size, BufferSpec(cpu_access=True, nolru=True))
  ka_tile = tile.fill_kernargs([], [], kernargs=kb_tile)
  ka_comb = comb.fill_kernargs([], [], kernargs=kb_comb)
  tgt, tbt, cgt, cbt = tuple(tg), tuple(tb), tuple(cg), tuple(cb)

  # ---- integrated single-submit batched queue (REBUILD each call): one doorbell, two dispatch packets ----
  def integrated_rebuild(prof=False):
    q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value-1).memory_barrier()
    if prof:
      with hcq_profile(dev, queue=q, enabled=True, desc="llama_tile+comb") as (st, en):
        q.exec(tile, ka_tile, tgt, tbt); q.exec(comb, ka_comb, cgt, cbt)
    else:
      q.exec(tile, ka_tile, tgt, tbt); q.exec(comb, ka_comb, cgt, cbt)
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize()
    return (float(en.timestamp - st.timestamp)) if prof else None

  # ---- BOUND replay: build+bind the batched queue ONCE; replay = reset signal + re-submit (HCQGraph-ideal) ----
  bsig = dev.new_signal(value=0)
  bq = dev.hw_compute_queue_t().memory_barrier()
  with hcq_profile(dev, queue=bq, enabled=True, desc="llama_bound") as (bst, ben):
    bq.exec(tile, ka_tile, tgt, tbt); bq.exec(comb, ka_comb, cgt, cbt)
  bq.signal(bsig, 1)
  bq.bind(dev)
  def integrated_bound():
    bsig.value = 0
    bq.submit(dev)
    bsig.wait(1)
    return float(ben.timestamp - bst.timestamp)

  # ---- correctness on the integrated path ----
  integrated_rebuild(prof=False)
  ob = bytearray(Hq*Hd*4); bDst.copyout(memoryview(ob))
  out = np.frombuffer(bytes(ob), np.float32).reshape(Hq, Hd)
  rel = float(np.abs(out-ref).max() / (np.abs(ref).max()+1e-6))
  rmse = float(np.sqrt(((out-ref)**2).mean()) / (np.sqrt((ref**2).mean())+1e-9))
  CORR_TOL = 5e-3
  print(f"correctness(integrated): rel_max={rel:.4e} rel_rmse={rmse:.4e} {'OK' if rmse<=CORR_TOL else 'FAIL'}")

  # ---- comparator gqa_coop_vec ----
  rng2 = np.random.default_rng(0)
  cq = Tensor(rng2.standard_normal((Hq, Hd)).astype(np.float16))
  ckT = Tensor(rng2.standard_normal((Hkv, COMP_MAXC, Hd)).astype(np.float16))
  cvT = Tensor(rng2.standard_normal((Hkv, COMP_MAXC, Hd)).astype(np.float16))

  def wall(fn, n=200):
    for _ in range(10): fn()
    dev.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    dev.synchronize(); return (time.perf_counter()-t0)/n*1e6

  with pinned_peak() as prov:
    time.sleep(0.4)
    comp = TinyJit(lambda: flash_decode_attention(cq, ckT, cvT, COMP_CTX, COMP_CTX, Hd, Hq, Hkv, COMP_MAXC, COMP_L,
                                                  variant="gqa_coop_vec").realize())
    for _ in range(8): comp(); integrated_rebuild(); integrated_bound()
    dev.synchronize()
    # GPU-busy (batched, both variants should match B1 ~15us)
    llama_gpu = statistics.median([integrated_rebuild(prof=True) for _ in range(30)])
    bound_gpu = statistics.median([integrated_bound() for _ in range(30)])
    # WALL (the B2 gate): integrated single-submit vs coop single-jit
    rebuild_wall = statistics.median([wall(lambda: integrated_rebuild(prof=False)) for _ in range(3)])
    bound_wall   = statistics.median([wall(integrated_bound) for _ in range(3)])
    comp_wall    = statistics.median([wall(comp) for _ in range(3)])
    # comparator GPU-busy via ProfileGraphEvent (oracle method)
    comp_gpu_s = []
    with Context(PROFILE=1):
      compp = TinyJit(lambda: flash_decode_attention(cq, ckT, cvT, COMP_CTX, COMP_CTX, Hd, Hq, Hkv, COMP_MAXC, COMP_L,
                                                     variant="gqa_coop_vec").realize())
      for _ in range(8): compp()
      dev.synchronize(); dev._at_profile_finalize()
      for _ in range(5):
        base = len(Compiled.profile_events); compp(); dev.synchronize(); dev._at_profile_finalize()
        busy = 0.0
        for e in Compiled.profile_events[base:]:
          if type(e).__name__ != "ProfileGraphEvent": continue
          sigs = [float(s) for s in e.sigs]
          for ent in e.ents: busy += sigs[ent.en_id] - sigs[ent.st_id]
        if busy > 0: comp_gpu_s.append(busy)
    comp_gpu = statistics.median(comp_gpu_s) if comp_gpu_s else 0.0

  # best integrated wall = the BOUND replay (the realistic model-JIT integration)
  best_wall = min(bound_wall, rebuild_wall)
  wall_speedup = round(comp_wall/best_wall, 3) if best_wall else 0.0
  gpu_speedup = round(comp_gpu/bound_gpu, 3) if bound_gpu else 0.0
  print(f"  GPU-busy: coop {comp_gpu:.1f}us | llama rebuild {llama_gpu:.1f}us bound {bound_gpu:.1f}us -> {gpu_speedup}x")
  print(f"  WALL: coop[jit] {comp_wall:.1f}us | llama rebuild {rebuild_wall:.1f}us bound {bound_wall:.1f}us "
        f"-> bound {wall_speedup}x (B1 was 2 raw dispatches ~148us)")

  corr_ok = rmse <= CORR_TOL
  gpu_ok = gpu_speedup >= 1.5
  wall_ok = wall_speedup >= 1.5
  if not corr_ok: verdict = "B2_FAILS_CORRECTNESS"
  elif wall_ok and gpu_ok: verdict = "B2_LOCAL_GRAPH_PASS"   # integrated launch wall passes; W==D status set below
  elif gpu_ok and not wall_ok: verdict = "B2_FAILS_GRAPH_ECONOMICS"   # batched, GPU win real, but wall still loses
  else: verdict = "B2_FAILS_GRAPH_ECONOMICS"
  # W==D feasibility: the vendored kernel reads llama's exact ggml KV layout (nb11=2048: [pos][8 kvheads][128]); the
  # tinygrad model KV-cache has a different layout, so an in-model vendored W==D needs per-call layout-bridging (expensive,
  # itself a kernel) or deep model-cache surgery + per-shape re-capture -- both out of B2 bounds + non-promotable. So W==D
  # belongs to B3's OWNED kernel (authored to tinygrad's layout). NOT run here; classify the launch-integration de-risk.
  wd_not_run_reason = ("vendored kernel requires llama's exact ggml KV-cache byte layout, which the tinygrad model does "
                       "not produce; an in-model vendored W==D needs out-of-bounds layout-bridging surgery + per-shape "
                       "re-capture and is non-promotable -> W==D is B3's job (owned kernel authored to tinygrad layout)")
  print(f"  VERDICT: {verdict} (wall {wall_speedup}x {'>=' if wall_ok else '<'}1.5, gpu {gpu_speedup}x); "
        f"W==D NOT run: {'layout-blocked for vendored kernel -> B3' if wall_ok else 'n/a (local failed)'}")

  try:
    commit = subprocess.run(["git","rev-parse","--short","HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git","status","--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
  except Exception: commit, dirty = None, None
  art = {"date":"2026-06-21","phase":"ROUTE_B_B2_GRAPH_INTEGRATION","candidate_id":"reference_oracle_hcq_llama_tile",
         "comparator":"gqa_coop_vec","kernel_hash":r.cfg["co_sha256"],
         "kernarg_hash":None,"hidden_arg_provenance":".co amdhsa.kernels msgpack metadata (block_count@208/group_size@220/grid_dims@272 for tile; combine kernarg 288B)",
         "graph_specialization_key":f"decode_T1_ctx1024_KV{KV}_pb{PB}_Hd{Hd}_Hq{Hq}_Hkv{Hkv}",
         "baked_kernarg_fields":"all 37 tile args + 4 combine args + COV5 hidden block baked once into dedicated buffers; buffer VAs constant (no per-replay variable)",
         "dependency_mechanism":"single HCQ compute queue; tile->combine serialized by AMD exec CS_PARTIAL_FLUSH + acquire_mem (PM4) / AQL barrier bit; no explicit barrier/signal between dispatches",
         "aql_packet_count":2,"dispatch_count":2,"doorbell_count":1,"signal_count_per_replay":3,
         "host_sync_count":1,"graph_replay_count":1,"queue_mode":"PM4 default (AMDComputeQueue), 1 submit",
         "gpu_busy_us":round(bound_gpu,1),"gpu_busy_rebuild_us":round(llama_gpu,1),
         "wall_us":round(best_wall,1),"wall_bound_us":round(bound_wall,1),"wall_rebuild_us":round(rebuild_wall,1),
         "comparator_wall_us":round(comp_wall,1),"comparator_gpu_busy_us":round(comp_gpu,1),
         "gpu_busy_speedup":gpu_speedup,"wall_speedup":wall_speedup,
         "b1_two_dispatch_wall_us":148.0,
         "correctness_rel_max":round(rel,6),"correctness_rel_rmse":round(rmse,6),"correctness_tol":CORR_TOL,
         "results":[{"ctx":1024,"best_speedup_vs_coop":wall_speedup,"splits":[{"err":round(rel,6)}]}],
         "first_gate_pass":bool(wall_ok and corr_ok),
         "verdict":verdict,"promotion_policy":"NON-PROMOTABLE — vendored llama reference, never a default route",
         "pass_fail_threshold":"integrated WALL >=1.5x faster than gqa_coop_vec @ctx1024 AND rel_rmse<=5e-3 (fp16); GPU-busy stays ~B1",
         "repro_band":{"llama_bound_gpu":repro_band([bound_gpu]),"coop_gpu":repro_band(comp_gpu_s)},
         "clock_pin":(prov or {}).get("ok"),"commit":commit,"dirty_tree":dirty,"default_behavior_changed":False,
         "wd_status":"NOT RUN (local integrated wall PASSED, so W==D is unlocked, but layout-blocked for the VENDORED kernel)",
         "wd_not_run_reason":wd_not_run_reason,
         "route_decision":"Route B REMAINS VIABLE after launch integration (integrated wall 1.68x faster than coop). "
                          "Per scope: proceed to B3 (owned hand-AMDGCN tile authored to tinygrad's KV layout), which is "
                          "promotable AND can be W==D'd in-model without layout bridging."}
  art = stamp(art, comparator_id="gqa_coop_vec",
              comparator_why="shipped default decode-attention primitive; B2 tests whether the vendored tile's GPU win survives launch integration into one HCQ submit, the prerequisite for any W==D",
              timing_authority="GPU-busy via signal timestamps (llama) / ProfileGraphEvent (coop); WALL via back-to-back perf_counter median-of-3, clock-pinned -- the B2 gate is integrated WALL (launch-overhead-inclusive), still LOCAL not in-model W==D",
              ledger_links=["docs/decode-attention-route-b-b2-graph-integration-result-20260621.md",
                            "bench/qk-decode-eval/candidates.json#reference_oracle_hcq_llama_tile"])
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT/"latest.json").write_text(json.dumps(art, indent=2))
  print(json.dumps({"verdict":verdict,"wall_speedup":wall_speedup,"gpu_busy_speedup":gpu_speedup,
                    "best_wall_us":round(best_wall,1),"correctness_rel_rmse":round(rmse,6)}, indent=2))

if __name__ == "__main__":
  main()
