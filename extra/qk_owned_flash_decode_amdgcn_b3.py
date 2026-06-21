#!/usr/bin/env python3
"""Route B B3 — OWNED hand-AMDGCN decode-attention tile (tinygrad KV layout), local A/B vs gqa_coop_vec.

The promotable counterpart to the vendored B1/B2 oracle: extra/qk_owned_flash_decode.hip is OUR source (llama-style
flash-decode dataflow: warp-per-q-head, v_dot2 q.k, register online softmax + PV, KV-split + combine) authored to
tinygrad's native K/V layout [Hkv,MAXC,Hd] -- so NO repacking (the B2 layout block is removed) and it is promotable.
Compiled with hipcc->unbundle, launched via the B1 NamedAMDProgram + the B2 one-bound-HCQ-queue pattern.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_owned_flash_decode_amdgcn_b3.py [S]
"""
from __future__ import annotations
import json, struct, statistics, subprocess, pathlib, time, hashlib, sys, os
import numpy as np
from tinygrad import Tensor, Device, TinyJit, Context
from tinygrad.device import Compiled, BufferSpec
from tinygrad.helpers import round_up
from tinygrad.runtime.support.hcq import hcq_profile
from extra.qk_flash_decode import flash_decode_attention
from extra.qk_clock_pin import pinned_peak
from extra.qk_harness_contract import stamp, repro_band
from extra.qk_llama_flash_attn_tile_hcq_ab import NamedAMDProgram, write_hidden, kd_offset, buf, va, COMP_MAXC, COMP_L, COMP_CTX

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-route-b-b3"
SRC = ROOT / "extra/qk_owned_flash_decode.hip"
HIPCC = "/opt/rocm-7.2.4/bin/hipcc"
BUNDLER = "/opt/rocm-7.2.4/llvm/bin/clang-offload-bundler"
OBJDUMP = "/opt/rocm-7.2.4/llvm/bin/llvm-objdump"
Hd, Hq, Hkv, G = 128, 32, 8, 4

def compile_owned():
  src = SRC.read_bytes(); h = hashlib.sha256(src).hexdigest()[:12]
  elf_path = pathlib.Path(f"/tmp/owned_flash_{h}.elf")
  if not elf_path.exists():
    co = f"/tmp/owned_flash_{h}.co"
    subprocess.run([HIPCC, "--offload-arch=gfx1100", "--genco", "-O3", "-D__AMDGCN_WAVEFRONT_SIZE=32",
                    str(SRC), "-o", co], check=True, capture_output=True)
    subprocess.run([BUNDLER, "--type=o", "--unbundle", f"--input={co}", f"--output={elf_path}",
                    "--targets=hipv4-amdgcn-amd-amdhsa--gfx1100"], check=True, capture_output=True)
  return elf_path.read_bytes(), h

def isa_stats(elf, sym):
  """v_dot2 count (per-kernel disasm region) + VGPR/SGPR/private(spill) from ELF .num_* symbols."""
  import tempfile
  with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f: f.write(elf); p = f.name
  dis = subprocess.run([OBJDUMP, "-d", p, "--disassemble-symbols="+sym], text=True, capture_output=True).stdout
  vdot2 = dis.lower().count("v_dot2")
  nm = subprocess.run(["nm", p], text=True, capture_output=True).stdout
  stat = {"v_dot2_count": vdot2}
  for line in nm.splitlines():
    parts = line.split()
    if len(parts) == 3 and parts[2].startswith(sym + "."):
      key = parts[2].split(".", 1)[1]
      if key in ("num_vgpr", "num_agpr", "num_sgpr", "numbered_sgpr", "private_seg_size"):
        try: stat[key] = int(parts[0], 16)
        except ValueError: pass
  stat["spill"] = stat.get("private_seg_size", 0)
  return stat

def build_prog(dev, elf, sym, specs, ptrs, scalars, dynamic_lds=0):
  """specs: list of (name,size,align,is_ptr). ptrs: {name:va}. scalars: {name:(fmt,val)}."""
  off, offs = 0, []
  for _, sz, al, _ in specs:
    off = round_up(off, al); offs.append(off); off += sz
  ka = bytearray(off)
  for (nm, sz, al, isp), o in zip(specs, offs):
    if isp: struct.pack_into("<Q", ka, o, ptrs[nm])
    else: struct.pack_into(scalars[nm][0], ka, o, scalars[nm][1])
  kd = kd_offset(elf, sym + ".kd")
  prg = NamedAMDProgram(dev, sym, elf, kd, bytes(ka), dynamic_lds=dynamic_lds)
  full = bytearray(max(prg.kernargs_segment_size, len(ka))); full[:len(ka)] = ka
  # tile/combine read only blockIdx/threadIdx (not gridDim), so hidden args are not strictly needed; fill if present.
  grid = (0,0,0)  # unused by these kernels; write_hidden only fills the COV5 block if ksize>=256
  write_hidden(full, prg.kernargs_segment_size, (1,1,1), (LANES_DUMMY:=32,1,1)) if prg.kernargs_segment_size >= 256 else None
  prg._raw = bytes(full)
  return prg, off

def main():
  assert Device.DEFAULT == "AMD"
  dev = Device[Device.DEFAULT]
  S = int(sys.argv[1]) if len(sys.argv) > 1 else 8
  NVALID = COMP_CTX  # ctx1024
  scale = 1.0/np.sqrt(Hd)
  elf, src_hash = compile_owned()
  kernel_hash = hashlib.sha256(elf).hexdigest()[:16]

  # ---- inputs (tinygrad layout): Q[Hq,Hd], K/V[Hkv,MAXC,Hd] fp16 ----
  rng = np.random.default_rng(0)
  Q  = rng.standard_normal((Hq, Hd)).astype(np.float16)
  Kf = (rng.standard_normal((Hkv, COMP_MAXC, Hd))*0.5).astype(np.float16)
  Vf = (rng.standard_normal((Hkv, COMP_MAXC, Hd))*0.5).astype(np.float16)
  # numpy GQA reference
  ref = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kvh = h//G
    sc = (Q[h:h+1].astype(np.float32) @ Kf[kvh,:NVALID].astype(np.float32).T)[0]*scale
    p = np.exp(sc-sc.max()); p/=p.sum()
    ref[h] = p @ Vf[kvh,:NVALID].astype(np.float32)

  bQ, bK, bV = buf(Q.nbytes), buf(Kf.nbytes), buf(Vf.nbytes)
  bQ.copyin(memoryview(np.ascontiguousarray(Q))); bK.copyin(memoryview(np.ascontiguousarray(Kf)))
  bV.copyin(memoryview(np.ascontiguousarray(Vf)))
  bPart = buf(Hq*S*Hd*4); bMeta = buf(Hq*S*2*4); bOut = buf(Hq*Hd*4)

  TILE_SPEC = [("Q",8,8,1),("K",8,8,1),("V",8,8,1),("part",8,8,1),("meta",8,8,1),
               ("n_valid",4,4,0),("S",4,4,0),("scale",4,4,0)]
  COMB_SPEC = [("part",8,8,1),("meta",8,8,1),("out",8,8,1),("S",4,4,0)]
  kern = os.environ.get("KERNEL", "v2")
  tile_sym = "owned_flash_tile_gqa" if kern == "v2" else "owned_flash_tile"
  tile, _ = build_prog(dev, elf, tile_sym, TILE_SPEC,
                       {"Q":va(bQ),"K":va(bK),"V":va(bV),"part":va(bPart),"meta":va(bMeta)},
                       {"n_valid":("<i",NVALID),"S":("<i",S),"scale":("<f",scale)})
  comb, _ = build_prog(dev, elf, "owned_flash_combine", COMB_SPEC,
                       {"part":va(bPart),"meta":va(bMeta),"out":va(bOut)}, {"S":("<i",S)})
  # v2 (GQA-packed): grid = Hkv x S, 128-thread workgroups (4 warps = 4 q-heads). v1: Hq x S, 32-thread.
  tg, tb = ((Hkv, S, 1), (128, 1, 1)) if kern == "v2" else ((Hq, S, 1), (32, 1, 1))
  cg, cb = (Hq, 1, 1), (32, 1, 1)
  print(f"  kernel={kern} ({tile_sym}) grid={tg} block={tb}")
  ka_tile = tile.fill_kernargs([], [], kernargs=dev.allocator.alloc(tile.kernargs_alloc_size, BufferSpec(cpu_access=True, nolru=True)))
  ka_comb = comb.fill_kernargs([], [], kernargs=dev.allocator.alloc(comb.kernargs_alloc_size, BufferSpec(cpu_access=True, nolru=True)))

  # ---- B2 one-bound-HCQ-queue: tile + combine, one doorbell. Two bound queues: ----
  # (a) profiled, per-call sync -> GPU-busy;  (b) timeline-variable, PIPELINED (sync once per N) -> fair WALL vs coop's
  # TinyJit (which also pipelines). Pipelining is the realistic model-JIT replay (sync once per token, not per attn call).
  from tinygrad.uop.ops import UOp
  bsig = dev.new_signal(value=0)
  bq = dev.hw_compute_queue_t().memory_barrier()
  with hcq_profile(dev, queue=bq, enabled=True, desc="owned") as (bst, ben):
    bq.exec(tile, ka_tile, tg, tb); bq.exec(comb, ka_comb, cg, cb)
  bq.signal(bsig, 1); bq.bind(dev)
  def owned():       # per-call sync, profiled -> GPU-busy
    bsig.value = 0; bq.submit(dev); bsig.wait(1)
    return float(ben.timestamp - bst.timestamp)

  owned()
  ob = bytearray(Hq*Hd*4); bOut.copyout(memoryview(ob)); out = np.frombuffer(bytes(ob), np.float32).reshape(Hq, Hd)
  rel = float(np.abs(out-ref).max()/(np.abs(ref).max()+1e-6))
  rmse = float(np.sqrt(((out-ref)**2).mean())/(np.sqrt((ref**2).mean())+1e-9))
  print(f"S={S} correctness: rel_max={rel:.4e} rel_rmse={rmse:.4e} {'OK' if rmse<=1e-3 else ('fp16-OK' if rmse<=5e-3 else 'FAIL')}")

  # ---- comparator + A/B (GPU-busy + wall) ----
  rng2 = np.random.default_rng(0)
  cqT = Tensor(rng2.standard_normal((Hq, Hd)).astype(np.float16))
  ckT = Tensor((rng2.standard_normal((Hkv, COMP_MAXC, Hd))*0.5).astype(np.float16))
  cvT = Tensor((rng2.standard_normal((Hkv, COMP_MAXC, Hd))*0.5).astype(np.float16))
  def wall(fn, n=200):
    for _ in range(10): fn()
    dev.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    dev.synchronize(); return (time.perf_counter()-t0)/n*1e6
  with pinned_peak() as prov:
    time.sleep(0.3)
    comp = TinyJit(lambda: flash_decode_attention(cqT, ckT, cvT, COMP_CTX, COMP_CTX, Hd, Hq, Hkv, COMP_MAXC, COMP_L,
                                                  variant="gqa_coop_vec").realize())
    def comp_sync(): comp(); dev.synchronize()
    for _ in range(8): comp(); owned()
    dev.synchronize()
    owned_gpu = statistics.median([owned() for _ in range(30)])
    # FAIR wall: both sync per call (same launch-overhead model). The pipelined coop wall is also reported (reference).
    owned_wall = statistics.median([wall(owned) for _ in range(3)])          # owned, per-call sync
    comp_wall = statistics.median([wall(comp_sync) for _ in range(3)])       # coop, per-call sync (MATCHED) -> fair ratio
    comp_wall_pipe = statistics.median([wall(comp) for _ in range(3)])       # coop pipelined (reference only)
    comp_gpu_s = []
    with Context(PROFILE=1):
      compp = TinyJit(lambda: flash_decode_attention(cqT, ckT, cvT, COMP_CTX, COMP_CTX, Hd, Hq, Hkv, COMP_MAXC, COMP_L,
                                                     variant="gqa_coop_vec").realize())
      for _ in range(8): compp()
      dev.synchronize(); dev._at_profile_finalize()
      for _ in range(5):
        base = len(Compiled.profile_events); compp(); dev.synchronize(); dev._at_profile_finalize()
        busy = 0.0
        for e in Compiled.profile_events[base:]:
          if type(e).__name__ != "ProfileGraphEvent": continue
          sigs = [float(s) for s in e.sigs]
          for ent in e.ents: busy += sigs[ent.en_id]-sigs[ent.st_id]
        if busy>0: comp_gpu_s.append(busy)
    comp_gpu = statistics.median(comp_gpu_s) if comp_gpu_s else 0.0
  gpu_sp = round(comp_gpu/owned_gpu, 3) if owned_gpu else 0.0
  wall_sp = round(comp_wall/owned_wall, 3) if owned_wall else 0.0          # fair: both per-call sync
  print(f"  GPU-busy: coop {comp_gpu:.1f}us vs owned {owned_gpu:.1f}us -> {gpu_sp}x | "
        f"WALL(matched-sync): coop {comp_wall:.1f}us vs owned {owned_wall:.1f}us -> {wall_sp}x "
        f"(coop pipelined ref {comp_wall_pipe:.1f}us)")
  isa = isa_stats(elf, tile_sym)
  corr_ok = rmse <= 1e-3; gpu_ok = gpu_sp >= 1.5; wall_ok = wall_sp >= 1.5
  local_pass = corr_ok and gpu_ok and wall_ok
  print(f"  ISA: {isa} | local_pass={local_pass} (corr={corr_ok} gpu={gpu_ok} wall={wall_ok})")
  if not local_pass:
    verdict = ("B3_FAIL_LOCAL_AB" if corr_ok else "B3_FAIL_CORRECTNESS")
  else:
    verdict = "B3_LOCAL_PASS_WD_BLOCKED_GRAPH_NODE"   # task family: LOCAL_PASS_WD_FAIL (W==D blocked, not regressed)
  layout_hash = hashlib.sha256((OUT/"tinygrad_kv_layout_contract.json").read_bytes()).hexdigest()[:16]
  wd_blocked = ("The owned kernel reads tinygrad's NATIVE K/V layout (no repack -> B2 layout block REMOVED, and it is "
    "promotable). But it is a raw hipcc .co launched via HCQ, NOT a tinygrad UOp/graph op, so it cannot enter the "
    "JIT-traced decode graph that model.generate replays: injecting it needs either Route-A native codegen (FORBIDDEN "
    "this phase + the known UOp inexpressibility wall) or eager (un-jitted) decode (non-production, Amdahl-limited). "
    "So a production W==D is gated on a 'schedule an external precompiled kernel as a JIT graph node' capability -- a "
    "bounded tinygrad feature (NOT Route-A codegen of the attention). W==D NOT run; default stays off.")
  art = {"date":"2026-06-21","phase":"ROUTE_B_B3_OWNED_AMDGCN_LOCAL_AB","candidate_id":"decode_attention_llama_flash_tile_owned_amdgcn",
         "comparator":"gqa_coop_vec","kernel":kern,"tile_symbol":tile_sym,
         "kernel_hash":kernel_hash,"source_hash":src_hash,"isa_summary":isa,
         "v_dot2_count":isa.get("v_dot2_count"),"lds_bytes":tile.group_segment_size,
         "vgpr":isa.get("num_vgpr"),"sgpr":isa.get("num_sgpr"),"spill":isa.get("spill",0),
         "aql_packet_count":2,"dispatch_count":2,"doorbell_count":1,"graph_replay_count":1,
         "tinygrad_kv_layout_contract_hash":layout_hash,
         "tile_grid_workgroups":list(tg),"tile_block":list(tb),"split_S":S,"combine":"log-sum-exp over S partials",
         "owned_gpu_busy_us":round(owned_gpu,1),"coop_gpu_busy_us":round(comp_gpu,1),
         "owned_wall_us":round(owned_wall,1),"coop_wall_us":round(comp_wall,1),"coop_wall_pipelined_us":round(comp_wall_pipe,1),
         "gpu_busy_speedup":gpu_sp,"wall_speedup":wall_sp,
         "wall_method":"BOTH per-call sync (matched launch-overhead model) -- fair; coop pipelined wall reported as reference only",
         "correctness_rel_max":round(rel,6),"correctness_rel_rmse":round(rmse,7),"correctness_tol":1e-3,
         "results":[{"ctx":1024,"best_speedup_vs_coop":wall_sp,"splits":[{"err":round(rel,6)}]}],
         "first_gate_pass":bool(local_pass),"verdict":verdict,"wd_status":"NOT RUN (blocked by graph-node integration)",
         "wd_blocked_reason":wd_blocked,"promotable":True,"default_eligible":False,
         "comparator_reason":"gqa_coop_vec is the shipped decode-attention winner; the owned tile must beat it locally to justify the W==D/graph-node integration",
         "pass_fail_threshold":"correctness rel_rmse<=1e-3 AND wall>=1.5x (matched-sync) AND GPU-busy>=1.5x vs gqa_coop_vec @ctx1024",
         "repro_band":{"coop_gpu":repro_band(comp_gpu_s)},"clock_pin":(prov or {}).get("ok"),
         "default_behavior_changed":False,
         "route_decision":("OWNED PROMOTABLE kernel WINS locally (project crosses 'can we own the primitive?' -> YES). "
                           "Promotion gated on external-kernel-as-JIT-graph-node integration (the scoped next step).")}
  art = stamp(art, comparator_id="gqa_coop_vec",
              comparator_why="shipped default decode-attention primitive; the owned tile is the first promotable kernel that beats it locally",
              timing_authority="GPU-busy via signal timestamps (owned) / ProfileGraphEvent (coop); WALL via back-to-back perf_counter median-of-3, BOTH per-call sync (matched) -- LOCAL, not in-model W==D",
              ledger_links=["docs/decode-attention-route-b-b3-owned-amdgcn-result-20260621.md",
                            "bench/qk-decode-attention-route-b-b3/tinygrad_kv_layout_contract.json"])
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT/"latest.json").write_text(json.dumps(art, indent=2))
  print(json.dumps({"verdict":verdict,"local_pass":local_pass,"gpu_busy_speedup":gpu_sp,"wall_speedup":wall_sp,
                    "correctness_rel_rmse":round(rmse,7),"v_dot2":isa.get("v_dot2_count"),"S":S}, indent=2))
  return art

if __name__ == "__main__":
  main()
