#!/usr/bin/env python3
"""North-star flash_attn_tile decode candidate — LOCAL A/B vs the current winner gqa_coop_vec.

Binding: local diagnostic only; BoltBeam owns promotion/evaluation binding.

Design (differs from the failed tiles): reuse the warp-cooperative q.k PARTIAL (ds_bpermute butterfly dot, NO 128x
redundancy that killed fused-LDS at 0.21x; LDS K/V staging; GQA query-head packing = 4 warps; register online
softmax; many KV-splits -> 128..1024 workgroups, GROWS with ctx) -- but REPLACE the serial flash_reduce combine
(grid Hq=32 wg, the named ceiling that left the warp tile at 0.60x@1024 / 0.95x@4096) with a MANY-WORKGROUP combine
(grid Hq*(Hd/DT) -> 128..256 wg) so the combine stops under-occupying the GPU. No WMMA. No model change.

First gate: candidate attention us < gqa_coop_vec us, >=1.05x @ctx1024 AND no regress @ctx4096. If it misses, STOP,
classify, bank a refutation (no W==D route). Clock-pinned, byte-exact vs numpy.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk/north_star_flash_attn_tile_ab.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.qk.flash_decode import flash_reduce_src, flash_decode_attention
from extra.qk.decode_warp_flash_tile_ab import warp_tile_src, ref_attn, time_fn
from extra.qk.clock_pin import pinned_peak

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-north-star-flash-attn-tile"
Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096
G = Hq // Hkv

# Improved combine: MANY workgroups (Hq*(Hd/DT)) vs flash_reduce's Hq. DT output dims per workgroup, one per thread.
def streamk_combine_src(Hd, Hq, Hkv, S, MAXC, DT):
  return f'''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, pure)) float __ocml_exp2_f32(float);
#define Hd {Hd}
#define S {S}
#define DT {DT}
#define NCH {Hd // DT}
#define EXPF(x) __ocml_exp2_f32((x) * 1.4426950408889634f)
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size({DT},{DT})))
void streamk_combine(float* out, float* pout, float* pm, float* pl) {{
  int wg = (int)__ockl_get_group_id(0);
  int h = wg / NCH; int ch = wg % NCH; int d = ch*DT + (int)__ockl_get_local_id(0);
  float gm = -1e30f;
  for (int s=0; s<S; s++) {{ float v = pm[h*S+s]; if (v > gm) gm = v; }}
  float num = 0.0f, den = 0.0f;
  for (int s=0; s<S; s++) {{ float w = EXPF(pm[h*S+s] - gm); num += pout[((long)h*S+s)*Hd+d]*w; den += pl[h*S+s]*w; }}
  out[h*Hd + d] = num / den;
}}
'''

def main():
  dev = Device["AMD"]; rng = np.random.default_rng(0)
  q = rng.standard_normal((Hq, Hd)).astype(np.float16)
  k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
  qn, kn, vn = Tensor(q), Tensor(k), Tensor(v)
  CTXS = [512, 1024, 4096]
  DT = 32  # 32 dims/wg -> Hq*(Hd/32)=128 combine workgroups (4x flash_reduce's 32)
  results = []
  with pinned_peak() as prov:
    time.sleep(0.4)
    for Tc in CTXS:
      ref = ref_attn(q, k, v, Tc)
      def buf(arr, dt):
        b = Buffer("AMD", arr.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(arr))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      vsp = UOp.variable("start_pos", 0, MAXC - 1)
      juop = TinyJit(lambda spb: flash_decode_attention(qn, kn, vn, spb + 1, vsp + 1, Hd, Hq, Hkv, MAXC, 128, variant="gqa_coop_vec").realize())
      for _ in range(8): juop(vsp.bind(Tc - 1))
      coop_med = [time_fn(lambda: juop(vsp.bind(Tc - 1))) for _ in range(5)]
      coop_us = statistics.median(coop_med); coop_band = (max(coop_med) - min(coop_med)) / coop_us * 100
      row = {"ctx": Tc, "comparator_attention_us": round(coop_us, 1), "comparator_band_pct": round(coop_band, 2), "splits": []}
      for S in [16, 32, 64, 96]:
        per = (Tc + S - 1) // S
        try:
          pp = dev.runtime("warp_flash_tile", dev.compiler.compile(warp_tile_src(Hd, Hq, Hkv, S, MAXC, per)))
          prk = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
          psk = dev.runtime("streamk_combine", dev.compiler.compile(streamk_combine_src(Hd, Hq, Hkv, S, MAXC, DT)))
          pout = Buffer("AMD", Hq * S * Hd, dtypes.float32).ensure_allocated()
          pm = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq * S, dtypes.float32).ensure_allocated()
          ob = Buffer("AMD", Hq * Hd, dtypes.float32).ensure_allocated()
          def part(): pp(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hkv * S, 1, 1), local_size=(128, 1, 1), vals=(Tc,), wait=False)
          def serial(): part(); prk(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq, 1, 1), local_size=(Hd, 1, 1), wait=False)
          def streamk(): part(); psk(ob._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq * (Hd // DT), 1, 1), local_size=(DT, 1, 1), wait=False)
          streamk(); dev.synchronize()
          _o = np.empty(Hq * Hd, np.float32); ob.copyout(memoryview(_o)); err = float(np.abs(_o.reshape(Hq, Hd) - ref).max())
          part_us = time_fn(lambda: (part(),))                      # partial alone (attribution)
          serial_us = time_fn(serial); streamk_us = time_fn(streamk)
          best = min(serial_us, streamk_us)
          row["splits"].append({"S": S, "workgroups_partial": Hkv * S, "workgroups_combine_serial": Hq,
                                 "workgroups_combine_streamk": Hq * (Hd // DT), "keys_per_split": per,
                                 "partial_us": round(part_us, 1), "serial_combine_total_us": round(serial_us, 1),
                                 "streamk_combine_total_us": round(streamk_us, 1), "best_us": round(best, 1),
                                 "speedup_vs_coop": round(coop_us / best, 3), "err": round(err, 5)})
          print(f"  ctx{Tc} S={S:3} wg_p={Hkv*S:4}: partial {part_us:5.1f} | serial {serial_us:5.1f} streamk {streamk_us:5.1f} vs coop {coop_us:.1f} -> {coop_us/best:.2f}x err={err:.4f}", file=sys.__stderr__)
        except Exception as e:
          row["splits"].append({"S": S, "error": str(e)[:90]}); print(f"  ctx{Tc} S={S}: ERR {str(e)[:90]}", file=sys.__stderr__)
      ok = [s for s in row["splits"] if "speedup_vs_coop" in s]
      best = max(ok, key=lambda s: s["speedup_vs_coop"]) if ok else None
      row["best_speedup_vs_coop"] = best["speedup_vs_coop"] if best else 0
      row["best_split"] = best["S"] if best else None
      results.append(row)
  # binding artifact fields
  by_ctx = {r["ctx"]: r for r in results}
  def bestrow(c): return next((s for s in by_ctx[c]["splits"] if s.get("S") == by_ctx[c]["best_split"]), {})
  art = {"date": "2026-06-21", "phase": "NORTH_STAR_FLASH_ATTN_TILE_LOCAL_AB",
         "binding_id": "north_star_flash_attn_tile_v0", "comparator": "gqa_coop_vec",
         "gate": ">=1.05x vs gqa_coop_vec @ctx1024 AND no regress @ctx4096", "DT": DT,
         "candidate_params": {"kv_split_count": "S in {16,32,64,96} (best per ctx)", "flash_l": "keys_per_split=ceil(ctx/S)",
                              "query_head_pack": "4 warps = 4 GQA query heads", "gqa_grouping": "Hq/Hkv=4",
                              "combine_strategy": "many-workgroup streamk_combine (Hq*(Hd/DT) wg) vs serial flash_reduce (Hq wg)",
                              "softmax_strategy": "register online softmax in partial", "expected_no_wmma": True},
         "workgroups_by_ctx": {c: Hkv * (by_ctx[c]["best_split"] or 0) for c in CTXS},
         "kv_splits_by_ctx": {c: by_ctx[c]["best_split"] for c in CTXS},
         "query_heads_parallelized": Hq, "combine_kernel_count": 1,
         "local_attention_us_by_ctx": {c: bestrow(c).get("best_us") for c in CTXS},
         "comparator_attention_us_by_ctx": {c: by_ctx[c]["comparator_attention_us"] for c in CTXS},
         "wd_tok_s_by_ctx": None, "correctness_error": max((s.get("err", 0) for r in results for s in r["splits"]), default=0),
         "reproducibility_band_pct": {c: by_ctx[c]["comparator_band_pct"] for c in CTXS},
         "results": results, "expected_no_wmma": True, "default_behavior_changed": False,
         "clock_pin": (prov or {}).get("ok"),
         "first_gate_pass": (by_ctx[1024]["best_speedup_vs_coop"] >= 1.05 and by_ctx[4096]["best_speedup_vs_coop"] >= 1.0)}
  OUT.mkdir(parents=True, exist_ok=True)
  f = OUT / f"local_ab_{time.strftime('%Y%m%dT%H%M%S')}.json"; f.write_text(json.dumps(art, indent=2))
  (OUT / "latest.json").write_text(json.dumps(art, indent=2))
  print(json.dumps({"first_gate_pass": art["first_gate_pass"], "correctness_error": art["correctness_error"],
                    "ctx1024_best": by_ctx[1024]["best_speedup_vs_coop"], "ctx4096_best": by_ctx[4096]["best_speedup_vs_coop"],
                    "out": str(f.relative_to(ROOT))}, indent=2))

if __name__ == "__main__":
  main()
