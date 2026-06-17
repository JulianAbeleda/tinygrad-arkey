#!/usr/bin/env python3
"""Arc 3 Phase 0/1: per-role GEMV efficiency, in-model vs standalone. For each QK GEMV role, effective weight-read
bandwidth = (role weight bytes / summed kernel tm) from a warm eager decode step (DEBUG=2). GEMVs are HBM-bound,
so eager per-kernel tm reflects real bandwidth (unlike the small-op tail). Compare %HBM-peak across roles to find
weak roles (in-model << the ~76% the standalone Q4_K GEMV hits). No code change -- audit only.

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_gemv_role_efficiency.py [model.gguf]
"""
from __future__ import annotations

import io, json, os, pathlib, re, contextlib, sys
from collections import defaultdict

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us.*?(\d+)\|(\d+)\s+GB/s")
_GEMV = re.compile(r"q([46])k_gemv\w*_(\d+)_(\d+)_")
HBM_PEAK_GBS = 960.0   # RX 7900 XTX spec; %peak is relative -- the cross-role + vs-standalone comparison is the signal
# (role, total weight bytes/token, qbits) -- from the byte census; q/o and k/v are lumped (same GEMV shape)
ROLE_BYTES = {"ffn_down": (1252.8e6, 6), "ffn_gate/up": (2 * 1019.2e6, 4), "lm_head": (510.5e6, 6),
              "attn_q/o": (2 * 339.7e6, 4), "attn_k/v": (84.9e6 + 104.4e6, 4)}
SHAPE_ROLE = {(151936, 4096): "lm_head", (4096, 12288): "ffn_down", (12288, 4096): "ffn_gate/up",
              (4096, 4096): "attn_q/o", (1024, 4096): "attn_k/v"}

def main():
  model = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  from tinygrad import Tensor, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox. " * 40)
  sp, tokid = 64, int(ids[64])
  with Context(DEBUG=0): m.logits(Tensor([ids[:sp]], dtype="int32").contiguous(), 0).realize()
  with Context(DEBUG=0):
    for _ in range(5): m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()

  roles = defaultdict(lambda: [0, 0.0, 0.0])   # role -> [kernels, total_tm_us, sum_read_gbs]
  for l in buf.getvalue().splitlines():
    mt = _LINE.search(_ANSI.sub("", l))
    if not mt: continue
    g = _GEMV.search(mt.group(1).lower())
    if not g: continue
    role = SHAPE_ROLE.get((int(g.group(2)), int(g.group(3))))
    if role is None: continue
    roles[role][0] += 1; roles[role][1] += float(mt.group(2)); roles[role][2] += max(int(mt.group(3)), int(mt.group(4)))

  rows = []
  for role, (cnt, tm_us, read_gbs) in sorted(roles.items(), key=lambda kv: -kv[1][1]):
    by, qb = ROLE_BYTES.get(role, (0, 4))
    eff_bw = by / (tm_us * 1e-6) / 1e9 if tm_us else 0   # GB/s = bytes / seconds
    rows.append({"role": role, "kernels": cnt, "total_tm_ms": round(tm_us / 1000, 3), "weight_MB": round(by / 1e6, 1),
                 "qbits": qb, "effective_read_GBs": round(eff_bw, 1), "pct_hbm_peak": round(100 * eff_bw / HBM_PEAK_GBS, 1),
                 "debug_max_GBs": round(read_gbs / cnt, 1) if cnt else 0})
  weak = [r["role"] for r in rows if r["pct_hbm_peak"] < 60]
  out = {"model_id": pathlib.Path(model).stem, "hbm_peak_GBs": HBM_PEAK_GBS,
         "standalone_ref": "Q4_K int-dot GEMV ~76% HBM (banked)", "rows": rows,
         "weak_roles_below_60pct": weak,
         "note": "effective_read_GBs = role weight bytes / summed warm-eager kernel tm; GEMVs are HBM-bound so this "
                 "reflects real bandwidth. Compare %hbm_peak across roles + vs standalone ~76% to find weak roles."}
  print(f"{'role':12} {'ker':>4} {'tm_ms':>7} {'wMB':>7} {'eff_GB/s':>9} {'%peak':>6}")
  for r in rows:
    print(f"{r['role']:12} {r['kernels']:4} {r['total_tm_ms']:7.2f} {r['weight_MB']:7.1f} {r['effective_read_GBs']:9.0f} {r['pct_hbm_peak']:5.1f}%")
  print(f"weak roles (<60% peak): {weak or 'none'}")
  art = pathlib.Path("bench/qk-gemv-role-efficiency/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
