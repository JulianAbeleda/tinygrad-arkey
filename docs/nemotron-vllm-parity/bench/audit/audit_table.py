"""Join the microbench timings and the ncu reports into the kernel-audit tables (markdown on stdout).
usage: audit_table.py DIR   (DIR holds vllm_gemm.json ours.json vncu_raw.csv oncu_raw.csv oncu.log)

Kernel-row parsing (ncu CSV -> counters) is shared with ncu_kernel_counters.py; this script only adds
the role/shape mapping and the audit-specific roofline/lever tables.
"""
import sys, json, re, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ncu_kernel_counters import read_ncu_csv_rows, parse_kernel_row
D = sys.argv[1].rstrip("/") + "/"
PEAK_TF, BW = 250e12, 1.69e12
MS_DEC, MS_PF = [8, 16, 32, 64, 128], [512, 1024, 2048, 4096, 8192]
ROLES = ["ssm_in", "ssm_out", "attn_q", "attn_kv", "attn_o", "ffn_up", "ffn_down", "output"]
V_ROLES = ["ssm_in", "ssm_out", "qkv", "attn_q", "attn_kv", "attn_o", "ffn_up", "ffn_down", "output"]
COUNT = {"ssm_in": 21, "ssm_out": 21, "attn_q": 4, "attn_kv": 8, "attn_o": 4, "ffn_up": 17, "ffn_down": 17, "output": 1}
NK = {"ssm_in": (17504, 3136), "ssm_out": (3136, 7680), "qkv": (7168, 3136), "attn_q": (5120, 3136),
      "attn_kv": (1024, 3136), "attn_o": (3136, 5120), "ffn_up": (12544, 3136), "ffn_down": (3136, 12544),
      "output": (131072, 3136)}

vj = {(r["role"], r["m"]): r for r in json.load(open(D + "vllm_gemm.json"))}
oj = {(r["role"], r["m"], r["mode"]): r for r in json.load(open(D + "ours.json"))}

def roof(role, m):
  n, k = NK[role]
  fl, by = 2 * m * n * k, 2 * (n * k + m * k + m * n)
  return fl, by, max(fl / PEAK_TF, by / BW) * 1e6, "mem" if by / BW > fl / PEAK_TF else "comp"

def v_us(role, m): return vj[(role, m)]["gpu_us"]
def o_us(role, m, mode="prod"):
  """Production time: prefill rows > 1024 run as pieces of 1024 (NemotronHPrefill piece=1024)."""
  if m > 1024: return m / 1024 * oj[(role, 1024, mode)]["gpu_us"]
  return oj[(role, m, mode)]["gpu_us"]

# Thin wrapper: parse via the shared module, then adapt to this file's legacy field names/units
# (smem_kb combines static+dynamic; the tables below were written against these names).
def ncu_rows(path):
  return read_ncu_csv_rows(path)

def summarize(k):
  row = parse_kernel_row(k, side="tinygrad")  # side is unused by this file; role/shape are assigned below
  # Legacy smem_kb is the ncu-computed per-block total (static + driver-reserved), not static+dynamic;
  # read it straight off the raw row to keep this file's existing table numbers unchanged.
  smem_kb = float(k.get("launch__shared_mem_per_block") or 0)
  return {"name": row["kernel"], "grid": row["grid"], "block": row["block"],
          "warps": int(k["launch__block_size"]) // 32, "regs": row["registers_per_thread"], "smem_kb": smem_kb,
          "us": row["duration_us"], "occ": row["achieved_occupancy_pct"], "tensor": row["tensor_pipe_util_pct"],
          "dram": row["dram_throughput_bytes_per_sec"], "sm": row["sm_throughput_pct"],
          "stalls": ", ".join(f"{s['reason']} {s['pct']:.0f}%" for s in row["top_stall_reasons"])}

# vLLM ncu rows in (role, M) order; the number of kernels per shape comes from the timing run
vrows, vncu = ncu_rows(D + "vncu_raw.csv"), {}
i = 0
for role in V_ROLES:
  for m in MS_DEC + MS_PF:
    nk = len(vj[(role, m)]["kernels"])
    ks = vrows[i:i + nk]; i += nk
    vncu[(role, m)] = [summarize(k) for k in ks]
assert i == len(vrows), (i, len(vrows))
orows, oncu = ncu_rows(D + "oncu_raw.csv"), {}
i = 0
for line in open(D + "oncu.log"):
  if line.startswith("SHAPE "):
    _, role, m, nk = line.split(); nk = int(nk)
    oncu[(role, int(m))] = [summarize(k) for k in orows[i:i + nk]]; i += nk
assert i == len(orows), (i, len(orows))

def vtile(name):
  t = re.search(r"_(\d+)x(\d+)_(\d+)x(\d+)_tn", name)
  if t: return f"{t[1]}x{t[2]}x{t[3]}", int(t[4])
  return "-", "-"

def gemm_of(ks):
  """The GEMM kernel of a shape's launches: the longest one that issues tensor-pipe work (else the longest)."""
  cand = [k for k in ks if "splitKreduce" not in k["name"] and k["tensor"] > 0.5]
  return max(cand or ks, key=lambda k: k["us"])

def fmt_tf(role, m, us): return 2 * m * NK[role][0] * NK[role][1] / us / 1e6

# our route table: (role, hi/lo rows) -> split-K and the candidate's schedule
RT = json.load(open(D + "../../../../tinygrad/llm/generated/dense_bf16_sm120_candidate_set.json")) if len(sys.argv) < 3 else json.load(open(sys.argv[2]))
CID = {e["canonical_identity"]: s["template"]["schedule"] for s in RT["sets"] for e in s["entries"]}
ROUTE = {(r["role"], r["m"]): (r["split_k"], CID[r["canonical_identity"]]) for r in RT["routes"]}
def our_route(role, m):
  if m <= 16: return "fp32 matvec (no TC), `nn.Linear` path"
  rows = min(2 * m, 512)
  sk, sc = ROUTE[(role, rows)]
  t = sc["tile"]
  chunks = f", {2 * m // 512} chunks" if 2 * m > 512 else ""
  return (f"hi/lo {2*m} rows -> route m={rows}{chunks}: tile {t['m']}x{t['n']}x{t['k']} (Mtok x Nout x K), "
          f"{sc['threads']//32} warps, {sc['pipeline']['buffer_count']} LDS bufs, splitK {sk}")

out = []
P = out.append
P("### A. Time per call (useful FLOPs = 2MNK; roofline = max(2MNK/250 TF, bf16 bytes/1.69 TB/s); * = derived from the M=1024 piece)\n")
for role in ROLES:
  P(f"\n#### {role} (N={NK[role][0]}, K={NK[role][1]}; {COUNT[role]}x per step)\n")
  P("| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |")
  P("|---|---|---|---|---|---|---|---|---|---|---|---|")
  for m in MS_DEC + MS_PF:
    if role == "output" and m > 128: continue
    fl, by, rt, bound = roof(role, m)
    ou, vu = o_us(role, m), v_us(role, m)
    if m > 16:
      g = oj[(role, min(m, 1024), "prod")]["gemm_us"] * (m / 1024 if m > 1024 else 1)
      prod_tf = f"{2 * 2 * m * NK[role][0] * NK[role][1] / g / 1e6:.0f}"
    else: prod_tf = "-"
    rt2 = max(2 * fl / PEAK_TF, by / BW) * 1e6 if m > 16 else rt  # the 2M-row hi/lo product's own roofline
    P(f"| {m}{'*' if m > 1024 else ''} | {bound} | {ou:.1f} | {fmt_tf(role, m, ou):.1f} ({prod_tf}) | {by/ou/1e3:.0f} | {rt/ou*100:.0f}% | {rt2/ou*100:.0f}% | "
      f"{vu:.1f} | {fmt_tf(role, m, vu):.1f} | {by/vu/1e3:.0f} | {rt/vu*100:.0f}% | {ou/vu:.2f}x |")

P("\n### B. Kernel structure and ncu (GEMM kernel only; vLLM tile from the CUTLASS name = Nout x Mtok x Ktile, cuBLAS column-major)\n")
for role in ROLES:
  P(f"\n#### {role}\n")
  P("| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |")
  P("|---|---|---|---|---|---|---|")
  for m in MS_DEC + MS_PF:
    if role == "output" and m > 128: continue
    vk = gemm_of(vncu[(role, m)]); tile, stages = vtile(vk["name"])
    sk = [k for k in vncu[(role, m)] if "splitKreduce" in k["name"]]
    gz = vk["grid"].strip("()").split(",")[-1].strip()
    kind = "wmma" if "wmma" in vk["name"] else "mma.sync"
    vdesc = f"`{kind} {tile}`, {stages} stages, splitK {gz}{' +splitKreduce' if sk else ''}"
    vg = f"{vk['grid']} / {vk['warps']} / {vk['regs']} / {vk['smem_kb']:.0f}KB"
    vn = f"{vk['occ']:.0f}%, {vk['tensor']:.0f}%, {vk['dram']:.2f}, {vk['stalls']}"
    if m <= 1024 and (ok := oncu.get((role, m))):
      g = gemm_of(ok)
      og = f"{g['grid']} / {g['warps']} / {g['regs']} / {g['smem_kb']:.0f}KB"
      on = f"{g['occ']:.0f}%, {g['tensor']:.0f}%, {g['dram']:.2f}, {g['stalls']}"
    else: og = on = "(as M=1024)"
    P(f"| {m} | {vdesc} | {vg} | {vn} | {our_route(role, min(m, 1024)) if m <= 1024 else 'pieces of 1024'} | {og} | {on} |")

# plain-bf16 (no hi/lo) routed variant
P("\n### Plain-bf16 routed variant (route_dense_bf16, no hi/lo; not production) vs production hi/lo vs vLLM, us\n")
P("| role | " + " | ".join(f"M={m} bf16 / hilo / vLLM" for m in [32, 64, 128, 512, 1024]) + " |")
P("|---|" + "---|" * 5)
for role in ROLES:
  P(f"| {role} | " + " | ".join(f"{oj[(role, m, 'bf16')]['gpu_us']:.0f} / {oj[(role, m, 'prod')]['gpu_us']:.0f} / {v_us(role, m):.0f}" if (role, m, 'bf16') in oj else "-" for m in [32, 64, 128, 512, 1024]) + " |")

# step weighting
def step(ms_ours, ms_v, mult=1.0):
  rows = []
  for role in ROLES:
    o = COUNT[role] * ms_ours(role) * mult
    if role == "attn_kv": continue
    if role == "attn_q":  # ours: q + k + v separately; vLLM: one fused qkv GEMM
      o = (COUNT["attn_q"] * ms_ours("attn_q") + COUNT["attn_kv"] * ms_ours("attn_kv")) * mult
      v = COUNT["attn_q"] * ms_v("qkv") * mult
      role = "attn_qkv"
    else: v = COUNT[role] * ms_v(role) * mult
    rows.append((role, o, v))
  return rows

P("\n### Lever ranking\n")
for title, fo, fv, denom in [
    ("RL decode step, B=32 (per step, ms; ours step 13.3 ms RolloutSampler)", lambda r: o_us(r, 32), lambda r: v_us(r, 32), 13.3),
    ("RL decode step, B=64 (per step, ms; ours step 21.6 ms)", lambda r: o_us(r, 64), lambda r: v_us(r, 64), 21.6),
    ("RL decode step, B=128 (per step, ms; ours step 35.2 ms)", lambda r: o_us(r, 128), lambda r: v_us(r, 128), 35.2),
    ("RL decode step, B=8 (per step, ms; ours step 8.8 ms)", lambda r: o_us(r, 8), lambda r: v_us(r, 8), 8.8),
    ("10k prefill (ms; ours 1650 ms; ours = 10000/1024 pieces of 1024, vLLM = 10000/8192 x the M=8192 kernel; lm_head excluded)",
     lambda r: 0 if r == "output" else o_us(r, 1024) * 10000 / 1024, lambda r: 0 if r == "output" else v_us(r, 8192) * 10000 / 8192, 1650)]:
  rows = step(lambda r: fo(r) / 1e3, lambda r: fv(r) / 1e3)
  to, tv = sum(o for _, o, _ in rows), sum(v for _, _, v in rows)
  P(f"\n{title}: GEMM total ours {to:.2f} ms ({to/denom*100:.0f}% of step) vs vLLM {tv:.2f} ms; gap {to-tv:.2f} ms\n")
  P("| role | ours ms | vLLM ms | gap ms | gap % of our step |")
  P("|---|---|---|---|---|")
  for role, o, v in sorted(rows, key=lambda x: -(x[1] - x[2])):
    P(f"| {role}{' (ours q+k+v; vLLM fused qkv)' if role == 'attn_qkv' else ''} | {o:.3f} | {v:.3f} | {o-v:.3f} | {(o-v)/denom*100:.1f}% |")
print("\n".join(out))
