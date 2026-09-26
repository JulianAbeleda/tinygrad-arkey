"""Per-kernel breakdown of the non-GEMM categories, reusing BoltBeam's lifecycle_compare attribution (no new logic
for segmentation/labeling): for each workload, kernel name -> (category, launches/step, ms/step, us/launch, grid)."""
import sys, json, collections, pathlib
sys.path.insert(0, "/home/ubuntu/BoltBeam")
from boltbeam.trace import lifecycle_compare as lc
M = json.load(open("/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/scan-20260926/lc_manifest.json"))
rows = json.load(open(M["ours_gemm_json"])); rows = rows["rows"] if isinstance(rows, dict) else rows
pattern = M["pattern"]
out = {}
for w in M["workloads"]:
  if len(sys.argv) > 1 and w["id"] not in sys.argv[1:]: continue
  kind = w["kind"]
  # ours
  meta = json.load(open(w["ours"]["meta"]))
  if kind == "decode": graphs = lc.load_hcq_profile_lines(w["ours"]["jsonl"], tuple(meta["lines"]))
  else: graphs = lc.load_hcq_profile_lines(w["ours"]["jsonl"], (meta["marks"][-2], meta["marks"][-1]))
  gemm_names, pre, post = lc.ours_gemm_names(rows); is_gemm = lc._ours_is_gemm_factory(gemm_names)
  if kind == "decode":
    step_lists = list(meta.get("step", {}).values())
    anchor = lc.strip_tinygrad_name(step_lists[-1][0]["name"])
    flush = lc.strip_tinygrad_name(meta["flush"][0]["name"]) if meta.get("flush") else None
    seg = lc.ours_decode_steps(graphs, anchor_name=anchor, flush_name=flush); steps = seg["steps"]
    expected = lc.layer_roles(pattern, "q_k_v"); merge, pol = False, "auto"
  else:
    steps = [[dict(k, flush=False) for g in graphs for k in sorted(g, key=lambda k: k["s"])]]
    expected = lc.layer_roles(pattern, "q_kv"); merge, pol = True, "last_only"
  agg = collections.defaultdict(lambda: [0, 0.0])
  for st in steps:
    body = [k for k in st if not k.get("flush")]
    lab, info = lc.attribute_step(body, expected, is_gemm=is_gemm, pre_helpers=pre, post_helpers=post, merge_same_name_runs=merge, output_policy=pol)
    it = iter(lab)
    for k in st:
      l = "state_flush" if k.get("flush") else next(it)
      a = agg[(l, k["name"])]; a[0] += 1; a[1] += k["e"] - k["s"]
  ns = len(steps)
  ours = [{"cat": c, "name": n, "per_step": v[0]/ns, "ms": v[1]/ns/1e3, "us_each": v[1]/v[0]} for (c, n), v in agg.items()]
  # vllm
  K, g = lc.load_nsys_sqlite(w["vllm"]["sqlite"])
  if kind == "decode": vs = lc.vllm_decode_steps(K, g)["steps"]
  else: vs = [K]
  vexp = lc.layer_roles(pattern, "fused_qkv")
  vagg = collections.defaultdict(lambda: [0, 0.0, None])
  for st in vs:
    lab, info = lc.attribute_step(st, vexp, is_gemm=lc._vllm_is_gemm, post_helpers=lc.VLLM_POST_HELPERS)
    for k, l in zip(st, lab):
      a = vagg[(l, k["name"])]; a[0] += 1; a[1] += k["e"] - k["s"]; a[2] = k.get("grid")
  nv = len(vs)
  vllm = [{"cat": c, "name": n, "per_step": v[0]/nv, "ms": v[1]/nv/1e6, "us_each": v[1]/v[0]/1e3, "grid": v[2]} for (c, n), v in vagg.items()]
  out[w["id"]] = {"ours": ours, "vllm": vllm, "ours_steps": ns, "vllm_steps": nv}
  print(w["id"], ns, nv, file=sys.stderr)
json.dump(out, open("/home/ubuntu/storage/audit-nongemm/per_kernel.json" if len(sys.argv) == 1 else f"/home/ubuntu/storage/audit-nongemm/per_kernel_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
