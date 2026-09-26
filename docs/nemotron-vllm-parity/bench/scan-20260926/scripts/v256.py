import sys, json
sys.path.insert(0, "/home/ubuntu/tinygrad-self-training/docs/nemotron-vllm-parity/bench/audit")
import vllm_gemm as v
out = []
for role in v.ROLES:
  for m in (256,):
    r = v.bench(role, m); out.append(r); print(role, m, round(r["gpu_us"], 2), [k["name"][:80] for k in r["kernels"]], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
