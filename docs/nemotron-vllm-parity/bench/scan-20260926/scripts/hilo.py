import json
o=json.load(open('ours.json')); v=json.load(open('vllm_gemm.json'))+json.load(open('vllm_gemm_256.json'))
V={(r['role'],r['m']):r['gpu_us'] for r in v}
P={(r['role'],r['m']):r['gpu_us'] for r in o if r['mode']=='prod'}; B={(r['role'],r['m']):r['gpu_us'] for r in o if r['mode']=='bf16'}
CNT={'ssm_in':21,'ssm_out':21,'attn_q':4,'attn_kv':8,'attn_o':4,'ffn_up':17,'ffn_down':17,'output':1}
MS=[8,16,32,64,128,512,1024]
def bbest(r,m): return min(B[(r,x)] for x in MS if x>=m and (r,x) in B and x<=max(m,128) or (x==m and (r,x) in B))
print("| role | M | ours hi/lo (2M rows) | ours plain bf16 (M rows, best route >= M) | vLLM M rows | vLLM 2M rows | hi/lo tax ours | hi/lo tax cuBLAS |")
print("|---|---|---|---|---|---|---|---|")
for r in CNT:
  for m in (32,64,128,1024):
    if (r,m) not in P: continue
    b=bbest(r,m); v1=V[(r,m)]; v2=V[(r,2*m)]
    print(f"| {r} | {m} | {P[(r,m)]:.1f} | {b:.1f}{'' if b==B[(r,m)] else ' ('+str(B[(r,m)])[:6]+' at M)'} | {v1:.1f} | {v2:.1f} | {P[(r,m)]/b:.2f}x | {v2/v1:.2f}x |")
print()
print("| workload | ours hi/lo ms | ours plain bf16 ms | vLLM (M rows) ms | vLLM at 2M rows ms | hi/lo tax, our kernels | hi/lo tax, cuBLAS kernels | kernel gap at plain bf16 |")
print("|---|---|---|---|---|---|---|---|")
for w,m,mult in (('decode B=32',32,1),('decode B=64',64,1),('decode B=128',128,1),('10k prefill (GEMMs, 9.77 x 1024 pieces)',1024,10000/1024)):
  s=lambda f: sum(CNT[r]*f(r) for r in CNT if (r,m) in P and not (r=='output' and m>128))*mult/1e3
  h=s(lambda r:P[(r,m)]); b=s(lambda r:bbest(r,m)); v1=s(lambda r:V[(r,m)]); v2=s(lambda r:V[(r,2*m)])
  print(f"| {w} | {h:.2f} | {b:.2f} | {v1:.2f} | {v2:.2f} | +{h-b:.2f} | +{v2-v1:.2f} | +{b-v1:.2f} |")
