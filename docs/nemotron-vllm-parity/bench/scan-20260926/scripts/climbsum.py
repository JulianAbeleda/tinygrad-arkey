import json, glob, os
v=json.load(open('vllm_gemm.json'))+json.load(open('vllm_gemm_256.json')); V={(r['role'],r['m']):r['gpu_us'] for r in v}
sel={(r['role'],r['m']):r for r in json.load(open('/home/ubuntu/tinygrad-self-training/extra/llm_research/prefill/dense_bf16_sm120_selection.json'))['rows']}
o=json.load(open('ours.json')); P={(r['role'],r['m'],r['mode']):r for r in o}
print("| role | GEMM rows | measured | promoted route (re-measured) us | best derived measured us | best config (tile m,n,k / warps / split / stages) | TF | vLLM same rows us | best/vLLM same rows | current production GEMM us at these rows |")
print("|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob('climb/*.jsonl'), key=lambda f:(f.split('/')[1].rsplit('-',1)[0], int(f.rsplit('-',1)[1][:-6]))):
  role, rows = os.path.basename(f)[:-6].rsplit('-',1); rows=int(rows)
  R=[json.loads(l) for l in open(f) if l.strip()]
  ok=[r for r in R if 'median_us' in r and r.get('finite') and not r.get('error') and r.get('max_rel_vs_unsplit',0)<=1e-4]
  b=min(ok,key=lambda r:r['median_us'])
  p=sel.get((role,rows)); pm='-'
  if p:
    pc=(p['geometry'],p['split_k'],p.get('pipeline',[2,False,False,False]))
    pr=[r for r in R if (r['geometry'],r['split_k'],r.get('pipeline',[2,False,False,False]))==pc and 'median_us' in r]
    pm=f"{pr[0]['median_us']}" if pr else f"{p['search_median_us']} (gate)"
  # current production GEMM at these rows: hi/lo prod at rows/2 tokens, or bf16 at rows tokens
  cur=[]
  if (role,rows//2,'prod') in P: cur.append(f"hi/lo M={rows//2}: {P[(role,rows//2,'prod')]['gemm_us']:.0f}")
  if (role,rows,'bf16') in P: cur.append(f"bf16 M={rows}: {P[(role,rows,'bf16')]['gemm_us']:.0f}")
  g=b['geometry']; vs=V.get((role,rows))
  print(f"| {role} | {rows} | {len(R)} | {pm} | {b['median_us']} | {g[0]}x{g[1]}x{g[2]} / {g[3]}x{g[4]} / s{b['split_k']} / p{(b.get('pipeline') or [2])[0]}{'r' if (b.get('pipeline') or [0,0])[1] else ''} | {b['tflops']} | {vs:.1f} | {b['median_us']/vs:.2f} | {'; '.join(cur) or '-'} |")
