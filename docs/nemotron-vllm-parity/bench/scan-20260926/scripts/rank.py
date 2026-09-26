import json, sys
o=json.load(open('/home/ubuntu/storage/scan-20260926/ours.json'))
v=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/home/ubuntu/storage/scan-20260926/vllm_gemm.json'))
V={(r['role'],r['m']):r['gpu_us'] for r in v}
P={(r['role'],r['m']):r['gpu_us'] for r in o if r['mode']=='prod'}
CNT={'ssm_in':21,'ssm_out':21,'attn_q':4,'attn_kv':8,'attn_o':4,'ffn_up':17,'ffn_down':17,'output':1}
W={'B32':(32,1),'B64':(64,1),'B128':(128,1),'pf10k':(1024,10000/1024)}
rows=[]
for w,(m,mult) in W.items():
  for r,c in CNT.items():
    if (r,m) not in P: continue
    vm = V[(r,m)] if r!='attn_kv' else V[(r,m)]
    rows.append((c*mult*(P[(r,m)]-vm)/1e3, w, r, m, P[(r,m)], vm))
for x in sorted(rows, reverse=True)[:14]: print(f'{x[0]:8.3f} ms {x[1]:6s} {x[2]:8s} M={x[3]} ours {x[4]:.1f} vllm {x[5]:.1f}')
