import json
A={ (r['role'],r['m']):r for r in json.load(open('scan/ncu_audit.json'))['rows']}
G={ (r['role'],r['m']):r for r in json.load(open('scan/gemm_scan.json'))['rows']}
o=json.load(open('ours.json')); P={(r['role'],r['m']):r for r in o if r['mode']=='prod'}
CNT={'ssm_in':21,'ssm_out':21,'attn_q':4,'attn_kv':8,'attn_o':4,'ffn_up':17,'ffn_down':17,'output':1}
NPAD={'ssm_in':(17504,17536),'ssm_out':(3136,3200),'attn_o':(3136,3200),'ffn_down':(3136,3200)}
W=(('B=32',32,1),('B=64',64,1),('B=128',128,1),('10k prefill',1024,10000/1024))
out={}
for w,m,mult in W:
  b={k:0.0 for k in ('ragged_k','stream_k','n_pad','single_stage','fragment_prefetch','barrier_64k','hilo_rows','aux(hi/lo split+sum, split-K reduce)','gemm_total_gap')}
  for r,c in CNT.items():
    if (r,m) not in A or (r,m) not in P: continue
    a=A[(r,m)]; k=a['gap_vs_reference']['known']; f=c*mult/1e3
    wt=max(0.0,k.get('wave_tail',0.0))
    cls=G[(r,m)].get('reference_pick_in_space','admitted')
    b['ragged_k' if 'ragged' in cls else 'stream_k']+=wt*f
    if r in NPAD: n0,n1=NPAD[r]; b['n_pad']+=P[(r,m)]['gemm_us']*(1-n0/n1)*f
    b['fragment_prefetch']+=(k.get('stall_excess:mio_throttle',0)+k.get('stall_excess:short_scoreboard',0))*f
    b['barrier_64k']+=k.get('stall_excess:barrier',0)*f
    b['hilo_rows']+=k.get('rows',0)*f
    aux=P[(r,m)]['gpu_us']-P[(r,m)]['gemm_us']
    b['aux(hi/lo split+sum, split-K reduce)']+=(aux - (a['reference']['aux_us']))*f
    b['gemm_total_gap']+=(P[(r,m)]['gpu_us']-a['reference']['total_us'])*f
  out[w]=b
keys=list(next(iter(out.values())))
print('| lever | '+' | '.join(w for w,_,_ in W)+' |'); print('|---'*(len(W)+1)+'|')
for k in keys: print(f'| {k} | '+' | '.join(f"{out[w][k]:.2f}" for w,_,_ in W)+' |')
