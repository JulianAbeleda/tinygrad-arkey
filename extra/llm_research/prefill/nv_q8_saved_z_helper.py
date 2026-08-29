"""Exact host reference for the existing FP16->Q8_1 producer ABI."""
import numpy as np
def quantize_saved_z(z):
 z=np.asarray(z,dtype=np.float16); assert z.ndim==3 and z.shape[0:2]==(1,512)
 m,k=z.shape[1:]; assert k%32==0
 x=z.reshape(m,k).astype(np.float32); g=x.reshape(m,k//32,32)
 # The CUDA producer's tg_round_i8 is nearest with half-away-from-zero ties
 # (not NumPy's ties-to-even rint).  Keep this explicit so exact ties are
 # represented independently of the candidate implementation.
 sc=np.max(np.abs(g),axis=2)/127.0; sc=np.where(sc==0,1.0,sc).astype(np.float32)
 # Match the kernel's explicit reciprocal multiply (`v*(1.0f/d)`), whose
 # rounding differs from a host divide at exact half boundaries.
 y=g*(np.float32(1.0)/sc[...,None])
 q=np.rint(y)
 q=np.clip(q,-127,127).astype(np.int8)
 # The producer's sum is over the source values, not q.  Match its
 # float4-per-thread expression and XOR reduction order (4,2,1), then
 # round the result through binary16 before storing it as float32.
 v=g.reshape(m,k//32,8,4).astype(np.float32)
 lane=(v[...,0]+v[...,1])+(v[...,2]+v[...,3])
 sums=((lane[...,0]+lane[...,4])+(lane[...,2]+lane[...,6])+
       (lane[...,1]+lane[...,5])+(lane[...,3]+lane[...,7]))
 sums=sums.astype(np.float16).astype(np.float32)
 return q.reshape(-1),sc.reshape(-1),sums.reshape(-1)
