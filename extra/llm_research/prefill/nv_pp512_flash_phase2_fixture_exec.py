#!/usr/bin/env python3
import pathlib, subprocess, sys, numpy as np
z=np.load(sys.argv[1]); out=pathlib.Path(sys.argv[2]); raw=out.with_suffix('.f32')
q=raw.with_suffix('.q'); k=raw.with_suffix('.k'); v=raw.with_suffix('.v')
z['q'].astype(np.float32).tofile(q); z['k'].astype(np.float16).tofile(k); z['v'].astype(np.float16).tofile(v)
subprocess.run(['/tmp/nv_pp512_flash_phase2_kstage',str(q),str(k),str(v),str(raw)],check=True)
np.save(out,np.fromfile(raw,dtype=np.float32).reshape(32,512,128))
