#!/usr/bin/env python3
import argparse, json, pathlib
import numpy as np
from nv_cleanroom_flash_pp512_oracle import make_fixture, fixture_identity

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', required=True)
ap.add_argument('--seed', type=int, default=20260830)
a = ap.parse_args()
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); f = make_fixture(a.seed)
arrays = {'q':f.q, 'k':f.k, 'v':f.v, 'expected':f.expected, 'mask':f.mask, 'canary':f.canary}
files = {}
for name, arr in arrays.items():
  path = out / (name + '.bin'); np.ascontiguousarray(arr).tofile(path)
  files[name] = {'file':path.name, 'dtype':str(arr.dtype), 'shape':list(arr.shape), 'strides_bytes':list(arr.strides), 'bytes':path.stat().st_size}
manifest = {'schema':'tinygrad.nv_cleanroom_flash_raw_fixture.v1', 'identity':fixture_identity(f),
  'geometry':{'S':512,'D':128,'Hq':32,'Hkv':8,'gqa':4}, 'layout':'C-contiguous head-major [heads,S,D]',
  'mask':'lower-right causal; mask[q,k] = (k <= q)', 'arrays':files,
  'output_contract':'candidate writes [32,512,128] fp32 logical output; canary is separate sentinel storage'}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(out / 'manifest.json')
