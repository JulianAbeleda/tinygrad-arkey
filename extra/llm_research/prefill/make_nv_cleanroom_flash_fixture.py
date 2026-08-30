#!/usr/bin/env python3
import argparse, json, pathlib
import numpy as np
from nv_cleanroom_flash_pp512_oracle import make_fixture, fixture_identity

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=20260830)
a = ap.parse_args()
f = make_fixture(a.seed)
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, q=f.q, k=f.k, v=f.v, expected=f.expected, mask=f.mask, canary=f.canary)
meta = out.with_suffix('.json')
meta.write_text(json.dumps(fixture_identity(f), indent=2, sort_keys=True) + '\n')
print(meta)
