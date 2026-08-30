#!/usr/bin/env python3
"""Adapter for Phase-1/2 Flash candidates using the clean-room fixture."""
import argparse, json, pathlib, subprocess
import numpy as np
from nv_cleanroom_flash_pp512_oracle import FlashFixture, check_output

ap = argparse.ArgumentParser()
ap.add_argument('--fixture', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--command')
ap.add_argument('--result', required=True)
a = ap.parse_args()
p = pathlib.Path(a.fixture)
z = np.load(p)
f = FlashFixture(z['q'], z['k'], z['v'], z['expected'], z['mask'], z['canary'])
if a.command: subprocess.run(a.command.format(fixture=str(p), output=a.output), shell=True, check=True)
actual = np.load(a.output)
metrics = check_output(actual, f)
result = {'schema':'tinygrad.nv_cleanroom_flash_fixture_result.v1', 'fixture':str(p), 'output':a.output,
  'geometry':{'S':512,'D':128,'Hq':32,'Hkv':8,'gqa':4},
  'strides':{'q':list(f.q.strides),'k':list(f.k.strides),'v':list(f.v.strides),'output':list(actual.strides)},
  'canary_check':'candidate ABI must preserve the separately allocated canary; runner validates logical full output only',
  'correctness':metrics, 'status':'PASS' if metrics['finite'] and metrics['allclose'] else 'STOP'}
pathlib.Path(a.result).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
