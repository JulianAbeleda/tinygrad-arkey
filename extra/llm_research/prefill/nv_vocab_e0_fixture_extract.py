#!/usr/bin/env python3
"""Extract the E0 real Q6 lm-head bytes and final composed-prefill hidden row."""
import argparse, hashlib, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = pathlib.Path('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf')

def sha_bytes(b): return hashlib.sha256(b).hexdigest()

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--model', type=pathlib.Path, default=MODEL)
  ap.add_argument('--depth', type=int, default=512)
  ap.add_argument('--out-dir', type=pathlib.Path, required=True)
  a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
  from tinygrad.llm.gguf import gguf_load_metadata
  _kv, meta = gguf_load_metadata(a.model)
  infos = {x[0]: x for x in meta['tensor_infos']}
  name = 'output.weight'
  if name not in infos: raise RuntimeError('current GGUF has no output.weight')
  _n, shape, typ, off = infos[name]
  if tuple(shape) != (4096, 151936) or typ != 14: raise RuntimeError(f'unexpected {name}: {infos[name]}')
  nbytes = 151936 * 4096 // 256 * 210
  raw = np.memmap(a.model, mode='r', dtype=np.uint8, offset=meta['data_start'] + off, shape=(nbytes,))
  q6_path = a.out_dir / 'output.weight.q6_k.bin'; q6_path.write_bytes(raw.tobytes())
  # Capture the exact tensor consumed by the live output projection, after all
  # current composed blocks and output RMSNorm have run.
  from tinygrad import Tensor
  from tinygrad.llm.generate import load_model_and_tokenizer
  from extra.llm_research.decode.decode_runtime_overhead import _make_prompt
  model, tok = load_model_and_tokenizer(a.model, a.depth + 1, seed=20260829)
  base = (tok.prefix() if hasattr(tok, 'prefix') else []) + tok.encode('the quick brown fox jumps. ' * 800)
  ids = _make_prompt(base, a.depth)
  captured = []
  from tinygrad import nn
  original = nn.RMSNorm.__call__
  def capture(norm, x):
    out = original(norm, x)
    if norm is model.output_norm: captured.append(out)
    return out
  nn.RMSNorm.__call__ = capture
  try: logits = model.logits(Tensor([ids], dtype='int32'), 0).realize()
  finally: nn.RMSNorm.__call__ = original
  if len(captured) != 1: raise RuntimeError(f'output norm capture count={len(captured)}')
  # Capture the exact post-output_norm row consumed by the output projection,
  # including the loaded RMSNorm weight, epsilon, dtype cast, and operation
  # ordering.  Capturing x here would leave the fixture at the pre-norm
  # boundary and make the Q8 producer contract invalid.
  hidden = captured[0][:, -1:, :].realize().numpy().astype(np.float32)
  if hidden.shape != (1, 1, 4096): raise RuntimeError(f'hidden shape={hidden.shape}')
  hidden = hidden.reshape(1, 4096); hidden_path = a.out_dir / 'final-hidden-row.f32'; hidden.tofile(hidden_path)
  out_path = a.out_dir / 'reference-logits.f32'; logits.numpy()[:, -1, :].astype(np.float32).tofile(out_path)
  sent = {k: float(hidden[0, i]) for k, i in [('first',0), ('middle',2048), ('last',4095)]}
  result = {'schema':'tinygrad.nv_vocab_e0_fixture.v1','model':str(a.model.resolve()),
    'model_sha256':sha_bytes(a.model.read_bytes()), 'route':'current Transformer.logits composed prefill',
    'depth':a.depth, 'q6_weight':{'path':str(q6_path),'shape':[151936,4096],'dtype':'Q6_K packed','bytes':nbytes,'sha256':sha_bytes(q6_path.read_bytes())},
    'final_hidden_row':{'path':str(hidden_path),'shape':[1,4096],'dtype':'float32','bytes':hidden_path.stat().st_size,'sha256':sha_bytes(hidden_path.read_bytes()),'sentinels':sent},
    'reference_logits':{'path':str(out_path),'shape':[1,151936],'dtype':'float32','sha256':sha_bytes(out_path.read_bytes())},
    'tolerance':{'rtol':0.02,'atol':0.5}, 'replay_command':f'python3 {pathlib.Path(__file__).resolve()} --model {a.model} --depth {a.depth} --out-dir {a.out_dir}'}
  (a.out_dir/'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
  print(json.dumps(result, sort_keys=True))
if __name__ == '__main__': main()
