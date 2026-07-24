# Attention Harness Centralization — Scope (2026-07-24)

## Goal
Extract the attention-harness RUNTIME PRIMITIVES that are copy-pasted across ~50 harness files into ONE
shared module, `extra/qk/attention_harness_common.py`, and re-point callers at it. Reuse, don't rewrite.

## Purpose
Every correctness/benchmark harness (and our new custom-kernel injection route) reimplements the same
mask / reference-SDPA / test-input / timing / proof / sha helpers, with subtle drift. Centralizing them
gives ONE golden `reference_attention` (so A4 correctness claims are trustworthy), one causal mask, one
timing authority, one candidate-context builder — shared by all harnesses.

## Canonical source (extract VERBATIM from here)
`extra/qk/benchmark_shared_attention.py` has the cleanest copies. Extract these EXACT bodies:
- `_sha(x)`                            (line 16)  -> `content_sha(text)`
- `_mask(q,kv,start)`                  (line 17)  -> `causal_mask(q_tokens, kv_tokens, start_pos)`
- `_sync()`                            (line 18)  -> `amd_sync()`
- `_proof(path)`                       (lines 19-25) -> `load_shared_attention_proof(path)`
- `_inputs(hq,hkv,q,kv,seed)`          (lines 26-29) -> `make_qkv(hq, hkv, q_tokens, kv_tokens, seed)`
- `_baseline(q,k,v,mask,ctx)`          (lines 31-33) -> `reference_attention(q, k, v, mask, hq, hkv)`  # THE golden
- `_time(fn,warmup,samples)`           (lines 34-39) -> `synced_time(fn, warmup, samples)`
- `_summary(x)`                        (lines 40-41) -> `timing_summary(values)`
- candidate-ctx build                  (line 59)  -> `candidate_context(profile, strategy, hq, hkv, kv, *, q_tokens=512, hd=128, causal=True, start_pos=None)`
                                                       where `start_pos = kv - q_tokens if start_pos is None`
- `ROUTES` table                       (line 14)  -> module-level `ROUTES` (8B/14B geometry)

### Exact new-module contract (must be behavior-identical to the extracted bodies)
```python
# extra/qk/attention_harness_common.py
import hashlib, statistics, time
import numpy as np
from tinygrad import Tensor, dtypes, Device

ROUTES = (("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",32,8),
          ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",40,8))

def content_sha(text:str) -> str: return hashlib.sha256(text.encode()).hexdigest()
def causal_mask(q_tokens:int, kv_tokens:int, start_pos:int) -> Tensor:
  return Tensor.full((1,1,q_tokens,kv_tokens), float("-inf"), dtype=dtypes.float16, buffer=False).triu(start_pos+1)
def amd_sync() -> None: Device["AMD"].synchronize()
def make_qkv(hq:int, hkv:int, q_tokens:int, kv_tokens:int, seed:int):
  rng = np.random.default_rng(seed)
  vals = [rng.normal(0,.04,(1,h,q_tokens if n==0 else kv_tokens,128)).astype(np.float16) for n,h in enumerate((hq,hkv,hkv))]
  return tuple(Tensor(x, device="AMD") for x in vals), vals
def reference_attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor, hq:int, hkv:int) -> Tensor:
  g = hq // hkv
  return q.scaled_dot_product_attention(k.repeat_interleave(g,dim=-3), v.repeat_interleave(g,dim=-3), attn_mask=mask)
def synced_time(fn, warmup:int, samples:int):
  for _ in range(warmup): fn().realize()
  amd_sync(); out=[]
  for _ in range(samples):
    amd_sync(); st=time.perf_counter_ns(); fn().realize(); amd_sync(); out.append((time.perf_counter_ns()-st)/1e6)
  return out
def timing_summary(values):
  x=sorted(values); return {"raw_ms":x,"median_ms":statistics.median(x),"p10_ms":np.percentile(x,10).item(),"p90_ms":np.percentile(x,90).item()}
def load_shared_attention_proof(path):
  # EXACT copy of benchmark_shared_attention._proof (schemas set + PASS gate)
  ...
def candidate_context(profile, strategy, hq, hkv, kv, *, q_tokens=512, hd=128, causal=True, start_pos=None):
  from tinygrad.uop.ops import SharedAttentionCandidateContext
  sp = kv - q_tokens if start_pos is None else start_pos
  return SharedAttentionCandidateContext(profile, strategy, q_tokens, kv, sp, hq, hkv, hd, causal).validate()
```

## Migration targets (replace local def with import; ONLY when the local body is character-identical)
Grep each file for the local helper; if its body matches the canonical one, delete the local def and import
from `attention_harness_common`. If a file's copy DIFFERS in any way, DO NOT migrate it — leave it and add a
one-line `# TODO(centralize): differs from attention_harness_common.<fn>` note. Report all such divergences.

- `extra/qk/benchmark_shared_attention.py` — `_sha,_mask,_sync,_proof,_inputs,_baseline,_time,_summary`, ROUTES,
  and the line-59 candidate-context build. (This is the canonical source; migrate it too so it consumes the module.)
- `extra/qk/generate_shared_attention_captures.py` — `_mask`, candidate-context build, and any test-input build.
- `extra/qk/benchmark_split_shared_attention.py` — `Device.synchronize` timing helper if a local `_sync`/`_time` exists.
- `extra/qk/shared_attention_capture.py` — candidate-context build.
- `extra/qk/shared_attention_evidence.py` — proof load.
- `extra/qk/prefill_whole_synced.py` — reference SDPA + sync ONLY IF a byte-identical local copy exists (this file
  is subtle/authoritative; prefer leaving it and just NOTE the overlap rather than risk it).
- sha256: files with a bare `def _sha(x): return hashlib.sha256(x.encode()).hexdigest()` may migrate to
  `content_sha`. Files whose `_sha` differs (e.g. hashes bytes, or a different digest) MUST NOT be migrated.

## INVARIANTS (do not break)
1. Behavior must be byte-identical. These are proof-gated numeric benchmarks; a changed reference/mask/seed/sha
   silently invalidates artifacts. Extract VERBATIM; do not "improve".
2. Do NOT touch: `extra/qk/prefill_harness.py` and `extra/qk/model_profiles.py` (config, already central);
   `extra/llm/eval_common.py` (JSON/eval plumbing — out of scope here).
3. Do NOT change any harness's CLI, output schema, artifact contents, or numeric tolerances.
4. `reference_attention` must remain the exact `scaled_dot_product_attention` + `repeat_interleave(hq//hkv)`
   form — this is the correctness golden; any deviation is a false-pass risk.

## Validation the agent MUST run (no GPU required for most)
1. `PYTHONPATH=/home/ubuntu/tinygrad-arkey .venv/bin/python -c "import extra.qk.attention_harness_common"` — imports.
2. For EVERY file touched: `.venv/bin/python -c "import <module.path>"` — still imports.
3. Source-equivalence check: for each migrated helper, confirm the extracted body is character-identical to the
   original (diff the old local def against the new module fn) and report a table of {file, helper, migrated|differs}.
4. Do NOT attempt GPU/behavioral runs (no DEV=AMD execution) — leave that to the main loop. Just ensure imports.

## Out of scope
- No refactor of harness LOGIC (only the shared leaf helpers).
- No changes to `fused_attention.py` (main loop will re-point it at `candidate_context`/`reference_attention` after).
- No new tests, no perf runs, no artifact regeneration.

## Deliverable
The new module + migrated imports + a report table of {file, helper, action(migrated/differs/skipped), reason}.
```
