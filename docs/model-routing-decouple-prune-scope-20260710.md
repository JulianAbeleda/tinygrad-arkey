# Model Routing Decouple + Prune Scope - 2026-07-10

Goal: reduce handwritten core size without deleting live model behavior. The target is the route/model-specific
scaffolding added during the Qwen 8B machine-search push, while preserving GGUF load -> model facts -> route plan ->
primitive install -> generate.

## Baseline

Before pruning, the real load/generate path was exercised:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python - <<'PY'
from tinygrad import Tensor, GlobalCounters
from tinygrad.llm.model import Transformer
model, kv = Transformer.from_gguf('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf', 512)
print('loaded', kv.get('general.architecture'), 'ctx', model.max_context)
Tensor.manual_seed(0)
g = model.generate([0])
for i in range(3):
  GlobalCounters.reset()
  tok = next(g)
  print('tok', i, tok)
print('ok')
PY
```

Result: `loaded qwen3 ctx 512`, tokens `50994, 82, 31109`, `ok`.

The retained wrapper `extra/llm/model_e2e_bench.py` was repaired in Wave 0: it no longer imports the deleted
`extra.qk.harness_contract`, and its small median/spread helper is local to the script. Do not restore
`harness_contract` for this path.

Current size checkpoints:

- `python3 sz.py`: `27994 / 31000` authored budgeted lines.
- raw handwritten `tinygrad/` Python excluding `tinygrad/runtime/autogen`: `37164` lines.
- `tinygrad/llm`: `3989` lines.
- `extra/qk`: `22348` lines across 103 Python files.

## Principles

- Prune only code whose callers prove it is not live, or decouple first until the caller graph becomes obvious.
- Keep CUDA/NVIDIA thin code in place; it is next-project surface, not this cleanup target.
- Runtime dispatch should route from tensor facts, quant type, role, shape, architecture, and policy data; not from
  model-size if/else trees.
- Authority artifacts may stay model-specific when they are evidence gates. Runtime scripts should not multiply per
  model.

## Wave 0 - Repair The E2E Authority Surface

`model_e2e_bench.py` is documented as the retained generate harness in `README.md` and
`docs/harness-consolidation.md`. The missing-module repair has landed:

- done: removed the stale `harness_contract` dependency and kept `model_e2e_bench.py` self-contained;
- still open: decide whether future README model tables should stay on generate-window E2E artifacts or migrate to
  `model_authority_bench.py` fixed-context authority artifacts;
- not acceptable: restore a broad shared harness module only to satisfy stale docs.

Done gate: both the short direct generate smoke and the retained documented E2E command run.

## Wave 1 - Delete Unearned Dispatch Framework

The unused generic route dispatch layer and its self-test were only exercised by their own docs/tests. Live decode and
prefill dispatch bind routes directly from model facts, policy, and primitive installers.

Delete those files and remove the stale docs. Expected savings: 92 core lines plus 75 test lines. This is the cleanest
immediate prune because no runtime decouple is needed.

Done gate: no references to the removed dispatch-layer type names remain outside git history.

## Wave 2 - Move Legacy Q4/Q6 Defaults Out Of Route Policy

`tinygrad/llm/route_policy.py` currently does two jobs:

- BoltBeam route-policy loading and route selection;
- legacy name-substring defaults for Q4_K/Q6_K primitive installation.

The second job should move into model-fact route planning. `model_route_plan.py` already sees tensor name, quant type,
rows, cols, module path, and role; its default entries should be expressed as role/quant/shape rules there, not as
`q4k_policy(name)` / `q6k_policy(name)` calls in the route-policy module.

Required steps:

- encode default Q4_K/Q6_K install decisions by `quant_label`/`typ` plus `role`;
- preserve the current defaults exactly for `ffn_gate`, `ffn_up`, `ffn_down`, `attn_q`, `attn_output`, `attn_k`,
  `attn_v`, Q6 `attn_v`, and Q6 `output.weight`;
- replace `_demote_q6k_to_q4`'s `_q4k_policy` fallback with role/shape logic or a model-route-plan entry;
- remove parity tests that bless the legacy policy API after the new fact-driven tests prove the same plan rows.

Done gate: `route_policy.py` only owns generated policy caches and route selection; Q4/Q6 primitive install defaults
live with model route planning.

## Wave 3 - Prune Stale Harness Docs And Merge Paths

After Wave 0, update or remove stale claims:

- `docs/harness-consolidation.md` should say `extra/qk/harness_contract.py` is removed, not live;
- `extra/llm/llama_cpp_bench.py` may continue merging into `model_e2e_bench` artifacts while README uses those
  artifacts, but it becomes a migration target if README moves to fixed-context authority artifacts;
- `extra/llm/model_authority_bench.py` should be described as the fixed-context authority artifact path, not an
  automatic replacement for the retained E2E artifact path.

Do not delete authority gates just because they are model-specific. Keep the 14B policy gates while they remain the
evidence surface for 14B.

## Wave 4 - Keep Scripts Profile-Driven, Not Model-Tree-Driven

The desired shape is one script taking `--profile`/roles where possible. Keep model-specific wrappers only when they are
evidence gates with different acceptance criteria.

Audit target:

- keep `extra/qk/model_profiles.py` as data;
- keep converted profile-driven scripts such as `s10_hybrid_role_trace.py`;
- convert remaining safe runtime/search scripts from embedded 8B/14B constants to profile lookup;
- leave exact authority gates intact if conversion would weaken their failure condition.

## Verification Matrix

Run after every wave:

```bash
python3 sz.py
python3 -m pytest test/unit/test_model_facts.py test/unit/test_model_route_plan.py \
  test/unit/test_llm_decode_routes.py test/unit/test_llm_prefill_routes.py \
  test/unit/test_qk_route_purity.py test/unit/test_tinygrad_boundary.py
```

Run after Wave 0 and final:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/llm/model_e2e_bench.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --id qwen3-8b-prune \
  --max_context 512 --decode-tokens 8 --warmup-skip 2 --prefill 0 \
  --out /tmp/qwen3-8b-prune.json
```

Run final:

```bash
python3 -m pytest
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python - <<'PY'
from tinygrad import Tensor, GlobalCounters
from tinygrad.llm.model import Transformer
model, kv = Transformer.from_gguf('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf', 512)
Tensor.manual_seed(0)
g = model.generate([0])
for _ in range(3):
  GlobalCounters.reset()
  print(next(g))
PY
```
