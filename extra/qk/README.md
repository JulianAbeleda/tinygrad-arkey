# QK Research Surface

`extra/qk` is the research and generated-route workspace. It is intentionally separate from `tinygrad/llm`, which owns the shipped runtime surface.

Use these roles when adding or reading files:

- `promoted`: files used by the runtime path or route manifests.
- `active`: current gates, audits, and microgates that decide whether a route can be promoted.
- `support`: reusable helpers, layouts, caches, lowering utilities, and harness contracts.
- `refuted`: historical probes that explain a closed path and should not be copied into runtime code.
- `scratch`: one-off experiments that should either be deleted or moved into BoltBeam once resolved.

Rules:

- New runtime-visible routes must enter through `tinygrad/llm/route_ops.py` or `tinygrad/llm/decode_routes.py`.
- New probes must have a clear role in `surface_audit.py`; unclassified files fail the audit.
- Refuted or scratch files are evidence, not defaults. Do not route to them from `tinygrad/llm`.
- Prefer adding reusable helpers to support files rather than copy-pasting kernels into new probes.

## The one rule for gates/audits/probes

**A new `active` experiment is a registry row + a `build()` — not a new file with its
own `main()`.** The gate surface grew to ~150 files because every experiment cloned its
own ROOT resolution, artifact writer, timestamp stamp, and exit logic. That scaffolding
now lives once in `gate_registry.py`. If you are about to copy a `main()` or a `build_*`
from another file, stop and extend the registry instead. (This is the `extra/qk` form
of the anti-re-sprawl rule in `structure/Development/tinygrad-coding-overrides.md`.)

## `gate_registry.py` — the single index of gates

Every pass/fail gate/audit/probe is one `GateSpec` row in `GATES` plus a pure `build()`
in its module. The runner owns everything the old mains cloned: ROOT, env-before-import
ordering, artifact write (`bench/<out_dir>/<artifact_name>`), optional dated snapshot,
stdout echo, traceback capture, exit-code policy.

```sh
PYTHONPATH=. python3 -m extra.qk.gate_registry list                        # all gates: kind, gpu, out_dir
PYTHONPATH=. python3 -m extra.qk.gate_registry list --no-gpu               # artifact-only subset
PYTHONPATH=. python3 -m extra.qk.gate_registry run <name> [<name>...]      # run specific gates
PYTHONPATH=. python3 -m extra.qk.gate_registry run --tranche artifact-only # non-GPU sweep = repo health check
```

`run --tranche artifact-only` is the post-refactor / post-prune health sweep: it
exercises every gate that only reads committed artifacts, so a broken `inputs=` path or
a stale canonical-doc list surfaces in one command.

### Adding a gate

1. Write `build()` in your module. Return **the verdict dict** (runner writes + prints
   it) or **an int exit code** (report-only checks that print their own findings).
   `build()` must NOT write artifacts, print the final JSON, or `sys.exit`. Keep
   env-sensitive tinygrad imports *inside* `build()` (lazy) so the module imports without
   freezing the AMD/JIT environment.
2. Add a `GateSpec` row: `name` (stable, no `_gate`/`_audit` suffix),
   `entry="extra.qk.<module>:build"`, `kind`, `needs_gpu`, `out_dir` (→ `bench/<out_dir>/`),
   `inputs` (repo-relative artifacts you read — declare them; they are grepped before
   prunes), `pass_verdicts` (frozenset meaning exit 0; `None` = exit 0 whenever `build()`
   completes), `env` (set BEFORE the entry import — sacred ordering), `artifact_name`
   (override when sharing a bench dir), `snapshot=True` (also write a dated copy).
3. Replace the module `__main__` with the standard shim:
   ```python
   if __name__ == "__main__":
     import sys; sys.path.insert(0, str(ROOT))
     from extra.qk.gate_registry import run
     raise SystemExit(run("<name>"))
   ```

### Parity when editing a gate (NFC discipline)

Byte-prove, don't assert: run old vs new, compare `latest.json` after normalizing only
genuinely run-volatile fields (`timestamp`; on GPU gates, kernel-timing `*_us`/`*_ms`
and uninitialized-slot reads — prove a field volatile by double-running the OLD script
first). Verdicts, numerics, booleans must match exactly. See `git log --grep gate_registry`.

## What is NOT a registry gate (leave standalone)

- **Benchmarks / A-B / W==D harnesses** (`*_ab.py`, `*_wd.py`, `prefix_cache_bench.py`,
  `quant/q4_k_bench.py`): they *measure*, they don't gate — a pass/fail schema is the
  wrong abstraction.
- **Full-model-loading tools** (`decode_token_match_check.py`,
  `decode_inkernel_combine_microgate.py`, `large_model_decode_route_gap_audit.py`): they
  load a real GGUF; run on demand, not in the health sweep.
- **CLI tools** (`experiment_matrix.py`): argv-parameterized, no fixed artifact.

## Timing controls

Short AMD benchmark windows must use the shared clock-pin wrapper when comparing
latency or throughput-sensitive runs:

```python
from extra.qk.timing_harness import add_clock_pin_arg, pinned_peak_from_env, set_clock_pin_env
```

`extra/qk/clock_pin.py` owns the privileged sysfs/`rocm-smi` mutations. `timing_harness.py`
owns the canonical `PREFILL_PIN_CLOCK=1` env flag and `--pin-clock` parser hook. Parent
tools set the env flag before spawning a worker; worker/probe code wraps only the measured
AMD window with `pinned_peak_from_env()`, records the yielded provenance in JSON, and
lets the context manager restore `auto` in `finally`.

## Load-bearing invariants (do not "simplify" away)

- **Env ordering is sacred:** `DEV`/`JIT`/`QK_*` before `from tinygrad import ...`. The
  registry `env=` field enforces it; gate modules import tinygrad lazily inside `build()`.
- **Subprocess isolation is intentional** where present (score-broadcast child dispatch,
  the `REG_STORE_DEVEC` fresh process in `tg_p10`, the ISA-capture subprocess in
  `decode_physical_tile.all_primitives`) — the runtime hook / memoized getenv don't
  compose in-process. Keep it.

Run the surface audit (itself a registry gate):

```sh
PYTHONPATH=. python3 -m extra.qk.gate_registry run surface
```
