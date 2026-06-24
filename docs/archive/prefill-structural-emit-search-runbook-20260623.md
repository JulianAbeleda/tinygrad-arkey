# Prefill structural-emit search runbook (2026-06-23)

## Why this document
Capture the exact sweep workflow used in this session, so you can resume in a tmux shell without re-discovering state.

## Scope
Searching `extra/qk_prefill_emit_search.py` candidate emit knobs for prefill GEMM route, including:
- LDS structural variants (`DBUF`, `PLRA`, `BK`, `PAD`)
- low-risk reloc/proportional knobs (`RELOC`, `RELOC_MAX_WGS`)
- structural pipeline frontier (`PIPELINE`, `PIPELINE_TM`, `PIPELINE_TN`)
- legacy probes (`8WAVE`, `LEANADDR`)

## Code changes made before the sweep
1) `extra/qk_prefill_emit_search.py` already updated for:
- strict filtering and significance gates
- multiple-comparison modes: `none`, `bonferroni`, `fdr`
- confirm pre-pass + pairwise confirm helpers
- richer candidate classification/risk metadata
- deterministic FDR mapping and cleaner pass logic

2) `extra/qk_prefill_graph_gemm_route.py` already contains occupancy-aware relocation gating:
- `PREFILL_GEMM_RELOC_MAX_WGS` controls activation threshold
- default route has relocation opt-in semantics preserved in environment
- default candidate path in search set now includes both `dbuf_reloc` and `dbuf_reloc_wgs4`

## Commands used in this session

### 1) Candidate set and flags used
Both runs in this session used:
- Candidate set: `--candidates default`
- Filter mode: `--strict`
- Correlation mode: `--corr-mode fdr` (alpha default `0.05`)
- Confirm disabled: `--confirm-k 0`
- Environment: `DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=.`
- Output root: `--out /tmp/prefill-emits`

### 1a) Script-level guardrails now enforced in `extra/qk_prefill_emit_search.py`
- `--spec` input is validated before launch.
  - Every row must be `[name, env]` with `env` as an object.
  - If `baseline_current_default` is present but not first, it is automatically re-ordered to first (recorded to stderr).
  - If no baseline row exists, the harness injects `baseline_current_default: {}` as candidate #1 and continues (recorded to stderr).
- If `--confirm-k > 0` but only baseline is present after normalization, confirm is disabled (`--confirm-k 0`) with a guardrail warning.
- Baseline failures now print an explicit reason, return code, and captured stderr tail (`BASELINE_ERROR`) to avoid the prior opaque `unknown` path.

### 2) Full sweep (strict + FDR) — attempted and intentionally stopped
```bash
cd /home/ubuntu/tinygrad-arkey
timeout 3600 env DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. \
  .venv/bin/python extra/qk_prefill_emit_search.py \
  --candidates default --strict --corr-mode fdr --repeats 6 \
  --contexts 512,1024,2048,4096,8192 --maxc 8704 --out /tmp/prefill-emits --confirm-k 0
```

Notes:
- This command was running when we decided to stop (~27m43s in), mid-pipeline.
- Baseline and a number of candidates had completed before halt.

### 3) Quick sweep (strict + FDR) that completed
```bash
cd /home/ubuntu/tinygrad-arkey
timeout 2400 env DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. \
  .venv/bin/python extra/qk_prefill_emit_search.py \
  --candidates default --strict --corr-mode fdr --repeats 3 \
  --quick --contexts 512,4096 --maxc 4608 --out /tmp/prefill-emits --confirm-k 0
```

Artifacts produced by this exact completed run:
- `/tmp/prefill-emits/emit-search-20260623-150134.json`
- `/tmp/prefill-emits/emit-search-20260623-150134.md`
- `/tmp/prefill-emits/emit-search-20260623-150134.csv`

### 4) Controlled stop / recovery commands
```bash
pkill -f "qk_prefill_emit_search.py"
```

For PID-level cleanup:
```bash
ps -ef | rg "qk_prefill_emit_search.py"
pkill -f "qk_prefill_emit_search.py"
```

Recovery before restart:
- Check partial run artifacts/logs:
  - `ls -1t /tmp/prefill-emits/emit-search-* | head`
  - `ls -1t /tmp/prefill-emits/full-sweep-*.log`
- Reopen latest JSON summary and confirm command args before continuing.

## Resume plan in tmux

Recommended workflow:

1. Start tmux session:
```bash
tmux new -s prefill-search
```

2. Run full sweep with explicit logs:
```bash
cd /home/ubuntu/tinygrad-arkey
mkdir -p /tmp/prefill-emits
nohup env DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. \
  .venv/bin/python extra/qk_prefill_emit_search.py \
  --candidates default --strict --corr-mode fdr --repeats 6 \
  --contexts 512,1024,2048,4096,8192 --maxc 8704 --out /tmp/prefill-emits \
  --confirm-k 0 \
  > /tmp/prefill-emits/full-sweep-$(date +%F-%H%M%S).log 2>&1 &
```

Current active full sweep in this session (launched here):
```bash
setsid env DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. \
  .venv/bin/python extra/qk_prefill_emit_search.py \
  --candidates default --strict --corr-mode fdr --repeats 6 \
  --contexts 512,1024,2048,4096,8192 --maxc 8704 --out /tmp/prefill-emits \
  --confirm-k 0 \
  > /tmp/prefill-emits/full-sweep-active.log 2>&1 &
```

### Live state (as of now)

- Active process: `814890` (launcher) with worker `814891`
- Command currently running:
  - `env DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_emit_search.py --candidates default --strict --corr-mode fdr --repeats 6 --contexts 512,1024,2048,4096,8192 --maxc 8704 --out /tmp/prefill-emits --confirm-k 0`
- Log file: `/tmp/prefill-emits/full-sweep-active.log`
- Status: actively running, baseline candidate in progress; only first line has been emitted so far.

3. Detach and later reattach:
```bash
tmux detach
# ... later
tmux attach -t prefill-search
```

4. Monitor and stop safely if needed:
```bash
tail -f /tmp/prefill-emits/full-sweep-*.log
ps -ef | rg "qk_prefill_emit_search.py"
```

5. On completion, gather outputs:
```bash
tail -n 60 /tmp/prefill-emits/full-sweep-*.log
ls -1t /tmp/prefill-emits/emit-search-*.json | head
```

6. Parse JSON ranking quickly:
```bash
python - <<'PY'
import glob, json
p = sorted(glob.glob('/tmp/prefill-emits/emit-search-*.json'))[-1]
print(p)
j = json.load(open(p))
print('baseline:', j.get('baseline'))
for x in j.get('ranking', [])[:10]:
    print(f"{x['candidate']}: {x['delta_pct@4096']:.2f}% pass={x['passes']} decision={x['decision']}")
PY
```

## Quick result snapshot (from completed strict quick run `emit-search-20260623-150134`)
- Exact command used: `--candidates default --strict --corr-mode fdr --repeats 3 --quick --contexts 512,4096 --maxc 4608 --confirm-k 0`
- Raw top candidates (all significant but currently `needs_review` due strict-gate failure at current context set):
  - `pipe_tm2_tn2` (`+81.16%`)
  - `pipe_tm4_tn2` (`+41.69%`)
  - `pipe_tm2_tn4` (`+14.24%`)
- Strict-filter survivors (only these currently `needs_confirm`):
  - `old_plra` (`+1.38%`, risk=0)
  - `eightwave` (`+2.21%`, risk=2)
- No confirms were executed in this run because `--confirm-k 0`.
- `dbuf_reloc_wgs4`, `dbuf_reloc`, `dbuf_bk16` remained below strict filter floor for this quick context sample.
- `pipe_tm4_tn4`, `plra_bk16`, `leanaddr` remained infeasible.

## Exact /tmp/prefill-emits artifact map
- Per-run outputs:
  - `emit-search-20260623-150134.json`
  - `emit-search-20260623-150134.md`
  - `emit-search-20260623-150134.csv`
- Live session logs:
  - `full-sweep-YYYY-MM-DD-HHMMSS.log`
- Historical outputs from future/full quick runs:
  - `emit-search-*.json`
  - `emit-search-*.md`
  - `emit-search-*.csv`

## Open items / next steps
- Resume and complete full strict/FDR sweep in tmux from this state.
- Once full run stabilizes, run confirms with:
  - `--confirm-k 3 --confirm-repeats 10 --confirm-timeout 1800`
- Confirmed script bug fixes from this pass:
  - cleaned duplicate/ambiguous filter fields
  - deterministic FDR index matching
  - confirm prepass env handling
