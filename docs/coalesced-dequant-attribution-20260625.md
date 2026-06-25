# Coalesced-dequant attribution memo (P0.2, 2026-06-25)

## Verdict

`P0_2_COALESCED_DEQUANT_ATTRIBUTION_PASS`

The exact packed scheduler arm has no duplicate packed-word loads in the rendered kernel. The scheduler-GEMV loss should stay attributed to thread-map / work decomposition, not redundant packed-word reloads.

## Scope

This memo checks the P0.2 arm from `docs/layout-codegen-full-scope-20260625.md`:

```text
Q4K_GEMV_SCHEDULER=2
extra.qk_q4k_scheduler_gemv.q4k_scheduler_matvec
```

This is the packed-word scheduler arm, not the later wordlane experiment.

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python - <<'PY'
import re
from collections import Counter
from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops
from extra.qk_q4k_scheduler_gemv import q4k_scheduler_matvec
rows, k = 12288, 4096
nb = k//256
words = Tensor.empty(rows, nb*36, dtype=dtypes.uint32)
x = Tensor.empty(k, dtype=dtypes.float16)
out = q4k_scheduler_matvec(words, x, rows, k)
src='\n'.join(next((u.arg for u in c.src[0].toposort() if u.op is Ops.SOURCE), '') for c in compile_linear(out.schedule_linear()).src if c.src[0].op is Ops.PROGRAM)
load_lines=[ln.strip() for ln in src.splitlines() if 'unsigned int val' in ln and 'data1_' in ln]
exprs=[]
for ln in load_lines:
  m=re.search(r'\*\(data1_[^+)]*(?:\+([^)]*?))?\)', ln)
  if m: exprs.append((m.group(1) or '0').strip())
print('program_chars', len(src), 'uint_load_lines', len(load_lines), 'unique_exprs', len(set(exprs)))
for expr,c in Counter(exprs).most_common(20):
  if c>1: print('DUP', c, expr)
PY
```

## Result

```text
program_chars 17395 uint_load_lines 24 unique_exprs 24
```

No `DUP` lines were printed.

Representative emitted loads:

```c
unsigned int val0 = (*(data1_7077888+(alu5+1)));
unsigned int val1 = (*(data1_7077888+(alu5+2)));
unsigned int val2 = (*(data1_7077888+(alu5+3)));
unsigned int val11 = (*(data1_7077888+alu5));
unsigned int val12 = (*(data1_7077888+(alu98+4)));
unsigned int val13 = (*(data1_7077888+(alu98+12)));
unsigned int val14 = (*(data1_7077888+(alu98+20)));
unsigned int val15 = (*(data1_7077888+(alu98+28)));
```

## Interpretation

The packed scheduler arm is already CSE/load-deduped at the packed-word level:

```text
24 rendered uint32 load lines
24 unique data1 address expressions
0 duplicate packed-word load addresses
```

Therefore, the remaining gap is not explained by repeated loads of the same packed word.

The loss remains attributed to the physical thread-map / work decomposition wall:

- the scheduler arm can express packed-word dequant as Tensor ops;
- the scheduler arm does not express the owned warp kernel's per-row workgroup map;
- the owned kernel uses `lane = block_group*8 + lane4`, with `lane4` reading adjacent packed words and `block_group` splitting K across the wave;
- P2 must target LaneMap-aware `add_gpudims`, not load-dedup.

## Gate

Gate passed:

```text
Rendered kernel shows no duplicate packed-word loads for Q4K_GEMV_SCHEDULER=2.
```

## Kill condition

Kill did not trigger. The exact P0.2 arm did not contradict the CSE/load-dedup assumption.
