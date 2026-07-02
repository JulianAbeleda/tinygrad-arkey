# TG-P14.0 Terminal: AMD Process Still Stuck

Verdict: **TG_P14_0_BLOCKED_AMD_PROCESS_STILL_STUCK**

TG-P14 stopped at the required recovery gate. A prior AMD Python worker is still in uninterruptible `D` sleep, so no new AMD
smoke, compiler microgate, model route gate, or promotion test was run.

Observed process:

```text
1840274       1 D          10:27 python3 -
```

The command used for the check was:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg "python3 -|qk_tg|AMDKFD|tinygrad" || true
```

## Decision

Do not run more AMD work into the wedged runtime. The dirty codegen edits remain unverified scratch:

- `tinygrad/codegen/late/devectorizer.py`
- `tinygrad/codegen/__init__.py`
- `extra/qk_tg_p11_reduce_upcast_microgate.py`

No compiler fix was committed. Owned HIP attention remains default.

## Resume Point

After the D-state process clears or the AMD runtime/host is reset, restart TG-P14.0:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg "python3 -|qk_tg|AMDKFD|tinygrad" || true
timeout 45s bash -lc 'DEV=AMD PYTHONPATH=. python3 - <<'"'"'PY'"'"'
from tinygrad import Tensor
print((Tensor([1.0], device="AMD") + 1).realize().numpy())
PY'
```

Only if that smoke passes should TG-P14 continue to P11 fix-off/fix-on, P10 fixed-mode, BoltBeam classification, and the
default-off route regression ladder.
