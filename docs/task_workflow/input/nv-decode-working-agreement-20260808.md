# NV decode working agreement - stopping the patch-and-repeat cycle

Date: 2026-08-08
Branch: tinygrad `nvidia-bringup-20260731`
Status: **operating agreement. Ratified at the post-wedge pause. Binds every agent and
human working in this repo until amended by a dated revision. Violations are process
failures and must be logged in the ledger, not silently absorbed.**

## 1. Why this exists

The 2026-08-08 wedge investigation surfaced three recurring failure classes that were
each patched repeatedly instead of root-caused:

1. **The default test suite was not green.** Environment failures (no AMD GPU, a
   macOS-only collection import, a CPU-JIT link limitation) drowned out real
   regressions. The graph-cycle bug was fixed once (`37a5a866e`), re-introduced
   silently (`768d04aad`), and survived because no default-suite test caught it.
   It was rediscovered by accident during gate work.
2. **GPU gate failures were not root-caused before retry.** The depth-4096 wedge
   (`Xid 109 CTX SWITCH TIMEOUT`) has recurred across multiple gate runs. Each run
   was re-attempted or worked around without naming the hanging kernel, so the
   wedge kept returning.
3. **Operational failures were endured, not fixed.** Probe runs under a 20 GB
   `systemd-run` cgroup cap were OOM-killed repeatedly; GPU runs were launched
   without the lock; artifacts were written to ad-hoc paths.

This agreement turns those three classes into enforced process.

## 2. Non-goals

- Do not rewrite git history or force-push.
- Do not gate promotion on anything other than measured evidence.
- Do not classify all tests as production-blocking; environment-bound tests get
  skip conditions, not deletions.
- Do not treat this document as permission to refactor unrelated code.
- Do not add process that cannot be checked in under two minutes.

## 3. Rule 1: the default suite is the tripwire

### 3.1 Hard requirement

`DEV=CPU python3 -m pytest test/unit -q` must pass **100%** on the dev box before any
new implementation work starts, after every landed fix, and before every gate run.
The gate run does not start if this is red.

### 3.2 How failures are classified

| Class | Meaning | Allowed handling |
| --- | --- | --- |
| Env-bound | Fails only because hardware/library is absent (no `/dev/kfd`, no macOS libs, CPU JIT cannot link `fmaxf`) | `skipif` with the repo idiom, or `xfail` with a written reason. Must reference the exact env condition. |
| Live regression | Passed at some known commit, fails at HEAD, no env condition | Must be root-caused and fixed. Never xfail a live regression without a dated reason and an owner. |
| Stale assertion | Test encodes an obsolete contract | Amend the test in the same commit as the contract change, with the old/new behavior in the commit body. |

### 3.3 Taxonomy of known env-bound conditions (checked 2026-08-08)

| Condition | Idiom | Applied at |
| --- | --- | --- |
| No AMD GPU | `@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")` | `test_online_softmax_tile.py` (13 hw tests), `test_attention_semantic.py` (`_has_amd()`) |
| CPU freestanding JIT cannot link `fmaxf` | `@pytest.mark.xfail(reason="...fmaxf...CPU freestanding JIT cannot link", strict=False)` | `test_attention_semantic.py`, `test_nested_composite_reduce.py` |
| macOS-only import (`libSystem.dylib`) | skip when `sys.platform != "darwin"` | `test_target_capability_facts.py` (open item, see 3.4) |

### 3.4 Open baseline items at ratification

| Item | State | Owner | Done when |
| --- | --- | --- | --- |
| `test_target_capability_facts.py` collection error on Linux | Open | agent + reviewer | skip condition lands; full suite green |
| Full-suite green run captured in a record | Open | agent | `/tmp/current_failures.txt` empty, record committed |

## 4. Rule 2: every fix lands with its regression test

### 4.1 Definition of done for a fix

1. Write a test that fails at the broken commit (or name the existing test that does).
2. Apply the fix.
3. Show the test passing in the same run as the pre-fix failure, or cite both runs.
4. Commit the test and the fix together.
5. The test must run in the **default** suite (`test/unit`, `DEV=CPU`), not only in
   an opt-in environment, unless the bug is genuinely env-bound.

### 4.2 Regression test must survive

If a fix is later reverted, the regression test must go red in the default suite.
The graph-cycle case is the template: `test_bounded_semantic_admission_handles_causal_mask_source`
and the GQA/additive variants are default-suite tests that crash with
`cycle detected while indexing` at HEAD-without-the-fix, and pass with it.

## 5. Rule 3: gate discipline

### 5.1 A gate failure is a bug report, not a retry request

Before re-running any failed gate:

1. Capture the artifact (`--artifact` path) and the full stderr.
2. Root-cause to a named kernel or a named failure path:
   - GPU sync timeout: re-run the failing arm with `DISPATCH_TRACE=1`; if the dump
     reports no traced dispatch in flight, the hang is inside a CUDA-graph replay and
     the next step is a graph-disabled run (or `JIT_NO_GRAPH_KERNEL_PREFIXES`) to name
     the kernel.
   - OOM kill: confirm cgroup cap and RSS; fix the harness cap, not the product.
3. Record the named cause in the ledger with a `WEDGE` or `FAIL` row.
4. Only then re-run, with the fix or with the cause documented as out-of-scope.

### 5.2 Promotion protocol

- A promotion record is edited (`promoted_targets`) only after a full gate PASS,
  committed with the gate record.
- No depth may be dropped or skipped to make a gate pass.
- `record` arm runs must match the checked-in policy file at gate time.
- The GPU lock (`flock -w 600 /tmp/gpu-bench.lock`) is required for every GPU command.
  The lock is held for the whole gate, including teardown.

### 5.3 Known wedge state at ratification (2026-08-08)

| Depth | State | Evidence |
| --- | --- | --- |
| 512 | PASS | 183.014 tok/s closed; pins 3/3; sha `227ad3ce...` |
| 2048 | PASS | 172.276 tok/s closed; pins 3/3; sha `aca13ac6...` |
| 4096 | WEDGE | `Xid 109 CTX SWITCH TIMEOUT`; wait timeout on graph replay; 34.1 GB free (not OOM); DISPATCH_TRACE: no traced dispatch in flight -> hang inside CUDA-graph replay |

## 6. Rule 4: operational hygiene

### 6.1 GPU and compute

- Every GPU command runs under the lock, with a timeout (`timeout 1200` style) and a
  watchdog where a hang is plausible.
- The default device on the dev box is NV; CPU work explicitly sets `DEV=CPU`.
- No two GPU processes at once, including teardown and allocator cache flushes.

### 6.2 Memory

- Host-RAM caps for model runs must be sized from measured RSS, not guesses. The
  Qwen3-8B-Q4_K_M harness needs more than 20 GB host RAM; the cgroup cap that
  killed runs is a harness bug, tracked separately.
- OOM kills are logged with the cgroup id and RSS at kill time.

### 6.3 Artifacts and identity

- Gate artifacts go to `/tmp/<name>-<date>.json` and are referenced in the record doc.
- Every gate result commit cites: HEAD sha, arm modes, GPU state, artifact path.
- Long-running sessions log to a file (`> /tmp/... 2>&1`) so the traceback survives.

## 7. Rule 5: the ledger is the single source of truth

`docs/task_workflow/input/nv-decode-parity-campaign-reconciled-ledger-20260805.md`
is authoritative for:

- Booked recoveries (never re-book a measured delta).
- The strict accounted remainder (the number we are reducing).
- Promotion rows (a gate PASS books a row; a WEDGE books a `WEDGE` row).

Every gate run updates the ledger in the same commit as the gate record. If a row
cannot be booked, the reason is written, not omitted.

## 8. Rule 6: weekly drift check

One review pass per week (or after every wedge):

1. `git status` - nothing uncommitted that is not part of an active gate.
2. Full-suite run - green, or only documented env-bound skips/xfails.
3. Ledger diff - every claim has evidence, every evidence has a claim.
4. Open items from 3.4 and 5.3 - status, owner, next action.
5. Update this document's date if amended.

The drift check output is committed as a short record so the check is auditable.

## 9. Consequence policy

- A fix without its regression test is not committed.
- A gate failure without a root-cause row is not re-run.
- A red default suite blocks new implementation work.
- Process violations are logged in the ledger with a dated row, not argued in chat.

## 10. Immediate ratification checklist

- [ ] Fix the 4096 graph-replay wedge and book the `WEDGE` row
- [ ] Fix `test_target_capability_facts.py` Linux collection
- [ ] Run full unit suite to green and commit the record
- [ ] Fix the harness host-RAM cap (measured RSS, not 20 GB)
- [ ] Commit this agreement
- [ ] First weekly drift check scheduled
