# LR-019 — three mechanisms for one flag, and a fourth reader

**Status:** blocker. Recorded before `OptimizationPlan` is threaded into lowering.
**Branch:** `refactor/lowering-architecture` (worktree `/home/ubuntu/lowering-refactor`), at `a47889d50`.
**Asking for:** a decision on which of the four options below to take, or a better one.

---

## The claim, demonstrated

`OptimizationPlan` (`tinygrad/codegen/plan.py`) exists so that a lowering decision is made once, explicitly,
instead of being re-read from ambient state deep inside passes. Its docstring: "The ONE place environment is
read." Threading it into lowering is the last step of LR-051.

It cannot be threaded in as it stands, because the plan and the passes it would govern can disagree about the
same variable inside a single process. This is not hypothetical — run in-tree:

```
getenv (cached at first read): 0     # what a pass sees
getenv (read again, same proc): 0
plan.gate:                     1     # what the plan says
plan.reduction_mode:           warp
DIVERGENT: True
```

The sequence that produces it is ordinary:

```python
first = getenv("WARP_REDUCE_LOWERING", 0)   # some pass reads it early
os.environ["WARP_REDUCE_LOWERING"] = "1"    # anything sets it later
plan = OptimizationPlan.from_env()          # plan reads live env
# pass says off, plan says warp-reduce on, in the same compile
```

## Why: there are three flag mechanisms, not one

| mechanism | where | read semantics | type |
|---|---|---|---|
| `getenv` | `helpers.py:165`, `@functools.cache` | frozen at **first read**, per `(key, default)` | `type(default)` |
| `ContextVar` | `helpers.py:238` (`NOOPT`, `DEBUG`, `SPEC`) | mutable at runtime, `with Context(...)` | int |
| `PLAN_GATES` / `from_env` | `plan.py:145`, `os.environ.get` | **live**, every call | `str` |

`OptimizationPlan` did not unify these. It added a fourth reader with a fourth set of semantics.

Two specific traps, both verified in-tree:

1. **`functools.cache` keys on `(key, default)`.** `getenv("WARP_REDUCE_LOWERING")` (no default → int `0`) and
   `getenv("WARP_REDUCE_LOWERING", "0")` (str) are **separate cache entries**, each frozen at its own first
   read. So "make `from_env` call `getenv`" does *not* by itself make them agree — it only agrees if the plan
   passes the byte-identical default the pass passes. Across 12 `PLAN_GATES` the *values* all match today
   (checked); the *types* do not — the plan stores `'0'`/`'1'` strings, every call site uses int defaults.

2. **`NOOPT` is a ContextVar, not an env read.** `PLAN_GATES` lists `NOOPT` and `from_env` reads
   `os.environ["NOOPT"]`, but nothing in `tinygrad/` calls `getenv("NOOPT")` — `codegen/__init__.py:504` puts
   the *ContextVar* in the `to_program` cache key. A `with Context(NOOPT=1):` block changes lowering and the
   plan cannot see it. `DECODE_FAST_EXP2` likewise has no `getenv` call site in `tinygrad/`.

## Options

**A. `from_env` reads through `getenv`.** Plan and passes share one cache entry, so they agree by construction.
Requires the plan to pass the byte-identical default each call site uses, and a test pinning that correspondence
— otherwise trap 1 silently reintroduces the divergence. Breaks ~6 LR-051 tests that mutate `os.environ` and
expect `from_env` to observe it; arguably those tests assert the wrong contract. Does nothing for `NOOPT`.

**B. Thread the plan everywhere and delete the `getenv` calls.** The actual goal. Large, touches every gated
pass, and cannot be verified by the byte-identical gates — 36 env-gated passes are certified in exactly one
configuration (all-default), so the gates would call the change safe whether or not it is.

**C. Make the plan refuse to be inconsistent.** `plan.validate()` already exists as the choke point before any
pass consumes a plan. Add a check that, for every gate, the plan's value equals what a pass reading it would
see, and raise `PlanRejected` otherwise. Cheap, no pass touched, converts a silent disagreement into a loud
one. Does not unify anything — it makes the seam honest and blocks the failure.

**D. Do not thread the plan in at all; keep it a description.** It is currently inert and useful as such. Say
so, and drop the "make it load-bearing" goal.

## Recommendation offered for review

C now, as the precondition, then B incrementally per-pass with C as the standing invariant — A rejected because
it makes the plan inherit freeze-at-first-read semantics while looking like it reads the environment, which is
the same class of confusion one layer down. But B's verification problem is unsolved: the gates cannot certify
a change to a pass that only executes under a non-default flag, and there is no independent upstream suite to
fall back on (`test_schedule`/`test_assign`/`test_ops` went with the 394-file sweep at `45cfc399c`).

**The question:** is C→B right, and what is the honest way to verify B given that every gate on this branch was
authored by the same process being gated?

## Pseudo-code for C

```python
def _observed_by_a_pass(name, default_str):
  # what getenv would return to a pass RIGHT NOW, without creating a new cache entry
  # (must not call getenv itself — that would populate the very cache we are inspecting)
  ...

def validate(self):
  ...                                   # existing capability/budget checks
  for name, value in self.gates:
    seen = _observed_by_a_pass(name, value)
    if seen is not None and str(seen) != str(value):
      raise PlanRejected(
        f"plan says {name}={value!r} but a pass reading it sees {seen!r}: "
        f"getenv froze this at first read, the plan read os.environ live")
```

Note the comment in the helper: inspecting `getenv.__wrapped__` or `getenv.cache_info()` is required, because
calling `getenv(name, default)` to find out what a pass sees would itself create the cache entry and make the
answer trivially agree. That is the part of C most likely to be wrong.
