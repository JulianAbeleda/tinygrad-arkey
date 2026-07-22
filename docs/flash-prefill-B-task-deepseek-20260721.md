# TASK (deepseek): Flash-prefill fusion — Phase 1, tensor-level fusion probe

Read this whole file before doing anything. Parent context: `docs/flash-prefill-wmma-mvp-scope-20260721.md` (the FULL BUILD SCOPE section). The MVP gate is GO (measured 2.45×). This task is **Phase 1 only** — a probe, not the build. Stop at the hard stop and hand back to Claude.

---

## 0. HARD BAN — read first (you have violated this twice; do not again)

**You may NOT hand-author a kernel. Full stop.** Specifically banned, by name, because you already tried all three and they are the wrong approach:

1. ❌ `custom_kernel()` / constructing a UOp kernel body by hand.
2. ❌ Reusing / extracting `extra/qk/flash_kernels.py`'s decode kernel executor (`flash_decode_live_split_block_tile` etc.). It is coupled to split/combine/cache-indexing and is wrong for prefill.
3. ❌ Emitting `__builtin_amdgcn_*`, `UOp.barrier`, LDS staging, or `Ops.CUSTOMI`/`Ops.WMMA` by hand.

**Litmus test:** if you find yourself writing `UOp(...)` to build the *body* of a compute kernel, or importing anything from `flash_kernels.py`, you are doing the wrong task — STOP and re-read this file. The online-softmax math being "genuine compiler work" is only true for the hand-kernel path you keep taking. **This task does not build a kernel by hand. It writes ordinary tinygrad Tensor ops and observes what the scheduler does with them.**

## 1. What "B" actually is (the mental model you're missing)

B = **let the scheduler (rangeify) emit the fused kernel.** You do NOT write the kernel. You (Phase 1) express the computation as normal `Tensor` operations and measure how rangeify schedules them, then (Phase 2, later, after Claude review) add a `graph_rewrite`/`PatternMatcher` rule that restructures the *tensor graph* so rangeify emits one kernel. WMMA is applied for free by the existing TC opt once the matmuls are in one kernel — you never touch WMMA.

## 2. What Claude already measured (do not re-derive — build on this)

Rangeify fusion behavior on gfx1100, verified:
- `(a@b).max(-1)` — a matmul-reduce followed by a reduce over a **different** axis — **fuses into 1 kernel.** So `QKᵀ` + rowmax already fuses. Good.
- `softmax(x)` — max-reduce **then** sum-reduce over the **same** axis — **splits into 2–3 kernels** (the serial `max→sum` dependency forces a buffer between them). **This split is the entire problem.** Online-softmax exists to do max+sum in one blocked pass with carried `(m, l, acc)` state; the question Phase 1 answers is exactly what rangeify does when you write that.
- WMMA is declarative (`tc.py:amd_rdna3`) and scheduler-applied (`postrange.py:_apply_tc_opt`) — not your concern.
- SDPA baseline at `T=KV=2048, H=8, Hd=128`: `QKᵀ` 304µs + softmax 362µs(HBM) + PV 1187µs ≈ 1853µs. Fused floor (measured) ≈ 756µs.

## 3. Phase 1 — the probe (this is your whole task)

Config: `B=1, H=8, T=KV=2048, Hd=128`, fp16, causal. Single GPU lane.

**Step 1 — Golden reference.** Compute attention the plain way (matches `model.py:583–598`): `scores=(q@k.transpose(-1,-2))*scale; scores+=causal_mask; p=scores.softmax(-1); out=p@v`. Keep `out` as the correctness golden. All in `Tensor` ops.

**Step 2 — Blocked online-softmax, purely in Tensor ops (NO hand kernel).** Write flash as ordinary tinygrad:
```
m = full((B,H,T,1), -inf); l = zeros((B,H,T,1)); acc = zeros((B,H,T,Hd))
for j in range(0, KV, BLK):                     # Python loop over KV blocks
    kb = k[...,j:j+BLK,:]; vb = v[...,j:j+BLK,:]
    s = (q @ kb.transpose(-1,-2)) * scale       # (B,H,T,BLK) — small
    s = s + causal_mask_block(j)                # additive -inf, NOT bool max
    m_new = maximum(m, s.max(-1,keepdim=True))
    corr = (m - m_new).exp()
    p = (s - m_new).exp()
    l = l*corr + p.sum(-1,keepdim=True)
    acc = acc*corr + (p @ vb)
    m = m_new
out = acc / l
```
Diff `out` vs the golden (fp16 tol ~1e-2). Get it **correct first.** This is all Tensor ops — if you're writing UOps, you're off-task.

**Step 3 — Observe the schedule (the actual point of Phase 1).** Run Step 2 under `DEV=AMD DEBUG=2` and answer, with evidence:
- How many kernels does the blocked version emit? Where do buffers get inserted (i.e. where does rangeify break the graph)?
- Is each block's score `s` kept resident, or materialized to HBM? Does `acc`/`m`/`l` round-trip HBM each block?
- Compare kernel count + total `tm` vs SDPA. (Expect: Python-unrolled blocks → many kernels, likely NOT yet faster — that's fine, it's the baseline that tells us what rangeify does natively.)
- Try 2–3 `BLK` values (e.g. 128, 256, 512) and note the effect.

**Step 4 — Pin the exact missing capability.** The deliverable is a precise answer to: *what would rangeify need to fuse the KV-block loop (with carried `m,l,acc`) into one kernel?* Candidates to investigate in `tinygrad/schedule/rangeify.py`: does it support a REDUCE with a composite/associative accumulator? Does the carried-state loop force a buffer at `pm_add_buffers_local`/`pm_store_ranges`? Is the online-softmax combine expressible as a single associative REDUCE? Point at real lines.

## 4. Deliverable + HARD STOP

Write findings to `docs/flash-prefill-fusion-probe-<date>.md`:
1. The correct blocked-online-softmax tensor code (committed, in `extra/qk/`).
2. Kernel-count + `tm` table: SDPA vs blocked, across BLK values.
3. Where rangeify inserts buffers (real `rangeify.py` lines) and whether the block score stays resident.
4. **The precise missing capability** for single-kernel fusion, with the candidate rangeify mechanism.
5. Your read: is Phase 2 (the `graph_rewrite` rule) a rewrite on top of existing rangeify capability, or does rangeify itself need extending? How big?

**Then STOP and hand to Claude.** Do NOT start Phase 2 (the graph_rewrite). Do NOT build any kernel. If Phase 1 correctness (Step 2) blocks you, report exactly what broke — do NOT fall back to a hand kernel.

## 5. Guardrails

- Everything is `Tensor` ops. Zero hand UOps, zero `flash_kernels.py` imports, zero WMMA/barrier/LDS by hand. (Re-read §0 if tempted.)
- Single GPU lane: `pkill` strays + `rocm-smi` VRAM check before runs; never background benches and report "waiting." Run, wait, read.
- `tm` not wall-clock, warm ≥200 dispatches. `/home/ubuntu/tinygrad-arkey/.venv/bin/python`. Temp in `/home/ubuntu/.claude/jobs/6db6b205/tmp/`.
- Commit on master, no branches, `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, push origin/master.
- No BEAM (hangs gfx1100).

**Your job in one line: write flash as plain tinygrad Tensor ops, get it correct, then tell Claude exactly what the scheduler does with it and what it would take to fuse it — without hand-building a single kernel.**
