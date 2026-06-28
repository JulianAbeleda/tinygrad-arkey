# Block tile: refactor → audit → sqtt decision tree — complete result (2026-06-28)

Executed the user's decision tree to find what closes the 12.8× block-tile gap (route-bound W==D 35.0/6.7 =
33.7%/7.1% of owned). All three branches done; the answer is convergent.

## 1. Refactor (split-score token loop) — REFUTED
`DECODE_ATTN_TILE_SPLIT_SCORE`: PASS 1 computes all TK independent dot+reduce scores (LDS buffer), PASS 2 the serial
merges — to pipeline the reduces. Microgate PASS (correct). **W==D 22.2/3.8 — WORSE than base 35.0/6.7** (= the
no-unroll level). The split **broke the unroll's interleaving** + added an LDS round-trip. *Its failure is
informative:* a kernel-level split can't pipeline the reduces without losing the unroll's ILP → the fix must be
**codegen-level** (preserve the unroll AND overlap the reduces across iterations), not a kernel rewrite.

## 2. Audit (flags / bugs, principles-based) — NO CHEAP FIX
- Harness measures the correct **route-bound** fast path (route_bound + token_match true).
- L/S chunk **matches owned** (`l_route = ceildiv(MAXC, 48)`, target_s=48 = owned's `AMDGCN_S`).
- fdot2 + warp-reduce (5 ds_bpermute, one butterfly) are **in-kernel** — not missing lowerings.
- Only real gap: the block tile lacks the sibling's `REG_STORE_DEVEC` PV-vectorization — but the **cycle budget
  disproves it**: that saves ~5 cyc/token; the gap is ~260 cyc/token.

**The cycle budget is the key finding:** gen **281 cyc/token** vs owned **22**. 5 `ds_bpermute` @ ~40-cyc LDS
latency, *fully exposed*, = ~200 cyc ≈ the entire gen per-token cost. **The gap is exposed cross-iteration reduce
latency** — owned overlaps the 5 bpermute across tokens; the generated exposes them. This **resolves the Phase-1
hotloop-diff ambiguity in favor of scheduling.**

## 3. Wire sqtt — CAPTURE WORKS, decode BLOCKED on a vendor lib
- Built `extra/qk_decode_block_tile_sqtt_capture.py` (headless `SQTT=1 PROFILE=1`, no viz server) → `profile.pkl`
  with SQTT packets for all 6 SEs + the block tile in the trace.
- Fixed a real `roc.py` bug (`get_profile(profile, data=data)` → signature drift → `get_profile(data, profile)`).
- **Blocked:** decoding the packets to per-instruction stall needs `librocprof-trace-decoder.so`, not installed.
  Install needs sudo (`sudo ./extra/sqtt/install_rocprof_decoder.py`) or an authorized download — the agent's
  attempt was (correctly) denied by the untrusted-binary guardrail. **User action unblocks it:** run
  `! sudo ./extra/sqtt/install_rocprof_decoder.py`, then `python extra/sqtt/roc.py --kernel flash_block_tiled_...`
  yields the per-instruction stall (expected: `ds_bpermute` dominates).

## Convergent answer
Refactor ❌, flag/audit ❌. The gap is **exposed cross-iteration `ds_bpermute` reduce latency** (cycle budget;
sqtt would confirm at instruction level). The fix is the **codegen modulo scheduler (Layer 3)** scoped in
`decode-codegen-scheduler-capability-scope-v2-references-20260627.md` — overlap iteration N+1's independent
dot+reduce into iteration N's serial-merge shadow, preserving the unroll. **Not** a refactor, a flag, or the
combine. The split-score refutation is positive evidence *for* the codegen approach (kernel-level pipelining loses
the unroll). sqtt instruction-level confirmation is one `sudo` install away.
