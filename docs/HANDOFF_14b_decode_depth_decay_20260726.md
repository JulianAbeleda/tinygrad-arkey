# HANDOFF: 14B decode loses achieved bandwidth with depth (2026-07-26)

Self-contained brief. Hardware: AMD RX 7900 XTX, gfx1100, wave32, ~960 GB/s HBM peak.
Repo: `/home/ubuntu/tinygrad-arkey`. Everything below is measured unless marked as hypothesis.

---

## 1. The defect

Decode throughput, fixed-depth authority (`extra/llm_research/decode/decode_runtime_overhead.py`,
`tinygrad.decode.fixed_depth.v2` — prefills to exactly the stated context before timing, so ctx columns
are real depth). llama is `llama-bench -p 0 -n 128 -d 512,4096 -ngl 99 -fa 1 -r 3`. Same session,
`flock`-serialized, profile `auto`.

| | ctx512 | ctx4096 | margin |
|---|---:|---:|---|
| 8B ours | 113.86 | 102.57 | +10.3% / +6.0% |
| 8B llama | 103.20 ± 1.02 | 96.80 ± 0.20 | |
| 14B ours | 68.39 | **59.41** | +2.7% / **−5.5%** |
| 14B llama | 66.58 ± 0.38 | **62.87 ± 0.05** | |

Converted to achieved HBM bandwidth (weights + KV, each read once per token; 8B 5.03 GB weights /
144 KiB KV per context-token, 40 blocks→14B 9.00 GB / 160 KiB):

| | ctx512 | ctx4096 | trend |
|---|---:|---:|---:|
| 8B ours | 581 | 578 GB/s | −0.6% |
| 8B llama | 527 | 545 | +3.5% |
| **14B ours** | **621** | **575** | **−7.5%** |
| 14B llama | 608 | 610 | +0.3% |

**We are the fastest of all four at 14B ctx512 and the only configuration whose achieved bandwidth FALLS
with depth.** That is the entire defect. It is not a general depth problem (our 8B is flat) and not a
model-size problem.

**Numeric target:** hold ~620 GB/s at ctx4096 → ~64 tok/s, which converts −5.5% into a small win.

## 2. Geometry

Both models: `Hkv=8`, `Hd=128`, `B=1`. The only structural difference is the group ratio.

| | blocks | Hq | Hkv | **G = Hq/Hkv** | threads/workgroup |
|---|---:|---:|---:|---:|---:|
| 8B | 36 | 32 | 8 | **4** | 128 (4 waves) |
| 14B | 40 | 40 | 8 | **5** | 160 (5 waves) |

`extra/llm_research/flash_kernels.py:26`:
```python
G = Hq // Hkv; W = Hd + 2; LANES = 32; WARPS = G; THREADS = LANES * WARPS; TK = 16
```

## 3. Hypotheses EXCLUDED (do not re-run these)

**A. KV re-read per head group — REFUTED by code.**
`LANES = 32` is fixed and `WARPS = G`, so G=5 is five *full* waves — there is no partial-wave second pass
and no lane waste from padding Hq to 32. K/V staging (`flash_kernels.py:74-90`) is **workgroup-cooperative**
(`i = st * THREADS + tid`, spanning all warps) into LDS, followed by a barrier; the dot-product loop reads
`ksh.after(bar)` / `vsh.after(bar)` — LDS, not global. **Each KV byte is fetched from global exactly once
per (kv_head, split, 16-token block) workgroup and broadcast to all G warps, identically for G=4 and G=5.**
Workgroup count is `Hkv * split = 384` for both. The only G-dependent waste is staging gate slack:
`STAGES = ceildiv(TK*Hd, THREADS)` → G4 16×128 = 2048 exact; G5 13×160 = 2080, ~1.5% masked. Not 2x.

**B. `split_size=48` mistuned for G=5 — REFUTED by measurement.**
Swept via a temporary `getenv("DECODE_SPLIT", ...)` override in `_FlashDecodeCandidate.bind` (reverted,
not committed), 14B, `--nmeas 20 --reps 3`:

| split | ctx512 | ctx4096 |
|---:|---:|---:|
| **48** | **68.61** | **59.54** |
| 64 | 67.40 | 56.76 |
| 96 | 66.50 | 58.41 |
| 128 | 64.89 | 57.07 |
| 192 | 62.36 | 56.36 |

48 is already optimal at both depths for G=5; every larger value is worse. The hardcoded constant is not
the defect.

**C. Route switch / dispatch growth — REFUTED by artifact.**
Route is `flash` at every depth for both models, and programs-per-token is constant with depth
(8B 1021, 14B 1133). No route change, no dispatch-count growth.

## 4. Remaining lead (UNSUPPORTED — no evidence yet)

Occupancy or register spilling at 160 threads/workgroup (G=5) vs 128 (G=4). It is consistent with
everything above — a per-wave resource limit would only bite as trip count grows with depth — but nothing
has been measured. Treat it as the next thing to test, not as a finding.

What to capture, per shape, for the flash decode kernel at ctx512 AND ctx4096:
- `private_segment_size` (scratch — nonzero means spilling), `group_segment_size` (LDS), VGPR/SGPR counts
- waves in flight / CU occupancy (`SQ_WAVES`, `SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES`; occupancy ≈
  `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES`)

## 5. Traps — each of these already produced a wrong or empty result today

1. **The harness runs work in isolated subprocesses** (`tinygrad/runtime/process_isolated.py`).
   Monkeypatching `AMDProgram.__init__` in the parent observes **nothing**. A probe doing exactly this
   printed its "installed" banner and zero kernels. Instrument inside the worker.
2. **There is no decode PMC path.** Only `extra/llm_research/prefill/prefill_boltbeam_trace.py` exists (prefill,
   needs sudo). `rocprofv3` is blind to `DEV=AMD`; the native collector is
   `/home/ubuntu/BoltBeam/boltbeam/collectors/tinygrad_native_pmc.py`.
3. **A PMC trace killed mid-run leaves `power_dpm_force_performance_level` stuck in `profile_standard`,
   silently costing ~40% on every later measurement.** Always restore to `auto` and verify before timing.
4. **Do not use marginal bandwidth** (Δbytes / Δwall between two depths). It is arithmetically valid but
   amplifies a 7% total-efficiency shift into an apparent 2.4x gap; it produced a badly misleading
   "266 vs 638 GB/s" here. Use total achieved bandwidth.
5. **Allocation ≠ depth.** `model_e2e_bench.py` decodes from a one-token seed over a growing window, so
   its ctx labels describe KV *allocation*. This inflated the README's 14B "ctx4096" to 68.2 when true
   depth measures 59.5. Any claim of flatness "across max_context" is not flatness across depth.
6. **Every probe needs a positive control that is known to fire**, and counts printed beside verdicts. A
   failure-set diff reported "IDENTICAL" today while comparing two empty files.

## 6. Structural defect — confirmed, independent of §4, fixable now

`tinygrad/llm/decode_routes.py:117-131`:
```python
route_id: str = "decode_flash_live_split_g4_kvboth"   # named g4, tuned at Hq=32
split_size: int = 48                                   # no Hq dependence anywhere
staging: str = "KV_BOTH"
def bind(...):
  if B != 1 or Hq <= 0 or Hd != 128 or Hkv != 8 or Hq % Hkv != 0: return None
```

`Hq % Hkv == 0` admits **both** G=4 and G=5, so 14B silently inherits a kernel named, tuned and evidenced
for G=4. `LiveSplitGeometrySpec` (`extra/llm_research/decode/flash_decode_attention_spec.py:26-40`) derives geometry
from `Tc` and `split_count` only — **no `Hq` dependence at all**.

Meanwhile `extra/llm_research/route_manifest.py:144-163` carries `decode_flash_block_tile_g5_konly`, shape-guarded to
`Hq=40`, `status: promoted_default`, `staging: K_ONLY`. It is **unreachable**:
`flash_decode_attention_route` never consults route policy for the flash candidate (unlike the Q4K/Q6K
candidates, which do), and `decode_routes.py:151` records why — *"K_ONLY assumes the old g5 V layout and
was verified to produce bad logits on 8B"*. So the one route actually searched for G=5 was retired for a
**correctness** bug on the *other* shape, and both shapes were collapsed onto the g4 route.

Two cautions before anyone treats that route as the fix:
- Its promotion artifacts `bench/gp-track/gp4_latest.json` and `docs/gp5-final-report.md` **do not exist**
  in the repo (only `gp3_microgate.json` is present).
- Its headline claim — *"flat across max_context, 69.24 @MAXC=1024 vs 69.04 @MAXC=8192, live ctx ~550"* —
  is flatness across **allocation at fixed shallow depth ~550**, i.e. trap §5.5. It is **not** evidence
  that K_ONLY fixes depth decay.

**Recommended regardless of the perf outcome:** tighten `bind()` so an unsearched shape cannot silently
inherit a tuned kernel — the same posture as `ADMITTED_GRIDS` on the prefill side, which is deliberately
narrower than what `validate()` alone would accept. Either shape-guard the g4 route to `G == 4` and give
G=5 its own searched candidate, or rename/re-evidence the route to cover both honestly.

## 7. Questions for the reviewer

1. Is the occupancy/spill lead (§4) the right next test, or is there a cheaper discriminator for
   "bandwidth falls with depth at 5 waves/workgroup but not 4"?
2. What mechanism would make achieved bandwidth fall with *trip count* while workgroup count, KV traffic
   and route all stay fixed?
3. Is re-searching a G=5 candidate the right fix, or should the g4 kernel be made Hq-aware?
4. K_ONLY halves LDS (4KB vs 8KB) — could that raise occupancy enough to matter at G=5, and is its 8B
   correctness bug shape-specific (i.e. safe to revive for Hq=40 only)?
5. Anything in §3 that I excluded too confidently?

## 8. Reproduce

```bash
export PYTHONPATH=/home/ubuntu/tinygrad-arkey
# ours (one ckpt at a time -- the multi-ckpt run hits a PRE-EXISTING gfx1100 compile bug, see below)
flock /tmp/gpu-bench.lock python3 extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --ckpts 4096 --max-context 4608 \
  --nmeas 40 --reps 5 --out /tmp/o.json
# llama
/home/ubuntu/env/llama.cpp/build/bin/llama-bench -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  -p 0 -n 128 -d 512,4096 -ngl 99 -fa 1 -r 3
```

**Pre-existing blocker:** `--ckpts 128,512,1024,4096 --max-context 4608` on 14B fails with
`CompileError: make_float32(...) = make_float32(...)` — "expression is not assignable", an invalid
vectorized store for gfx1100. Each checkpoint compiles fine **in isolation**. Confirmed pre-existing
(reproduces at `cf0deb072`, before the 2026-07-26 commits, verified in an isolated worktree). This is why
`bench.py --decode` cannot emit a 14B table.
