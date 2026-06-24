# SCOPE — Prefill clock/DPM authority project (gfx1100, tinygrad WMMA vs Tensile vs llama)

Establish reproducible, auditable clock-control lanes so prefill perf claims are not distorted by GPU clock state.
Grounded in the P0 control inventory already run on THIS machine
(`bench/qk-prefill-clock-dpm-authority/supported_controls.json`). **Measure-only: no kernel, route, or default changes.**

Driver: `extra/qk_prefill_clock_dpm_authority.py`. Artifacts: `bench/qk-prefill-clock-dpm-authority/`.

> **⚠ EXECUTED → verdict CORRECTED in `prefill-clock-dpm-authority-result-20260619.md` (READ THAT).** The first cut
> here ("clock not a confound / Tensile 1.76× robust / SOLVED") was WRONG — it used ONE process and a faulty
> `pp_dpm_sclk`-NOMINAL clock reader (reported a fake "2330 pinned"). Multi-run testing shows the comparison is
> confounded by a **per-process GPU boost lottery** (ROCm #6289): the same WMMA path measures **~1437 (SMU stuck,
> idle-gapping) OR ~2674 (boosted @ real sclk 2315, ~87% llama = matches Tensile)**, stable within a process,
> varying across launches, warming into boost over a session. Tensile stable ~2640. Ratio swings 0.997×↔1.83× =
> artifact of WMMA's boost state, not a kernel win; at WMMA's best Tensile gives nothing. Real lever = make WMMA
> sustain boost (dependency-free), not Tensile. Use `--showgpuclocks` from a SEPARATE process; report N≥5 launches.

---

## 1. Purpose
The prefill reconciliation (`prefill-RECONCILIATION-source-of-truth-20260619.md`) showed prefill conclusions were
distorted by GPU clock state:
- tinygrad WMMA concrete-KV prefill is **clock-VOLATILE**: ~1449–2675 tok/s across sessions.
- the Tensile route is **clock-STABLE**: ~2640 tok/s.
- so Tensile is **1.0×** when WMMA happens to land at ~2675, but **1.76–1.83×** when WMMA sits in the typical
  ~1450–1550 regime (4 independent measurements). The earlier "Tensile no advantage / 0.997×" verdict was a
  **high-WMMA-clock outlier** and is retracted.

Core hypothesis (from P0 grounding): the ~1500 regime == sclk DPM **level 1 (~1498 MHz)**; the ~2675 outlier == sclk
ramped to **level 2 (2304 MHz)**. tinygrad WMMA prefill is **bursty** (729 short kernels with inter-kernel gaps) →
AUTO-DPM treats it as light load → holds sclk at level 1. Tensile is denser/clock-stable.

**Goal:** establish controlled, telemetry-verified clock lanes for the three engines (tinygrad WMMA, tinygrad+Tensile,
llama.cpp) so every future prefill number carries an audited clock-lane label — OR prove this consumer RX 7900 XTX
cannot be reliably pinned and mandate telemetry-binned analysis. No new prefill conclusion is issued until that
authority (or its impossibility) is telemetry-proven.

## 2. Non-goals
- **NO kernel changes.**
- **NO default-route changes** (`PREFILL_TENSILE_GEMM` stays research-only/off; concrete-KV default unchanged).
- **NO new prefill conclusion** until clock authority — or its impossibility — is established and telemetry-proven.
- **Do NOT treat any one-off session as canonical** — neither the high-clock (2675) nor a low-clock session is
  authority on its own. Authority requires a telemetry-verified lane reproduced across runs.

## 3. Control surface — P0 inventory  [DONE on this machine]
Recorded in `bench/qk-prefill-clock-dpm-authority/supported_controls.json` (+ `supported_controls_live.json`).
Controls are ASIC/firmware-dependent, so this is what is verified to exist on THIS RX 7900 XTX / gfx1100:

- **`amd-smi`: ABSENT** (not installed). All the brief's `amd-smi metric/set/--perf-determinism` commands are
  unavailable here → rocm-smi + sysfs only. (The amd-smi command set is still listed in §4 as an audit trail of what
  was attempted and why it does not apply.)
- **sysfs `power_dpm_force_performance_level`** accepts: `auto, high, manual, profile_peak, profile_standard,
  profile_min_sclk`. The `perf_determinism` string is **REJECTED** here.
- **DPM levels (discrete only):** sclk `{0:500, 1:~1498 (dynamic mid; idle reads 0), 2:2304}` MHz;
  mclk `{0:96, 1:456, 2:772, 3:1249}` MHz. Set via `manual` + `echo <idx> > pp_dpm_sclk|pp_dpm_mclk`.
- **`pp_od_clk_voltage`: ABSENT** → NO arbitrary OverDrive clock ranges; only the discrete DPM levels above.
- **Determinism:** `rocm-smi --setperfdeterminism <SCLK>` IS available (sets GFXCLK SoftMax) + `--resetperfdeterminism`.
  AMD-documented caveat: may still **downclock under light load** → MUST telemetry-verify hold.
- **Telemetry available:** rocm-smi `--showclocks`/`--showclkfrq`/`--showperflevel`; sysfs `gpu_busy_percent`,
  `mem_busy_percent`, active `pp_dpm_*` (the `*`-marked level), hwmon `power1_average` (µW), `temp1_input` (m°C).
- **Throttle/violation fields: NOT exposed** via rocm-smi/sysfs on this card (those are amd-smi-only) → infer throttle
  from clock-below-cap together with power/temp at ceiling.
- All clock writes require **root** (passwordless sudo confirmed on this box).

Early P0 observations to verify in P2/P3 (not yet authority): `high` stayed ~1515; `profile_peak` gave only ~+4%;
`manual` was previously **erratic** (same config measured 570–1551). These are exactly why P2/P3 require telemetry-hold
before any lane is called "controlled."

## 4. Phase plan

### P0 — Inventory  [DONE — grounds this scope]
rocm-smi/sysfs control discovery: accepted perf-levels, discrete DPM levels, determinism availability, telemetry
fields, presence of `amd-smi`/`pp_od_clk_voltage`. **Artifact: `supported_controls.json` (+ `_live.json`).**
Run: `python extra/qk_prefill_clock_dpm_authority.py inventory`.

### P1 — Telemetry sampler  [IMPLEMENTED — `Sampler` in the driver]
Background sampler (separate thread reading sysfs; ~no perturbation) logging at a fixed interval (default 60 ms)
DURING every benchmark:
`{t, sclk, mclk, fclk, socclk, gpu_busy%, mem_busy%, power_w, temp_c, perf_level, run_id}`.
Per-run summary (busy-only): sclk/mclk **median/min/max**, power max, temp max, % of in-workload samples inside the
intended lane. Must be wall-clock-aligned to the benchmark and must not materially perturb it.
**Artifact: `telemetry.jsonl`** (one JSON object per sample, tagged by `run_id`).

### P2 — Minimal clock-state probes (no full model)  [IMPLEMENTED — `probe`]
Short (~3–5 s) sustained probes, telemetry on, to see whether workload **shape** induces different DPM states:
- **idle** (baseline)
- **tinygrad WMMA-shaped kernel** — the prefill gate/up matmul (12288×4096×512) looped
- **Tensile-shaped kernel** — the extracted Tensile gateup via HCQ launch, looped
- **llama/HIP-shaped kernel** — a HIP fp16 GEMM if practical; else note `llama-bench pp512` as the proxy
Goal: does the **bursty** WMMA path hold a lower sclk than a sustained loop / Tensile / llama? Confirms or refutes the
"DPM downclocks bursty WMMA" hypothesis. **Artifact: `dpm_probe_matrix.json`** (per probe: clock medians + lane + verdict).
Run per lane: `python extra/qk_prefill_clock_dpm_authority.py probe --lane <auto|high|profile_peak|manual_peak|determinism> [--det-mhz N]`.

> Implementation note: the current `probe` covers idle + the WMMA loop. P2 completion adds the Tensile-shaped and
> llama/HIP-shaped probe rows to the same matrix (wire the extracted Tensile gateup HCQ launch + a HIP GEMM proxy).

### P3 — Prefill benchmark matrix (lanes × engines, telemetry on)  [DRIVER STUB — needs harness wiring]
Fixed harness = **`model.forward`** (NOT `model.logits`), interleaved/clock-fair within run, best-of-30, T=512, same
prompt, warmup ≥ 20. Reuse the reconciliation harness shape.

**ROWS (clock lanes)** — each included only if P2/telemetry proves it HOLDS:
- `AUTO`
- `HIGH`
- `PROFILE_PEAK` (if it holds)
- `DETERMINISM` caps via `--setperfdeterminism` — e.g. **1500 / 1900 / 2300 MHz** (if supported & holds)
- `MANUAL` discrete sclk/mclk (sclk idx 2 = 2304 / mclk idx 3 = 1249) (if it holds)

**COLS (engines):**
- symbolic-KV
- concrete-KV WMMA
- concrete-KV + Tensile FFN
- concrete-KV + Tensile FFN + q/o (if available)
- llama.cpp pp512

Each cell logs telemetry and records tok/s (best + median) plus the lane-hold %. llama runs **separately** — it needs
the 4.68 GB VRAM the tinygrad process holds — sequenced under the **same** lane (lane set once, both engines run).
**Artifact: `prefill_clock_matrix.json`.**
Run: `DEV=AMD PREFILL_V2=1 python extra/qk_prefill_clock_dpm_authority.py matrix --lanes auto,high,manual_peak,...`.

> Implementation note: the `matrix` subcommand currently prints the driver contract. P3 completion wraps the
> reconciliation interleaved A/B (`model.forward`, symbolic|concrete|+TensileFFN|+TensileFFN+qo) per lane with the
> `Sampler`, then calls `classify()`.

### P4 — Reconciliation logic (classify every cell)  [classifier IMPLEMENTED — `classify`]
Classify each result as exactly one of:
- **user-realistic AUTO** — lane=auto; telemetry shows natural DPM behavior.
- **controlled clock-authority** — telemetry proves sclk AND mclk stayed in the intended lane (≥95% of in-workload
  samples), no power/temp ceiling hit.
- **clock-confounded** — the two compared engines ran at different measured clocks, or sclk variance > ~5%.
- **unsupported control** — the lane string was accepted but telemetry shows clocks did NOT move/hold.
- **thermal/power-throttled** — power at ~card limit (~330 W) or temp at junction limit while clock sits below cap.

**Decision:** is Tensile's reproduced **1.76–1.83×** robust under a controlled lane where WMMA and Tensile run at the
**same verified sclk**? Specifically: at sclk=2304 held for BOTH, does WMMA catch Tensile (→ ~1.0×), or does Tensile
still win? This is the exact question the reconciliation could not pin (it could not force WMMA to 2304).

### P5 — Policy update  [pending P3/P4 results]
Update `inference-perf-measured-map-20260619.md` + the reconciliation doc so **every future prefill claim MUST report**
the clock provenance. Add this one-line **prefill-claim checklist** template:

```
PREFILL CLAIM — {tok/s}  engine={wmma|wmma+tensile|llama}
  clock_lane        = {auto|high|profile_peak|manual_sclk2|determinism_<MHz>}
  perf_level        = {power_dpm_force_performance_level value}
  sclk MHz          = {median} (min {min} / max {max})
  mclk MHz          = {median} (min {min} / max {max})
  power_W / temp_C  = {max} / {max}
  warmup / interleave = {N iters} / {interleaved? y/n}
  harness_path      = model.forward            # never model.logits
  classification    = {user-realistic | controlled-authority}
```
A claim missing any field is not citable as authority.

## 5. Discovery commands (canonical)
Both `amd-smi` and `/opt/rocm/bin/rocm-smi` were attempted; **amd-smi is absent on this box**, so the amd-smi block is
recorded as an audit trail (what was tried + why N/A), and the rocm-smi/sysfs block is the operative path.

```bash
# --- amd-smi (ATTEMPTED — NOT INSTALLED here; commands recorded for audit/portability) ---
amd-smi metric                       # N/A (amd-smi absent)
amd-smi metric --clock               # N/A
amd-smi metric --power               # N/A
amd-smi metric --temperature         # N/A
amd-smi metric --perf-level          # N/A
amd-smi set --perf-level HIGH        # N/A
amd-smi set --perf-level AUTO        # N/A
amd-smi set --perf-determinism <MHz> # N/A  (use rocm-smi --setperfdeterminism instead)

# --- rocm-smi (OPERATIVE) ---
rocm-smi --showclocks ; rocm-smi --showperflevel ; rocm-smi --showclkfrq
sudo rocm-smi --setsclk 2            # discrete sclk level (if it holds — telemetry-verify)
sudo rocm-smi --setmclk 3            # discrete mclk level
sudo rocm-smi --setperfdeterminism 1900 ; sudo rocm-smi --resetperfdeterminism

# --- sysfs (OPERATIVE; root) ---
DEV=/sys/class/drm/card0/device
cat $DEV/pp_dpm_sclk $DEV/pp_dpm_mclk $DEV/power_dpm_force_performance_level
cat $DEV/gpu_busy_percent $DEV/mem_busy_percent
cat $DEV/hwmon/hwmon*/power1_average $DEV/hwmon/hwmon*/temp1_input
sudo bash -c "echo high         > $DEV/power_dpm_force_performance_level"
sudo bash -c "echo profile_peak > $DEV/power_dpm_force_performance_level"
sudo bash -c "echo manual > $DEV/power_dpm_force_performance_level; echo 2 > $DEV/pp_dpm_sclk; echo 3 > $DEV/pp_dpm_mclk"
sudo bash -c "echo auto         > $DEV/power_dpm_force_performance_level"   # restore
```

## 6. Gates (hard)
- **Do NOT label a benchmark "controlled"** unless telemetry proves sclk AND mclk stayed in the intended lane (≥95% of
  in-workload samples; idle gaps excluded).
- **Do NOT compare tinygrad vs Tensile** unless BOTH ran under the SAME verified clock lane AND telemetry shows no
  thermal/power throttle in either.
- **Do NOT trust `profile_peak`/`high`/`determinism` as authority** unless telemetry proves it MOVED and HELD clocks
  (P0 already flagged: profile_peak ~+4% only, high ~1515, manual erratic 570–1551 — all suspect until verified).
- **If all controls fail to hold:** classify AUTO as **user-realistic only**, drop all fixed-clock-authority claims,
  and switch to **telemetry-binned analysis** — bin every prefill sample by measured sclk and compare engines only
  WITHIN the same sclk bin.

## 7. Deliverables
- `docs/prefill-clock-dpm-authority-scope-20260619.md` — this scope.
- `extra/qk_prefill_clock_dpm_authority.py` — telemetry `Sampler` (P1, done), lane setters + verification + `restore`
  (done), `inventory` (P0, done), `probe` (P2; idle+WMMA done, Tensile/llama-HIP rows TODO), `matrix` (P3 driver stub —
  needs reconciliation-harness wiring), `classify` (P4, done).
- `bench/qk-prefill-clock-dpm-authority/`:
  `supported_controls.json` **[DONE]**, `supported_controls_live.json` **[DONE]**, `telemetry.jsonl` (P1 output),
  `dpm_probe_matrix.json` (P2 output), `prefill_clock_matrix.json` (P3 output), `README.md` **[DONE]**.
- README pointer in `docs/README.md` **[DONE]**.
- **NO route/default changes.**

## 8. Definition of "done"
EITHER:
- **(A) Controlled authority achieved** — at least one lane (likely MANUAL sclk=2304/mclk=1249, or a DETERMINISM cap)
  telemetry-proven to HOLD clocks through the bursty prefill for BOTH WMMA and Tensile, enabling a same-clock
  tinygrad-vs-Tensile-vs-llama comparison. Then re-answer: **is Tensile 1.76× robust at equal sclk?**
OR:
- **(B) Pinning proven impossible** — every lane fails telemetry-hold on this consumer RX 7900 XTX (DPM downclocks the
  bursty load regardless of perf-level). Then the canonical policy becomes: **AUTO = user-realistic**, and ALL prefill
  claims must be **telemetry-binned by measured sclk** (no fixed-clock-authority claims permitted).

Either outcome SETTLES the methodology: no future prefill number is reported without its clock-lane provenance (§5
checklist).

## Open risks
- DPM may downclock the bursty 729-kernel WMMA prefill regardless of perf-level (the AMD "downclock under light load"
  caveat) → outcome **(B)** is the likely realistic deliverable; the telemetry-binned fallback is the durable result.
- Manual DPM was previously ERRATIC (same config 570–1551) → telemetry will show whether it actually holds.
- llama and tinygrad cannot co-reside in VRAM → measured in sequence under the same lane (lane set once, both run).
- Sampler perturbation: sysfs reads are cheap, but verify the sampler interval does not itself depress clocks; cross-
  check a no-sampler control run.
