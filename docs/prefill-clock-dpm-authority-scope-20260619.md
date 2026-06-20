# SCOPE — Prefill clock/DPM authority project (gfx1100, tinygrad WMMA vs Tensile vs llama)

Establish reproducible, auditable clock-control lanes so prefill perf claims are not distorted by GPU clock state.
Grounded in the P0 control inventory already run on THIS machine (`bench/qk-prefill-clock-dpm-authority/
supported_controls.json`).

## 1. Purpose
The recent prefill reconciliation (`prefill-RECONCILIATION-source-of-truth-20260619.md`) showed tinygrad WMMA
concrete-KV prefill is **clock-volatile (1449-2675 tok/s)** while Tensile is **clock-stable (~2640)**, so the
Tensile-vs-WMMA ratio (1.0x .. 1.83x) is decided by which sclk DPM level the bursty WMMA workload lands on
(level 1 ~1498 vs level 2 ~2304). Establish **controlled, telemetry-verified clock lanes** for the three engines
(tinygrad WMMA, tinygrad+Tensile, llama.cpp) so every future prefill number is labeled by an audited clock lane —
or, if no control holds, prove this consumer RDNA3 card cannot be pinned and mandate telemetry-binned analysis.

## 2. Non-goals
- NO kernel changes. NO default-route changes (`PREFILL_TENSILE_GEMM` stays research-only/off).
- NO new prefill conclusion until clock authority (or its impossibility) is established + telemetry-proven.
- Do NOT treat any one-off high-clock (e.g. the 2675 session) or low-clock session as canonical.
- Do NOT use `profile_peak` (or any lane) as authority unless telemetry proves it actually changes AND HOLDS clocks
  through the bursty prefill.

## 3. Control surface (P0, ALREADY INVENTORIED on this machine)
- **amd-smi: ABSENT** -> rocm-smi + sysfs only.
- **sysfs `power_dpm_force_performance_level`** accepts: `auto, high, manual, profile_peak, profile_standard,
  profile_min_sclk`. `perf_determinism` string REJECTED.
- **DPM levels:** sclk {0:500, 1:~1498(dynamic mid), 2:2304} MHz; mclk {0:96, 1:456, 2:772, 3:1249} MHz. Set via
  `manual` + `echo <idx> > pp_dpm_sclk|pp_dpm_mclk`.
- **`pp_od_clk_voltage` ABSENT** -> NO arbitrary OverDrive ranges; only the discrete DPM levels above.
- **Determinism: `rocm-smi --setperfdeterminism <SCLK>` IS available** (GFXCLK SoftMax) + `--resetperfdeterminism`.
  AMD-documented caveat: may still downclock under light load -> MUST telemetry-verify hold.
- **Telemetry:** rocm-smi `--showclocks`; sysfs `gpu_busy_percent`, `mem_busy_percent`, hwmon
  `power1_average`(uW), `temp1_input`(mC). **NO throttle/violation fields** exposed (amd-smi-only; absent here) ->
  infer throttle from clock-vs-cap + power/temp ceilings.
- All clock writes need root (passwordless sudo confirmed working on this box).

## 4. Phase plan
### P0 — Inventory  [DONE, grounding this scope]
rocm-smi/sysfs control discovery; which perf-levels accepted; DPM levels; determinism availability; telemetry
fields. Artifact: `supported_controls.json` (written).

### P1 — Telemetry sampler
A background sampler (thread or subprocess) logging at fixed interval (e.g. 50-100ms) DURING every benchmark:
`{t, sclk_mhz, mclk_mhz, fclk, socclk, gpu_busy%, mem_busy%, power_w, temp_c, perf_level}` from rocm-smi
`--showclocks` + sysfs hwmon. Must not perturb the benchmark materially (sample in a separate process; record
wall-clock-aligned). Per-run summary: sclk/mclk median/min/max, power max, temp max, % of samples in the intended
lane. Artifact: `telemetry.jsonl` (one line per sample, tagged by run-id).

### P2 — Minimal clock-state probes (no full model)
Short (~3-5s) sustained probes, telemetry on, to see whether workload SHAPE induces different DPM states:
- idle (baseline)
- tinygrad WMMA-shaped kernel (the prefill gate/up matmul 12288x4096x512, looped)
- Tensile-shaped kernel (the extracted Tensile gateup via HCQ launch, looped)
- llama/HIP-shaped kernel (a HIP fp16 GEMM if practical; else note llama-bench pp512 as the proxy)
Goal: does the BURSTY WMMA path hold a lower sclk than a SUSTAINED loop / Tensile / llama? (Confirms or refutes the
"DPM downclocks bursty WMMA" hypothesis.) Artifact: `dpm_probe_matrix.json` (per probe: clock medians + the lane).

### P3 — Prefill benchmark matrix (lanes x engines, telemetry on)
Fixed harness = `model.forward` (NOT model.logits), interleaved/clock-fair within run, best-of-30, T=512, same
prompt, warmup>=20. Reuse the reconciliation harness shape.
ROWS (clock lanes): AUTO | HIGH | PROFILE_PEAK | MANUAL(sclk2/mclk3) | DETERMINISM(1500/1900/2300 via
--setperfdeterminism) -- each only if P2/telemetry proves it HOLDS.
COLS (engines): symbolic-KV | concrete-KV WMMA | concrete-KV+Tensile-FFN | concrete-KV+Tensile-FFN+q/o | llama.cpp.
(Note: llama runs separately -- it needs the 4.68GB VRAM the tinygrad process holds; sequence them, same lane.)
Each cell logs telemetry; record tok/s (best+median) + the lane-hold %. Artifact: `prefill_clock_matrix.json`.

### P4 — Reconciliation logic (classify every cell)
Classify each result:
- **user-realistic AUTO** (lane=auto, telemetry shows natural DPM behavior)
- **controlled clock-authority** (telemetry proves sclk AND mclk stayed in the intended lane >=95% of samples, no
  power/temp ceiling hit)
- **clock-confounded** (the two compared engines ran at different measured clocks, or sclk variance > ~5%)
- **unsupported control** (the lane string accepted but telemetry shows clocks did NOT move/hold)
- **thermal/power-throttled** (power at ~card limit ~330W or temp at junction limit while clock below cap)
DECISION: is Tensile's 1.76-1.83x ROBUST under a controlled lane where WMMA and Tensile run at the SAME verified
sclk? (i.e., at sclk=2304 for both, does WMMA catch Tensile -> 1.0x, or does Tensile still win?) -- this is the
question the reconciliation could not pin (couldn't force WMMA to 2304).

### P5 — Policy update
Update `inference-perf-measured-map-20260619.md` + the reconciliation doc: EVERY future prefill claim MUST report
{clock_lane, perf_level, sclk median/min/max, mclk median/min/max, power_max, temp_max, warmup, interleaving,
harness_path (forward not logits), classification (user-realistic vs controlled-authority)}. Add a one-line
"prefill-claim checklist" template.

## 5. Discovery commands (canonical; amd-smi absent -> rocm-smi/sysfs)
```
# telemetry
rocm-smi --showclocks ; rocm-smi --showperflevel ; rocm-smi --showclkfrq
cat /sys/class/drm/card0/device/{pp_dpm_sclk,pp_dpm_mclk,power_dpm_force_performance_level,gpu_busy_percent,mem_busy_percent}
cat /sys/class/drm/card0/device/hwmon/hwmon4/{power1_average,temp1_input}
# perf-level lanes (root)
sudo bash -c 'echo high > .../power_dpm_force_performance_level'
sudo bash -c 'echo profile_peak > .../power_dpm_force_performance_level'
# MANUAL discrete levels (root)
sudo bash -c 'echo manual > .../power_dpm_force_performance_level; echo 2 > .../pp_dpm_sclk; echo 3 > .../pp_dpm_mclk'
# DETERMINISM (GFXCLK SoftMax)
sudo rocm-smi --setperfdeterminism 1900 ; sudo rocm-smi --resetperfdeterminism
# restore
sudo bash -c 'echo auto > .../power_dpm_force_performance_level'
# (amd-smi NOT present on this box; the amd-smi metric/set commands in the brief are unavailable here)
```

## 6. Gates (hard)
- Do NOT label a benchmark "controlled" unless telemetry proves sclk AND mclk stayed in the intended lane (>=95% of
  in-workload samples; exclude idle gaps).
- Do NOT compare tinygrad vs Tensile unless BOTH ran under the SAME verified clock lane AND telemetry shows no
  thermal/power throttle in either.
- Do NOT trust `profile_peak`/`high`/`determinism` as authority unless telemetry proves it MOVED and HELD clocks
  (P0 already showed profile_peak gave only +4% and HIGH stayed ~1515 -> suspect; manual was ERRATIC 570-1551).
- If ALL controls fail to hold: classify AUTO as **user-realistic only**, drop all fixed-clock-authority claims,
  and switch to **telemetry-binned analysis** (bin every prefill sample by measured sclk; compare engines within
  the same sclk bin).

## 7. Deliverables
- `docs/prefill-clock-dpm-authority-scope-20260619.md` (this).
- `extra/qk_prefill_clock_dpm_authority.py` (telemetry sampler + lane setters w/ verification + probe/matrix drivers
  + classifier). [skeleton committed; phases runnable]
- `bench/qk-prefill-clock-dpm-authority/` artifacts: `supported_controls.json` [DONE], `telemetry.jsonl`,
  `dpm_probe_matrix.json`, `prefill_clock_matrix.json` (produced by running P1-P3).
- README pointer.
- NO route/default changes.

## 8. Definition of "done"
EITHER:
- **(A) Controlled authority achieved:** at least one lane (likely MANUAL sclk=2304/mclk=1249 or DETERMINISM)
  telemetry-proven to HOLD sclk through the bursty prefill for BOTH WMMA and Tensile, enabling a same-clock
  tinygrad-vs-Tensile-vs-llama comparison. Then re-answer "is Tensile 1.76x robust at equal sclk?".
OR:
- **(B) Pinning proven impossible:** every lane fails telemetry-hold on this consumer RX 7900 XTX (DPM downclocks
  the bursty load regardless). Then the canonical policy becomes: AUTO = user-realistic, and ALL prefill claims
  must be **telemetry-binned by measured sclk** (no fixed-clock authority claims allowed).
Either outcome SETTLES the methodology so no future prefill number is reported without its clock-lane provenance.

## Open risks
- DPM may downclock the bursty 729-kernel WMMA prefill regardless of perf-level (the AMD "downclock under light
  load" caveat) -> outcome (B) likely; the telemetry-binned fallback is the realistic deliverable.
- Manual DPM was previously ERRATIC (same config 570-1551) -> telemetry will show whether it actually holds.
- llama and tinygrad cannot co-reside in VRAM -> measured in sequence under the same lane (lane set once, both run).
