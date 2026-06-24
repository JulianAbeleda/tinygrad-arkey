# Harness Evaluator-Contract Audit + Application — Result

Date: 2026-06-21

Authority: `structure/Development/performance-primitive-research-principles.md` §§ "Harnesses Are Performance
Primitives Too" + "Machine Search Is Generate, Evaluate, Prune, And Remember" (grounded in **MLPerf Inference**,
**SPEC RG** reproducible-evaluation methodology, **Ansor/TVM**, **Triton**). The new rule:

> A performance claim is only valid when the evaluator captures **workload, comparator, correctness/quality, timing
> authority, environment, repeats/noise, candidate metadata, and promotion policy** — and a valid artifact records 13
> fields (workload+context · candidate id+class · comparator id+why-winner · command+env · git commit+dirty ·
> hardware+clock state · warmup/compile · repeats/median/spread/band · correctness/quality gate · local-vs-W==D
> authority · pass/fail threshold · verdict+stop reason · ledger/refutation links).

This task **(1)** exhaustively audited the **live lifecycle harness set** against that contract, **(2)** found the
systemic gaps + two real distortion risks, and **(3)** applied the harness logic: a centralized contract helper, an
evaluator that auto-flags any non-conforming child artifact, a conforming reference harness, and a fix to the one
correctness-distorting harness.

## Scope of the search (why this set)

There are ~290 perf-touching scripts under `extra/qk_*.py`; the contract only **matters** for the **live lifecycle
harnesses** — those whose artifacts drive a promotion/refutation verdict (the `ab_script` runners in
`candidates.json`, the W==D authority, and the search infra). One-off probes/scopes that are superseded or never
re-run are out of scope (they inherit the contract if/when re-registered). The live set was derived mechanically from
`candidates.json` rungs + `binding_templates.json` + `refutations.json` evidence + the `_ab.py` suffix.

## Phase 1 — exhaustive audit of the live set

Per-script audit (objective grep matrix + three parallel deep reads + direct `contract_audit()` on emitted artifacts):

| harness (role) | comparator | timing authority | clock-pin | git/dirty | repro band | verdict | ledger link | conformance |
|---|---|---|---|---|---|---|---|---|
| **`qk_decode_eval.py`** (evaluator) | gqa_coop_vec (prose) | clean W==D = promotion; local/PROFILE = diagnostic | mode recorded | **yes** | **yes (band)** | yes+expected-match | yes | **CONFORMS (12/13)** — only field 3 prose-only |
| `qk_decode_runtime_overhead.py` (W==D authority) | route-agnostic | its own `tok_s_W` (the W==D number) | auto (silent) | no | median only | yes | no | PARTIAL — thin; **wrapped** by decode_eval (band+provenance added there) |
| `qk_lifecycle_search_loop.py` (search loop) | n/a | delegates to decode_eval W==D | n/a | n/a | n/a | n/a | dedups vs refutations | **CONFORMS** (8/8 loop elements; refute/promote propose-only by design) |
| `qk_candidate_template_gen.py` (generator) | n/a | none (specs only) | n/a | n/a | n/a | n/a | `refutation_for` links | **CONFORMS** (8/8; template space shallow) |
| `qk_fused_flash_concrete_gate_ab.py` | gqa_coop_vec (+why) | local throughput proxy (labeled) | yes | **yes** | **yes (0.51%)** | yes+stop | **yes** | **CONFORMS (13/13)** — upgraded this task |
| `qk_matmul_pv_diagnostic_ab.py` | gqa_coop_vec | local throughput | yes | no | median only | yes | no | WEAK (6/13) |
| `qk_fused_softmax_v_tail_ab.py` | gqa_coop_vec | local throughput | yes | yes | median only | yes | no | WEAK (7/13) |
| `qk_north_star_flash_attn_tile_ab.py` | gqa_coop_vec | local throughput proxy (labeled, no W==D) | yes | no | band (comparator only) | gate bool | binding id | WEAK |
| `qk_llama_flash_attn_tile_oracle_ab.py` | gqa_coop_vec | mixed: rocprofv3 GPU-time + ProfileGraphEvent | coop side | no | none | gate | source paths | WEAK → **provenance fixed** this task |
| `qk_decode_vector_flash_tile_ab.py` | gqa_coop_vec | local throughput | yes | no | none | yes | no | PARTIAL (closed lane) |
| `qk_decode_warp_flash_tile_ab.py` | gqa_coop_vec | local throughput | yes | no | none | gate bool | no | PARTIAL (closed lane) |
| `qk_decode_fused_flash_tile_ab.py` | gqa_coop_vec | wall-clock throughput | yes | no | none | yes | no | PARTIAL (closed lane) |
| `qk_decode_fused_lds_tile_ab.py` | gqa_coop_vec | wall-clock throughput | yes | no | none | yes | prose "result doc" | PARTIAL (closed lane) |
| `qk_decode_ffn_activation_producer_fusion_ab.py` | un-fused silu*up (correct base) | isolated throughput | yes | no | none | **no verdict** | no | PARTIAL (closed lane) |
| `qk_gateup_sched_ab.py` | old prefill schedule (correct base) | in-model tok/s wall-clock | **NO (claims "clock-controlled" but unpinned)** | no | none | **no** | no | **WEAK** (no artifact; stdout only) |

### Systemic gaps (across the standalone harnesses)
1. **repeats/median/spread band (field 8)** — the single most-missing field. Nearly every standalone harness emits a
   **bare median**, which cannot distinguish a real 1.05× from host jitter. (decode_eval adds a band for W==D; local
   A/B harnesses did not.)
2. **git commit + dirty (field 5)** — absent in most standalone harnesses → results un-anchored to a tree state.
3. **comparator id + WHY it is the current winner (field 3)** — carried as **prose only** everywhere, including the
   evaluator. The comparator id is present (`gqa_coop_vec`); the machine-readable "why winner" was not.
4. **ledger/refutation link (field 13)** and **hardware/command in-artifact (fields 4,6)** — absent in most.
5. **timing-authority honesty (field 10)** — most local harnesses are throughput **proxies**, not GPU-time/W==D; the
   better ones label this, the weaker ones do not.

### Two real distortion risks (not just missing metadata)
- **`qk_llama_flash_attn_tile_oracle_ab.py` (LIVE `PASS_ORACLE` candidate):** `_llama_pertoken_table()` **ignores the
  trace file it reads and returns HARDCODED constants** (tile 266/342/881, combine 115/116/152); so **ctx512/4096 are
  derived from fixed constants** while labeled merely "derived", and `splits[].err` is synthesized `0.0`. Only
  **ctx1024 is freshly measured**. The headline (~5.7×@1024) rests on measured data, but the 512/4096 figures are
  constant-derived. **FIXED** (below).
- **`qk_gateup_sched_ab.py`:** emits **no artifact** (stdout only), is **not clock-pinned** despite a "clock-controlled"
  docstring claim, and prints a bare speedup with **no verdict/threshold**. A prefill-schedule A/B with an unanchored
  number. **Flagged** (closed/superseded; not retrofitted — see Deferred).

### Machine-search-loop conformance
The loop (`qk_lifecycle_search_loop.py`) + generator (`qk_candidate_template_gen.py`) implement all 8 closed-loop
elements (template space → generated specs → closed-lane pruning **before** benchmarking → evaluator binding →
machine-readable artifact → lifecycle verdict → ledger/refutation memory → local-A/B→W==D promotion path), with
refutation/promotion correctly **propose-only / owner-gated**. The shallow point: the "template space" is mostly fixed
enumerations binding to pre-existing candidates, not a deep generative grid (Ansor/Triton lesson — a future lever, not
a defect).

## Phase 2 — application (the harness logic applied)

1. **Centralized the contract** — new `extra/qk_harness_contract.py` (import-safe, no GPU): `provenance()`
   (command+env+git+hardware+perf-state), `repro_band(samples)` (the missing spread/noise band), `contract_audit(art)`
   (which of the 13 fields are present/missing → `CONFORMS|PARTIAL|WEAK`), and `stamp(art, comparator_id,
   comparator_why, timing_authority, ledger_links)` (additive envelope). Self-test: thin artifact → WEAK; stamped →
   CONFORMS 13/13. This is the **one authority point** future harnesses call instead of re-deriving (and forgetting)
   provenance.
2. **Enforced it at the evaluator** — `qk_decode_eval.run_ab_script` now runs `contract_audit` on **every consumed
   child artifact** and surfaces a `child_artifact_contract` block + a `HARNESS-CONTRACT: …` note when a child is
   non-conforming. So a thin `ab_script` harness is now **flagged centrally** by the lifecycle, not silently trusted.
   Additive only — re-running `fused_flash_concrete_gate` returns the **same** `FAIL_LOCAL_AB` verdict (match=True).
3. **Upgraded the current harness to the conforming reference** — `qk_fused_flash_concrete_gate_ab.py` now records a
   5-sample `repro_band` (candidate ctx1024 spread **0.51%** ≪ the 3.5% gap to the 1.05× gate → the 0.967× FAIL is
   robustly outside noise), top-level `repeats`/`warmups`/`pass_fail_threshold`/`stop_reason`, and `stamp()`s the full
   envelope (comparator-why, timing authority, ledger links). `contract_audit` → **CONFORMS 13/13**.
4. **Fixed the llama-oracle distortion** — disclosed the hardcoded per-token basis (`data_provenance:
   derived_from_constant_pertoken_table` per ctx + a top-level `data_provenance_caveat`), relabeled the "source"
   string to "DERIVED from HARDCODED … (NOT re-measured)", and `stamp()`ed the contract. No numeric change (so
   decode_eval still reads it); the artifact is now **honest** that only ctx1024 is measured.

### Deferred (documented, not retrofitted)
The **closed-lane** harnesses (`vector`/`warp`/`fused_flash_tile`/`fused_lds_tile`/`ffn_activation`/`gateup_sched`) are
superseded negatives that will not re-run to drive a new verdict; they **inherit the contract via the evaluator** if
ever re-registered (the child-contract flag will mark them WEAK). Retrofitting them is low value vs. risk. The active
diagnostic harnesses (`matmul_pv_diagnostic`, `fused_softmax_v_tail`, `north_star`) remain WEAK as standalone
artifacts but are **flagged by the evaluator** on every run and could be stamped with a one-line `stamp()` call when
next touched. The evaluator's own field-3 (comparator-why) prose-only gap is the highest-value remaining tightening
(add a structured `comparator` block to `decode_eval_run_v1`).

## Acceptance / verification
- `extra/qk_harness_contract.py` self-test → `SELFTEST_PASS` (thin WEAK, stamped CONFORMS 13/13).
- `qk_fused_flash_concrete_gate_ab.py` re-run → artifact `CONFORMS 13/13`; verdict unchanged
  `FUSED_FLASH_CONCRETE_GATE_FAIL_LOCAL_AB` (0.967×, band 0.51%).
- `decode_eval --candidate fused_flash_concrete_gate` → `FAIL_LOCAL_AB` (match=True), now records
  `child_artifact_contract: CONFORMS`.
- `contract_audit` on the existing thin artifacts correctly returns WEAK (matmul_pv 6/13, fused_softmax 7/13,
  north_star 1/13, llama_oracle-old 2/13) → enforcement demonstrably discriminates.
- `git diff tinygrad/` empty; policy guard PASS.

## Boundary
Audit + harness-infrastructure only. No `tinygrad/` change, no model route/default change, no decode verdict changed
(the `fused_flash_concrete_gate` `FAIL_LOCAL_AB` and `REST_DECODE` stance are unchanged). The llama oracle stays
non-promotable; its fix is provenance-honesty, not a number change.
