# NV R-residual PDL/concurrency adjudication scope (2026-08-22)

Status: measurement gate only; no production code authorization.

1. Row and edge: the production-conditioned residual `R` across `q4k_warp_coop
   Q` (3.107 us/call), `q4k_g3 Q` (0.953), `O` (1.486), K/V routes
   (0.97-1.41), and flash score (2.614). Edge: DRAM-bound GEMV producer to its
   support consumer (partial -> completion, score -> combine, logits ->
   argmax).
2. Dominant term: `R = P - C`, unnamed, ~404 us/token aggregate.
3. Code paths: none changed. Add wait-exit timestamps and counters under
   `extra/llm_research/**` and the existing HCQ graph profile payload only.
4. Legality: `R` may be cache state, dependency wait, memory visibility, QMD
   scheduling, or placement. Each is a separate counter-observable.
5. Fallback: if `R` is not serialization, the named ceilings cap at ~404 us
   (229 tok/s) and this path closes to `240_UNCLOSED` for the R term.
6. Contract: partition `R` into `wait_exit - last_producer_ready`, launch
   gap, and cache-state components with zero unexplained remainder.
7. Arms: isolated = clean chained HCQ `C` (retained); installed = `PROFILE=1`
   command interval `P` with wait-exit timestamps. PDL-on vs PDL-off controls.
8. Census gate: 596 nodes unchanged; no node, copy, or materialization added.
9. Reverse wall bracket, token-SHA gate, +50 us promotion bar.
10. Rollback: measurement only; no production edit.
11. Projected ceiling ~404 us, labelled unmeasured.
12. Prohibited: model-name or block-list dispatch.

## 2026-08-23 exact-live protocol amendment (authoritative)

The earlier synthetic C0/C2/C3 procedure is historical and is not sufficient
for adjudicating `R`. The following protocol is mandatory for remaining K/V
and O rows.

* **[MEASURED]** Exact live Q occurrence-0 closed its timing identity with
  zero residual. Authority: `output/nv-r-predecessor-conditioned-exact-result-20260823.md`.
* **[INFERRED]** The Q result supports a predecessor-working-set/production
  prefix effect, not a generic QMD-depth or overlap fix. Direct L2/TLB versus
  instruction-state attribution remains **[UNMEASURED]**.
* **[UNMEASURED]** Nothing is generalized from Q occurrence-0 to K/V or O
  without repeating the live capture.

### Required per-row gate

1. Capture the installed producer prefix and target by VA, command sequence,
   allocation sizes, and cubin SHA in one fresh process. No substitutes.
2. Measure installed `P` and every arm in-session with locked clocks and
   `PROFILE=1`; retain raw command/profile records.
3. Run disjoint `C0`, `C2`, `C3`, `C4`, `C5`, `C6`, and installed `P`; use both
   arm orders and fresh processes.
4. Validate output SHA after every arm and token SHA for wall promotion. Use
   reverse-bracket medians and retain raw sessions.
5. Require `P-C0` closure to the component sum with zero residual. Report
   `node_sum`, `union`, `wall`, `useful_body`, and `overlap` separately.
6. Cache and schedule mechanisms are **[UNMEASURED]** without positive
   eviction/counter evidence or predecessor-conditioned timestamps,
   respectively.
7. Book recovery only after a token-SHA reverse wall bracket reaches the
   promotion bar; ceilings remain attribution until then.

Run K/V first, then O, using this gate verbatim. Do not implement a generic
scheduler or fusion change from a single-row result. No production,
renderer, scheduler, runtime, or model code is authorized by this amendment.
