# Research Paper Brief — "search vs hand-written primitives" / the stack-ownership angle

Status: EXPLORATORY. The grand "TempleOS bet" (simplify+own the stack so the
machine optimizes it) is NOT novel on its own — it restates superoptimization,
HW/SW co-design, DSAs, and tinygrad's own public thesis. Do NOT write that paper.
The goal is to find whether a NARROW, FALSIFIABLE, NOVEL claim survives a
literature review, grounded in our actual campaign data.

## Codex deliverable #1 (GATE): novelty assessment, BEFORE any claim
Survey the prior art and report, for each candidate angle below, whether it is
ALREADY CLAIMED, PARTIALLY CLAIMED, or appears OPEN. Cite specific papers.
Do not formulate our claim until this is done. If everything is already claimed,
say so — a negative result here saves months.

Prior art to differentiate against (find the exact claims, not just the topic):
- AutoTVM, Ansor (OSDI'20), MetaSchedule/AutoTensorIR — what each claims about
  search-space generation vs human templates.
- Superoptimization: STOKE, Souper, Denali — claims about closed/small spaces.
- Cost-model accuracy literature — is "search efficacy is bounded by cost-model
  achievability" already an established result? (This is the crux.)
- Hardware/software co-design + DSA papers — claims about simplification enabling
  automation.
- Any "experience report" / empirical-characterization papers on autotuning
  failure modes, esp. on AMD/consumer GPUs (most are NVIDIA/datacenter).
- LLM-agent kernel synthesis (BOLT, FACT) — what they claim about automating L2.

## Candidate claims (each MUST pass the novelty gate; ranked by likely novelty)
1. EMPIRICAL (most defensible): an experience report + the layer-1/layer-2
   regime framework, with the gfx1100 quantized-decode case study — search (BEAM)
   found nothing, hand-seeded primitives gave 3.7x, correctness-gated throughout.
   Contribution = rigorous measurement on the NEGLECTED consumer-AMD path +
   a clarifying regime characterization. Modest but real.
2. THEORETICAL (higher risk/reward): "search efficacy is governed by cost-model
   achievability, which is governed by hardware opacity, not abstraction depth."
   Falsifier: search should do better on more-transparent targets at equal
   abstraction depth. Needs the cost-model-accuracy lit-check (claim #1 of gate)
   — this may already be known.
3. ARTIFACT/SYSTEMS: contribute quant-GEMV as a SEARCHABLE primitive integrated
   into tinygrad's BEAM (TC-style), + measurement of the blend. A systems
   contribution, not a theory claim.

## Our evidence assets (in-repo)
- bench/ : ROCm baseline, tinygrad-native baseline, BEAM sweep (nothing),
  Q4/Q6 primitive results (3.7x), profiles, greedy A/B correctness.
- docs/amd-rocm-llamacpp-research.md, docs/amd-decode-optimization-plan.md :
  the full measured campaign, falsifications recorded.

## Hard rules (lessons from this session)
- Verify novelty before claiming it. One empty search is NOT proof of novelty.
- Every claim needs a falsifier and an experiment that could kill it.
- Distinguish "clarifying framing" (publishable as experience/measurement) from
  "novel theoretical result" (needs the cost-model lit to be genuinely open).
- If the honest finding is "already known," report that. That is a success.

## Refinement (2026-06-12) — the action-space framing + the in-hand experiment

### Sharper lens for the theoretical claim (candidate #2), to lit-check
Reframe "hardware opacity" as an ACTION-SPACE boundary (substrate-independent):
- An agent's actions form tiers by how much they edit the representation:
  T0 schedule (tile/unroll/vectorize), T1 representation-parameter (layout/dtype/
  instruction from a fixed menu), T2 representation-generation (propose structures
  from rules), T3 open representation-editing (reinterpret/transfer/hypothesize).
- Claim: an agent closes the optimization loop IFF its action set reaches the tier
  containing the edit the optimum requires. BEAM = T0; the quant optimum needs
  ~T2/T3; hence BEAM cannot reach it — independent of "machine vs human".
- Decompose "edit the canvas" into propose / validate / evaluate / aim. Finding:
  PROPOSE (rewriters/synthesizers) and VALIDATE (equivalence checking) are
  tooling-solved; the binding gap is AIM — navigating an astronomical edit-space
  needs PRIORS (learned knowledge of which edits help), and EVALUATE needs a cost
  model (bounded by opacity). So the gap is priors + perception, NOT tooling.
- NOVELTY CAVEAT: this is very likely a clean articulation of known ideas
  (inductive bias / search-space theory; compiler phase-ordering; the
  "no free lunch" line). Codex: check whether the action-tier framing OR the
  "priors not tooling" decomposition is an established result. Probably it is.
  If so, it is framing, not contribution.

### The strongest empirical contribution may already be in-hand
The campaign itself instantiates the experiment, by accident:
- BEAM (a T0 agent, no prior) on gfx1100 quant-GEMV -> found nothing, crashed.
- Codex (an LLM-agent, T3, carrying priors learned from human kernels) -> proposed
  the packed-uint32 + register-dequant representation and won 3.7x, correctness-
  gated (greedy A/B 32/32).
This is suggestive evidence that the search-vs-hand-written gap is a PRIORS gap:
the same problem, unsolved by search, solved by an agent with priors editing the
canvas. NOT a clean controlled study (post-hoc, single case, no ablation).

Candidate claim #4 (empirical, possibly novel): a CONTROLLED demonstration that
the gap is priors, not search-effort or tooling — same kernel/target, compare
(a) BEAM + cost model [perception, no prior], (b) BEAM over a larger move set,
(c) an LLM-agent proposer + cost-model evaluator + verifier [prior + perception +
validation], measured on consumer AMD with correctness gates. If (c) closes what
(a)/(b) cannot, the priors-not-tooling claim is demonstrated, not just argued.
The asset: we already have the (a)-fails / informal-(c)-succeeds halves documented.

Codex lit-check additions: program synthesis with learned priors; LLM kernel
synthesis (FACT, BOLT, KernelBench-style) — do any already claim/demonstrate
"priors close the search-vs-hand-written gap" with a controlled comparison? If a
controlled priors-vs-search study on real GPU kernels exists, claim #4 is taken.
