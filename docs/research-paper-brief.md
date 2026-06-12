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
