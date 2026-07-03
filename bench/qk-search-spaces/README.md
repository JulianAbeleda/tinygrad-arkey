# QK Search-Space Manifests

This directory is tinygrad-side route context, not the active search engine.

BoltBeam owns candidate generation, evaluation policy, roofline reporting, and ledgers. tinygrad keeps these files so
runtime gates can explain which generated route is expected for a shape, which routes are rollback/reference only, and
which axes are refuted.

## Active Files

- `default_route_manifest.json` - local source of truth for default/rollback/refuted route state.
- `search_profiles.json` - profile/shape context retained for audits; candidate generation lives in BoltBeam.
- `quant_semantics.json` - quant layout facts used by route/audit tooling.
- `targets/*.json`, `profiles/*.json` - static profile and target descriptors.

## Rule

Do not use this directory to resurrect a tinygrad-local search loop. If a candidate/evaluator change is needed, implement
it in BoltBeam and update tinygrad only with the resulting route policy, runtime adapter, or verification gate.
