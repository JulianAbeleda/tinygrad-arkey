# Harness Principles Template

Use this template when a project needs benchmark, evaluator, or experiment harnesses that can influence engineering
decisions.

The goal is not to standardize one benchmark framework. The goal is to keep performance, correctness, quality, and
promotion claims reproducible enough that a future reader can tell what was measured, why it mattered, and whether it
should change product behavior.

## Core Rule

A harness is part of the system under test.

If the harness changes workload shape, lifecycle, environment, comparator, correctness or quality gate, timing source,
compile/warmup handling, dispatch path, or promotion policy, it can change the result.

Do not treat harness code as disposable glue when its output is used for a claim.

## Authority Layers

Separate measurement layers explicitly.

| authority layer | use | can promote? |
|---|---|---|
| whole-system / end-to-end gate | final product-facing performance, quality, or behavior claim | yes |
| isolated local A/B gate | diagnostic comparison against the current winner | no |
| profiler / tracing output | attribution and root-cause analysis | no |
| debug logs / print timing | debugging only | no |
| synthetic microbench | diagnostic only unless the shipped path is the same synthetic path | no |

Promotion is reported by a harness, not silently applied by it. Default, policy, or rollout changes should remain an
owner decision unless the project has an explicit automated promotion system.

## Required Artifact Contract

A claim-bearing artifact should record:

1. workload shape and context;
2. candidate id and primitive or feature class;
3. comparator id and why it is the current winner or baseline;
4. exact command and relevant environment;
5. source revision and dirty state;
6. hardware, platform, and performance-state details relevant to the claim;
7. warmup and compile handling;
8. repeats, median, spread, and reproducibility/noise band;
9. correctness, quality, or acceptance gate;
10. local diagnostic timing versus whole-system authority;
11. pass/fail threshold;
12. final verdict and stop reason;
13. ledger, decision record, or refutation links.

If a project uses machine-readable artifacts, make these fields schema-visible instead of relying on prose reports.

## Comparator Discipline

Every harness that claims improvement needs a live comparator.

The artifact must say:

- what candidate was measured;
- what it was compared against;
- why that comparator is current;
- whether the comparator represents a shipped/default path, a known best local result, or a fixed external baseline.

Do not compare against stale baselines when a current winner exists.

## Correctness And Quality First

Run correctness, quality, or acceptance checks before speed is treated as meaningful.

For approximate or reordered computations, state:

- the tolerance;
- the reference output or behavior;
- the reason the tolerance is valid;
- whether failures stop the run or merely classify the candidate.

Speed without a valid acceptance gate is a diagnostic, not a product claim.

## Reproducibility Bands

A bare median is not enough.

For repeated measurements, record at least:

- sample count;
- median;
- min and max;
- mean when useful;
- spread percentage;
- median absolute deviation or another robust noise estimate;
- the threshold required to overcome the measured noise band.

Near-threshold movement is learning, not promotion. Record it as inconclusive, local-only, or failed according to the
project's verdict vocabulary.

## Verdict Discipline

Use a small shared verdict vocabulary.

Example verdict classes:

- `PASS_PROMOTE` - passed whole-system authority and may be considered for owner-approved promotion.
- `LOCAL_PASS_SYSTEM_FAIL` - isolated result looked good, whole-system authority did not.
- `FAIL_LOCAL_AB` - did not clear the diagnostic comparator gate.
- `FAIL_CORRECTNESS` - failed the required acceptance gate.
- `MEASUREMENT_UNSTABLE` - noise or environment made the result unusable.
- `REFUTED` - closed by prior evidence unless new scope or evidence appears.

Do not invent one-off verdict strings in child harnesses if a central evaluator or ledger consumes them.

## Lifecycle Loop

A reusable search or optimization harness should follow this loop:

```text
template space
-> generated candidate
-> structural or policy pruning
-> correctness or quality gate
-> isolated diagnostic measurement
-> whole-system authority gate when applicable
-> machine-readable artifact
-> ledger or refutation memory
-> next candidate
```

If a candidate cannot be generated, evaluated, pruned, and remembered, it is still a manual experiment, not a durable
search row.

## Cleanup Policy

Historical probes can remain as provenance when they are not live decision authorities.

If an old probe becomes live again, first bring it under the current artifact contract. Do not let a one-off script
influence defaults, claims, or roadmap decisions without comparator, acceptance, reproducibility, and authority data.

## Portability Rules

- Replace project-specific benchmark names with local authority layers.
- Replace hardware-specific fields with the platform state that matters for the project.
- Keep artifact fields stable even if the harness implementation changes.
- Keep promotion policy separate from measurement code.
- Store refutations where future work will actually find them.
- Keep generated artifacts free of secrets, private paths, and transient chat state.

