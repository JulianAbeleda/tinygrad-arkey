# Model-Agnostic Audit Profile System Scope

## Goal

Convert the current audit system from a Qwen3-8B/gfx1100-centered decode evaluator into a profile-driven audit framework.

The goal is not to copy upstream BEAM or require fully general kernel search. The goal is to make every benchmark, candidate, threshold, comparator, shape assumption, and hardware assumption explicit in a profile so the audit can answer:

```text
Does this candidate work for this declared model/device/workload profile?
```

Instead of implicitly answering:

```text
Does this candidate work for the current Qwen3-8B-Q4_K_M / RX 7900 XTX / gfx1100 decode setup?
```

## Current State

The current evaluator has a good modular skeleton:

- candidate registry: `bench/qk-decode-eval/candidates.json`
- verdict vocabulary: `extra/qk_modes.py`
- evaluator runner: `extra/qk_decode_eval.py`
- lifecycle loop: `extra/qk_lifecycle_search_loop.py`
- harness contract: `extra/qk_harness_contract.py`
- decode W==D authority runner: `extra/qk_decode_runtime_overhead.py`

But the implementation still embeds one benchmark profile in multiple places.

## Problem Statement

The audit system is modular at the registry/verdict layer, but not fully model-agnostic at the measurement layer.

Known hard gates and implicit profile assumptions include:

| area | current issue | consequence |
|---|---|---|
| Model path | `extra/qk_paths.py` defaults to `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`. | A different model can be supplied via `QK_MODEL`, but the default profile remains implicit. |
| Context ladder | `extra/qk_decode_runtime_overhead.py` defaults `QK_CKPTS` to `128,512,1024,4096`. | Candidate-declared contexts are not automatically authoritative. |
| Max context | `extra/qk_decode_runtime_overhead.py` hardcodes `MAXC=4608`. | Other model/context limits require env or code changes. |
| Local flash A/B shape | `extra/qk_decode_eval.py` hardcodes `Hd=128`, `Hq=32`, `Hkv=8`, `MAXC=4608`, `Tc=1024`. | The local attention diagnostic is Qwen3-8B-shaped, not model-derived. |
| Promotion thresholds | `candidates.json` uses fixed gates such as `wd_min_pct_ctx1024` and `wd_min_pct_ctx4096`. | Threshold semantics are tied to one context ladder. |
| Comparator | `extra/qk_harness_contract.py` fixes `DECODE_COMPARATOR = "gqa_coop_vec"`. | Comparator is centralized, but not profile-specific. |
| Hardware string | `extra/qk_harness_contract.py` returns `RX 7900 XTX / gfx1100`. | Actual device/profile mismatch is not first-class. |
| Runner parsing | `extra/qk_decode_eval.py` expects fields like `best_speedup_vs_coop` and `ctx == 1024`. | New primitive families need custom evaluator code instead of normalized metrics. |

## Target Design

Introduce an explicit benchmark profile as the first-class audit boundary.

A profile defines:

```text
model + quantization + architecture + device + workload phase + context ladder + max context + batch/token shape + comparator + thresholds + route expectations
```

A candidate defines:

```text
primitive family + env/flags + runner rungs + supported profiles + required evidence + expected verdict
```

A runner emits:

```text
normalized metrics + correctness/quality + structural evidence + authority metadata
```

A decision policy maps:

```text
profile + candidate + normalized metrics -> verdict
```

## Proposed Profile Schema

Create a profile registry, for example:

`bench/qk-audit-profiles/profiles.json`

Example shape:

```json
{
  "schema": "qk_audit_profiles_v1",
  "profiles": [
    {
      "id": "qwen3_8b_q4k_m_gfx1100_decode_v1",
      "phase": "decode",
      "model": {
        "path_env": "QK_MODEL",
        "default_path": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",
        "arch": "qwen3",
        "quantization": "Q4_K_M"
      },
      "device": {
        "backend": "AMD",
        "arch": "gfx1100",
        "label": "RX 7900 XTX / gfx1100"
      },
      "workload": {
        "batch": 1,
        "token_phase": "T=1 decode",
        "contexts": [512, 1024, 2048, 4096],
        "max_context": 4608,
        "prompt_seed": 20260617
      },
      "shape_expectations": {
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "gqa_ratio": 4
      },
      "comparator": {
        "id": "gqa_coop_vec",
        "why": "current shipped default decode-attention primitive for this profile"
      },
      "thresholds": {
        "local_min_speedup": 1.05,
        "repro_band_max_pct": 5.0,
        "promotion": {
          "min_gain_by_ctx_pct": {
            "1024": 5.0,
            "4096": 7.0
          },
          "max_regress_by_ctx_pct": {
            "512": 1.0
          }
        },
        "quality": {
          "correctness_tol": 0.02,
          "dnll_max": 0.01
        }
      }
    }
  ]
}
```

## Required Refactor

### 1. Add profile registry

Add `bench/qk-audit-profiles/profiles.json` with the current Qwen3-8B/gfx1100 decode profile as the first profile.

This should not change behavior. It only makes the current implicit profile explicit.

### 2. Add profile loader

Add a small import-safe helper, for example:

`extra/qk_audit_profile.py`

Responsibilities:

- load profile by `--profile` or `QK_AUDIT_PROFILE`
- expose model path, contexts, max context, comparator, thresholds, hardware label
- validate that required profile fields exist
- avoid importing tinygrad

### 3. Wire evaluator to profile

Update `extra/qk_decode_eval.py` to accept:

```text
--profile qwen3_8b_q4k_m_gfx1100_decode_v1
```

Then use profile fields for:

- model path
- context ladder
- thresholds
- comparator metadata
- hardware/profile labels
- route expectations

### 4. Wire W==D runner to profile

Update `extra/qk_decode_runtime_overhead.py` so `MAXC`, `QK_CKPTS`, prompt seed, and model path come from profile unless explicitly overridden.

The candidate `contexts` field should either:

- override profile contexts for that run, or
- be validated as a subset of the profile contexts.

Do not leave candidate contexts as documentation-only metadata.

### 5. Normalize runner output

Require child harnesses to emit a normalized metrics block:

```json
{
  "metrics_by_ctx": {
    "1024": {
      "speedup_vs_comparator": 1.08,
      "candidate_us": 55.2,
      "comparator_us": 59.6
    }
  },
  "correctness": {
    "checked": true,
    "metric": "max_err",
    "value": 0.0001,
    "threshold": 0.02,
    "passed": true
  },
  "authority": "clock-pinned local diagnostic",
  "structural_evidence": {
    "route_fired": true,
    "materialization_absent": true,
    "isa_summary": "optional"
  }
}
```

The evaluator may keep backward-compatible adapters for historical artifacts, but new candidates should use the normalized block.

### 6. Make thresholds profile-relative

Replace hard-coded evaluator logic like:

```text
ctx1024 >= 5% OR ctx4096 >= 7%, with ctx512 no worse than -1%
```

With profile policy:

```text
promotion.min_gain_by_ctx_pct
promotion.max_regress_by_ctx_pct
promotion.aggregate_min_gain_pct, if needed
```

This keeps the current policy intact while allowing another model/profile to define a different ladder.

### 7. Make shape guards explicit

Candidate registry should declare supported profiles and shape requirements.

Example:

```json
{
  "id": "decode_attention_llama_flash_tile_owned_amdgcn_b4",
  "supported_profiles": ["qwen3_8b_q4k_m_gfx1100_decode_v1"],
  "shape_requirements": {
    "device_arch": "gfx1100",
    "batch": 1,
    "token_phase": "T=1 decode",
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128
  }
}
```

A shape-gated candidate is acceptable. The requirement is that the gate is declarative and checked before execution.

## Non-Goals

This scope does not require:

- upstream-style BEAM replacement
- fully automatic kernel generation
- making owned AMDGCN kernels portable
- supporting every GGUF architecture immediately
- removing shape-specific optimized kernels
- changing model defaults

Shape-specific kernels are allowed. Hidden shape-specific assumptions are not.

## Model-Agnostic Definition

For this project, model-agnostic means:

```text
The audit framework can load a declared model/device/workload profile,
validate whether a candidate supports that profile,
run the correct benchmark contexts for that profile,
and produce a verdict whose thresholds and comparator came from that profile.
```

It does not mean every candidate must support every model.

## Success Criteria

| criterion | pass condition |
|---|---|
| Current behavior preserved | Running the default decode suite with the default profile reproduces current behavior. |
| Profile is explicit | Model path, device, contexts, max context, comparator, and thresholds are loaded from a profile. |
| Candidate contexts are real | Runner uses candidate/profile contexts instead of silently using hardcoded defaults. |
| Shape support is checked | A candidate that only supports Qwen3-8B/gfx1100 declares that fact and is pruned for incompatible profiles. |
| Thresholds are profile-owned | Evaluator no longer names `ctx1024`/`ctx4096` gates directly. |
| Child artifacts normalize metrics | New harnesses emit `metrics_by_ctx`, correctness, authority, and structural evidence. |
| Hardware/model mismatch is visible | Artifact records both declared profile and detected runtime facts. |

## Migration Plan

### Phase 1: Documentation and profile seed

- Add this scope document.
- Add `profiles.json` with the current profile only.
- Add `qk_audit_profile.py` loader.
- No behavior change.

### Phase 2: Decode evaluator profile plumbing

- Add `--profile` to `extra/qk_decode_eval.py`.
- Load thresholds and comparator from profile.
- Keep existing defaults as compatibility fallback.
- Emit profile ID in every artifact.

### Phase 3: W==D profile plumbing

- Add profile support to `extra/qk_decode_runtime_overhead.py`.
- Pass `QK_CKPTS` and `MAXC` from profile/candidate.
- Make candidate `contexts` executable rather than descriptive.

### Phase 4: Shape-support pruning

- Add `supported_profiles` and `shape_requirements` to live candidates.
- Teach lifecycle loop to prune candidates whose profile/shape requirements do not match.
- Emit `PRUNE_PROFILE_UNSUPPORTED` or `PRUNE_SHAPE_UNSUPPORTED`.

### Phase 5: Normalized child metrics

- Add `metrics_by_ctx` to new child harnesses.
- Keep adapters for historical result fields.
- Update contract audit to prefer normalized metrics.

### Phase 6: Add a second profile

- Add one intentionally different profile, for example a different context ladder or a second model if available.
- The point is not performance. The point is to prove the evaluator is no longer hardwired to the first profile.

## Expected End State

A future audit run should be launched as:

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py \
  --profile qwen3_8b_q4k_m_gfx1100_decode_v1 \
  --candidate decode_aggressive_probe_promotion
```

And the artifact should state:

```text
profile: qwen3_8b_q4k_m_gfx1100_decode_v1
model: /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
contexts: [512, 1024, 2048, 4096]
comparator: gqa_coop_vec
thresholds: loaded from profile
candidate support: matched profile and shape requirements
verdict: profile-relative decision
```

That is the practical line between a hardcoded benchmark harness and a model-agnostic audit framework.
