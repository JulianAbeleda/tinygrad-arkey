# S0 clean-source probe

Status: **BLOCKED**.

An isolated `git archive` of commit `131b22a8b` was created at
`/tmp/tinygrad-arkey-clean-s0-20260829-3229515/source`. The required runner
`extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py` is absent from
that archive, including the required `--q4-v` route flag. It exists only in
the dirty working tree, so no mixed-source GPU run was performed.

The absolute current cut policy was retained as `current-cut-policy.json`.
Archive and per-file manifests, plus the exact status, are recorded here.
