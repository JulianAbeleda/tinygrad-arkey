# CPU-only staged `ffn_gate_up` artifact

This directory retains the exact `ffn_gate_up (512,17408,5120)` compact-K256
staged family generated from clean revision
`3fa4cd6195e460930417732fb404521e33c9cf3c`.

- Family identity: `sha256:4e82044650b7c03a579b42b8dd270389d280c2f5eab9f9004a3bc83dbe79917f`
- PROGRAM key: `14f2a216a8a7609e8a251fe3869b3fb146fd5d5a8ca0ec468120e0fbcbd54a60`
- HSACO SHA256: `149ba322c1a99c1fa056d25c6230bc8908c27f15fe94b177276c5808eebe8bf3`
- Retained source-SINK key: `a3a4f98c4ebebfe8f770f2f3f4e611c22f92510845e482d1bc79dfb75963a495`

The seven-file `bundle/` is a v2 frozen target artifact. It was compiled once
with the CPU-only `AMDISARenderer` path under the exact environment retained in
its manifest; no `Device`, GPU runtime, allocation, queue, or dispatch was
created. The `evidence/` directory retains its staged-family identity,
generation provenance, passing HSACO audit, zero-spill native resources, and
the deterministic C3a/C3b memory-certificate closeout.

The retained C3 file is a compact summary. Replay its complete child proofs
from the repository root (about six CPU minutes on the generating host):

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from extra.qk.mmq_exact_role_spec import exact_role_spec
from extra.qk.mmq_frozen_staged_family import load_frozen_staged_family_manifest
from extra.qk.mmq_frozen_staged_memory_certificate import certify_frozen_staged_full_memory

root = Path("docs/artifacts/qwen3-14b-prefill-ffn-gate-up-staged-3fa4cd619-20260719")
family = load_frozen_staged_family_manifest(
  root / "evidence/qk-ffn-gate-up-staged-3fa4cd619-r1-20260719-family.json",
  role_spec=exact_role_spec("ffn_gate_up"), frozen_bundle=root / "bundle")
certificate = certify_frozen_staged_full_memory(family)
summary = json.loads(
  (root / "evidence/qk-ffn-gate-up-staged-3fa4cd619-c3-full-summary.json").read_text())
assert certificate["certificate_sha256"] == summary["certificate_sha256"]
assert certificate["c3a"]["certificate_sha256"] == summary["children"]["C3a"]["certificate_sha256"]
assert certificate["c3b"]["certificate_sha256"] == summary["children"]["C3b"]["certificate_sha256"]
assert certificate["c3b"]["compact_program"]["workitems_exhaustively_evaluated"] == \
  summary["children"]["C3b"]["exhaustive_launch_coordinate_count"]
assert certificate["c3b"]["final_native"]["native_address_arithmetic"]["projected_address_evaluations"] == \
  summary["children"]["C3b"]["projected_address_evaluations"]
print(certificate["certificate_sha256"])
PY
```

The retained PM4 C4 evidence
`evidence/qk-ffn-gate-up-staged-8cad0c4ba-c4-pm4-20260719.json` subsequently
passes guarded runtime preconstruction with zero target dispatches, clean
pre/post health probes, and no kernel-fault marker. This is not numerical
correctness: AQL C4, all C5+ target execution, matched full-role timing,
whole-model validation, and production promotion remain open and are not
claimed.

The first guarded PM4 C5 prefix-1 attempt is retained as
`evidence/qk-ffn-gate-up-staged-f0a46ff09-c5-pm4-prefix1-20260719.json`.
Its single target dispatch raised a precise `0xFFFFFFBFE000` MMU fault; the
parent observed SQ/page-fault/MES/reset markers, the reset recovered, and the
postflight tiny-health probe passed. The result is `BLOCKED`, not a health or
correctness pass. No retry, prefix-3, full-20, direct, transition, or AQL
attempt followed, and no output comparison was reached.

The current CPU-only audit is repairing exception-path evidence propagation so
the already-captured five argument VAs and kernarg qwords cannot be discarded
when synchronous dispatch raises. Until that repair and its injected tests are
complete, another target dispatch is not permitted. The older frozen
`99c7ee0c...` control executed the same ABI and geometry correctly; the new
`149ba322...` schedule/order delta is under audit but is not yet a proven root
cause.
