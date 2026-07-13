from extra.qk.prefill.pure_register_direct_l2_decision import candidate, decide

IDENTITY = "a" * 64
SHAPE = {"m": 512, "n": 4096, "k": 4096}


def _row(storage, binary, samples, ok=True):
  return candidate(role="attn_qo", shape=SHAPE, identity=IDENTITY, binary_sha256=binary,
                   storage=storage, artifact={"status": "pass"}, correctness={"status": "pass" if ok else "blocked"},
                   environment={"commit": "82f5a586a", "target": "gfx1100", "abi": "amdgpu_kernel",
                                "launch": [256, 1, 1], "protocol": "paired-v1"}) | {
      "samples_ms": samples,
      "counters": {group: {"status": "live"} for group in ("l2", "memory", "compute")}}


def test_cpu_pair_promotes_only_material_stable_direct_l2():
  report = decide({"direct_l2": _row("direct_l2", "b" * 64, [8.0] * 12),
                   "lds": _row("lds", "c" * 64, [10.0] * 12)})
  assert report["decision"] == "promote_direct_l2"


def test_cpu_pair_retains_lds_when_not_materially_faster():
  report = decide({"direct_l2": _row("direct_l2", "b" * 64, [9.9] * 12),
                   "lds": _row("lds", "c" * 64, [10.0] * 12)})
  assert report["decision"] == "retain_lds"


def test_cpu_pair_blocks_missing_identity_or_evidence():
  direct = _row("direct_l2", "b" * 64, [8.0] * 3, ok=False)
  report = decide({"direct_l2": direct, "lds": _row("lds", "c" * 64, [10.0] * 12)})
  assert report["status"] == "blocked"
  assert any("correctness" in reason or "samples" in reason for reason in report["blockers"])


def test_cpu_pair_accepts_distinct_candidate_identities_with_shared_semantic_key():
  direct = _row("direct_l2", "b" * 64, [8.0] * 12) | {"canonical_identity": "d" * 64, "pair_key": "semantic-v1"}
  lds = _row("lds", "c" * 64, [10.0] * 12) | {"canonical_identity": "e" * 64, "pair_key": "semantic-v1"}
  assert decide({"direct_l2": direct, "lds": lds})["decision"] == "promote_direct_l2"
