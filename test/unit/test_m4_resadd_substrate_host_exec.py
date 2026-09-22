"""M4 resadd rangeify substrate S3 host-execution locks (CPU hermetic, no GPU).

Scope: `docs/task_workflow/input/m4-resadd-rangeify-substrate-scope-20260806.md` S3 fallback arm:
execute a single-block folded epi_resadd subgraph on CPU with the residual as the block-output
AFTER and assert numeric equality against the copy-ABI variant. The proof kernel is the production
epi_resadd emitter shape reading the residual arg with a flat row index; this locks both the fold
(`_validated_residual_view` fires on the real block-output chain and rejects layer 0) and the
shaped-arg codegen fold (`pm_index_is_shrink`) that makes the folded kernel renderable.
"""
from extra.llm_research.decode.m4_resadd_substrate_host_exec import run_proof


def test_fold_fires_on_real_chain_and_layer0_fails_closed():
  result = run_proof()
  assert result["fold"] and result["fold_base"] == "Ops.GETTUPLE"
  assert result["layer0_reject"]


def test_folded_epi_resadd_matches_copy_abi_bitwise():
  result = run_proof()
  assert result["fold_eq_copy_bitwise"]
  assert result["sha_fold"] == result["sha_copy"]


def test_zero_dot_fold_reads_the_producer_buffer_directly():
  result = run_proof()
  assert result["zero_dot_fold_eq_copy"]
  assert result["zero_dot_fold_eq_producer"]
