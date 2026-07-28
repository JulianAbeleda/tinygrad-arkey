import pathlib, re


ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTERS = {
  ROOT / "tinygrad/llm/route_ops.py",
  ROOT / "tinygrad/codegen/experimental.py",
}
ROUTE_IMPORT = re.compile(r"^\s*(?:from|import)\s+extra\.(?:qk|audit|reference)", re.MULTILINE)


def test_tinygrad_qk_extra_imports_are_behind_adapters():
  offenders = []
  for path in (ROOT / "tinygrad").rglob("*.py"):
    if path in ADAPTERS: continue
    if ROUTE_IMPORT.search(path.read_text()):
      offenders.append(str(path.relative_to(ROOT)))
  assert offenders == []


def test_route_ops_exposes_only_live_route_adapters():
  from tinygrad.llm import route_ops
  assert set(route_ops.__all__) == {
    "automatic_promoted_prefill_graph_policy", "install_memory_adaptive_model_adapters", "route_pf16_graph_gemm",
    "describe_q4k_packed_prefill_generated", "emit_q4k_packed_prefill_kernel", "q4k_g3_lanemap_gemv_kernel",
    "q6k_spec_for_role", "emit_q6k_gemv_kernel", "q6k_vocab_scalar_reduce_eligible",
    "emit_q6k_vocab_scalar_reduce_kernel", "describe_q6k_packed_prefill", "emit_q6k_packed_prefill_kernel",
    "flash_decode_live_split_block_tile", "select_packed_wmma_prefill_candidate",
    "build_packed_wmma_warmstart_tables",
  }
