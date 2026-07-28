import pickle, os

ROOT = "/home/ubuntu/tinygrad-arkey"
d = pickle.load(open("/tmp/audit_graph.pkl", "rb"))
edges = d["edges"]
mod2file = d["mod2file"]
file2mod = d["file2mod"]

def F(rel):
    p = os.path.join(ROOT, rel)
    assert os.path.exists(p), p
    return p

entry_points = [
    "tinygrad/llm/model.py",
    "tinygrad/llm/prefill_routes.py",
    "tinygrad/llm/route_ops.py",
    "tinygrad/llm/prefill_policy.py",
    "tinygrad/llm/admission.py",
    "extra/llm_research/prefill/prefill_whole_synced.py",
    "extra/llm_research/prefill/packed_wmma_prefill_candidates.py",
    "extra/llm/generate.py",
]
entry_points = [F(x) for x in entry_points]

# Dynamic (string-based importlib) targets confirmed reachable by grepping actual
# call sites of qk_ops.<fn> / cg_extras.<fn> / _PRODUCTION_ADAPTER_LOADERS within
# files that are already part of the live closure. AST import scanning misses
# these because the module name is a runtime string, not a literal import stmt.
dynamic_roots = [
    "extra/llm_research/mmq_ds4_logical_emitter.py",          # prefill_research_routes.py: qk_ops.packed_*candidate/pack_q8_1_mmq_*/emit_q4k_q8_mmq_ds4
    "extra/llm_research/decode/current_decode_execution_adapter.py",  # operand_path_execution_worker.py _PRODUCTION_ADAPTER_LOADERS
    "tinygrad/codegen/late/recurrence.py",   # tinygrad/codegen/__init__.py direct unroll_recurrence
    "extra/llm_research/coalesced_load_lowering.py",     # tinygrad/codegen/__init__.py cg_extras.coalesce_loads
    "tinygrad/codegen/late/warp_reduce.py",    # tinygrad/codegen/__init__.py direct pm_warp_reduce; qk callers use core primitives
    "tinygrad/codegen/late/reg_store.py",      # tinygrad/codegen/__init__.py direct register-store matcher
    "tinygrad/codegen/late/fdot2.py",           # tinygrad/codegen/__init__.py direct fdot2 hooks; gemm_consumer.py lower_fdot2_add
    "tinygrad/codegen/late/list_scheduler.py", # tinygrad/codegen/late/linearizer.py direct list_schedule/structural_ops
    "extra/llm_research/codegen_extensions.py",          # tinygrad/renderer/isa/extensions.py experimental.amd_isa_extension_descriptors
    "extra/llm_research/q6k_route_spec.py",              # route_ops.py qk_ops.emit_q6k_gemv_kernel / q6k_spec_for_role (called from tinygrad/llm/*)
    "extra/llm_research/memory_adaptive_runtime_collector.py",  # route_ops.py qk_ops.install_memory_adaptive_model_adapters
    "extra/llm_research/gemv_g3_codegen_lowering.py",    # route_ops.py qk_ops.q4k_g3_lanemap_gemv_kernel
    "extra/llm_research/quant/q6_k_gemv_primitive.py",   # route_ops.py qk_ops.q6k_parse_opt
    "extra/llm_research/decode/flash_decode_attention_executor.py",  # route_ops.py qk_ops.flash_decode_live_split_block_tile
]
entry_points += [F(x) for x in dynamic_roots]

seen = set()
stack = list(entry_points)
while stack:
    f = stack.pop()
    if f in seen:
        continue
    seen.add(f)
    for nxt in edges.get(f, ()):
        if nxt not in seen:
            stack.append(nxt)

qk_keep = sorted(f for f in seen if "/extra/llm_research/" in f)
other_extra_keep = sorted(f for f in seen if "/extra/" in f and "/extra/llm_research/" not in f)
tinygrad_keep = sorted(f for f in seen if "/tinygrad/" in f and "/extra/" not in f)

print("=== extra/llm_research KEEP files ===")
for f in qk_keep:
    print(os.path.relpath(f, ROOT))
print()
print("count qk keep:", len(qk_keep))
print()
print("=== other extra/ KEEP files ===")
for f in other_extra_keep:
    print(os.path.relpath(f, ROOT))
print()
print("=== tinygrad/ files touched (for reference, not the focus) ===")
print(len(tinygrad_keep))

pickle.dump({"seen": seen, "qk_keep": qk_keep}, open("/tmp/audit_seen.pkl", "wb"))
