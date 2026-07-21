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
    "extra/qk/prefill_whole_synced.py",
    "extra/qk/prefill/packed_wmma_prefill_candidates.py",
    "extra/llm/generate.py",
]
entry_points = [F(x) for x in entry_points]

# Dynamic (string-based importlib) targets confirmed reachable by grepping actual
# call sites of qk_ops.<fn> / cg_extras.<fn> / _PRODUCTION_ADAPTER_LOADERS within
# files that are already part of the live closure. AST import scanning misses
# these because the module name is a runtime string, not a literal import stmt.
dynamic_roots = [
    "extra/qk/mmq_ds4_logical_emitter.py",          # prefill_research_routes.py: qk_ops.packed_*candidate/pack_q8_1_mmq_*/emit_q4k_q8_mmq_ds4
    "extra/qk/decode/current_decode_execution_adapter.py",  # operand_path_execution_worker.py _PRODUCTION_ADAPTER_LOADERS
    "extra/qk/codegen_recurrence_unroll.py",   # tinygrad/codegen/__init__.py cg_extras.unroll_recurrence
    "extra/qk/coalesced_load_lowering.py",     # tinygrad/codegen/__init__.py cg_extras.coalesce_loads
    "extra/qk/warp_reduce_lowering.py",        # tinygrad/codegen/__init__.py cg_extras.warp_reduce_pm()
    "extra/qk/reg_store_devec.py",             # tinygrad/codegen/__init__.py cg_extras.reg_store_devec_pm()
    "extra/qk/fdot2_lowering.py",              # tinygrad/codegen/__init__.py cg_extras.fdot2_pm/line_lower_fdot2; codegen/opt/gemm_consumer.py lower_fdot2_add
    "extra/qk/codegen_list_scheduler.py",      # tinygrad/codegen/late/linearizer.py cg_extras.list_schedule/structural_ops
    "extra/qk/codegen_extensions.py",          # tinygrad/renderer/isa/extensions.py experimental.amd_isa_extension_descriptors
    "extra/qk/q6k_route_spec.py",              # route_ops.py qk_ops.emit_q6k_gemv_kernel / q6k_spec_for_role (called from tinygrad/llm/*)
    "extra/qk/memory_adaptive_runtime_collector.py",  # route_ops.py qk_ops.install_memory_adaptive_model_adapters
    "extra/qk/gemv_g3_codegen_lowering.py",    # route_ops.py qk_ops.q4k_g3_lanemap_gemv_kernel
    "extra/qk/quant/q6_k_gemv_primitive.py",   # route_ops.py qk_ops.q6k_parse_opt
    "extra/qk/flash_decode_attention_executor.py",  # route_ops.py qk_ops.flash_decode_live_split_block_tile
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

qk_keep = sorted(f for f in seen if "/extra/qk/" in f)
other_extra_keep = sorted(f for f in seen if "/extra/" in f and "/extra/qk/" not in f)
tinygrad_keep = sorted(f for f in seen if "/tinygrad/" in f and "/extra/" not in f)

print("=== extra/qk KEEP files ===")
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
