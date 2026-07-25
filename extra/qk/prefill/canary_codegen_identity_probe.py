"""Compile the 6 packed-WMMA canary kernels for 14B shapes and sha the code objects. Compile-only."""
import sys, hashlib, json
from extra.qk.prefill.packed_wmma_prefill_candidates import PACKED_WMMA_GEOM, _payload_for_shape, _mutate_payload
from extra.qk.model_profiles import profile_by_id
from extra.qk.runtime_specs import _canonical_full_kernel_identity
from extra.qk.prefill.current_prefill_execution_adapter import prepare_current_prefill_compile
out={}
pr=profile_by_id("qwen3_14b_q4k_m_gfx1100")
for (q,r),g in sorted(PACKED_WMMA_GEOM.items()):
    try:
        shape=pr.role_shape(r).mnk
        mut=_mutate_payload(_payload_for_shape(r,shape),g)
        ident=_canonical_full_kernel_identity(mut)
        _,ev=prepare_current_prefill_compile(mut, ident, device="AMD")
        out[f"{q}/{r}"]={"shape":list(shape),"identity":ident[:16],
                         "binary_sha256":(ev.get("binary_sha256") if isinstance(ev,dict) else None)}
    except Exception as e:
        out[f"{q}/{r}"]={"error":f"{type(e).__name__}: {str(e)[:90]}"}
print(json.dumps(out, indent=1, sort_keys=True))
