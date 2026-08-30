"""Default-off graph-owned Q/K/V projections for exact Qwen3-8B prefill."""
from __future__ import annotations
from dataclasses import dataclass
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_llama_packed_q4k_pp512_binding import FP16_DS4_SOURCE
from extra.llm_research.prefill.nv_llama_packed_q4k_pp512_binding import MAIN_SYMBOL as Q4_MAIN, FIXUP_SYMBOL as Q4_FIX
from extra.llm_research.prefill.nv_llama_packed_q6k_down_pp512_binding import FP16_D4_SOURCE
from extra.llm_research.prefill.nv_llama_packed_q6k_down_pp512_binding import MAIN_SYMBOL as Q6_MAIN, FIXUP_SYMBOL as Q6_FIX
from extra.llm_research.prefill.nv_packed_q4k_q8_llama_candidate import ARTIFACTS, SHARED_BYTES, SCRATCH_FLOATS, fastdiv

M, K, QN, KVN = 512, 4096, 4096, 1024
Q8_RECORD_BYTES = M*(K//128)*144 + 128*144
MAIN_BLOCK, FIXUP_BLOCK = (32, 8, 1), (32, 4, 1)
_BINDINGS = {}
DS4_RECORD_BYTES = M*(K//128)*144 + 128*144
D4_RECORD_BYTES = M*(K//128)*144 + 128*144

def supports(*, model_family, role, weight_type, m, n, k, device):
  return (model_family == "qwen3_8b" and device == "NV" and m == M and k == K and
          ((role == "attn_q" and weight_type == "Q4_K" and n == QN) or
           (role in ("attn_k", "attn_v") and weight_type == "Q4_K" and n == KVN) or
           (role == "attn_v" and weight_type == "Q6_K" and n == KVN)))

def _vals(n):
  f1, f16, f4 = fastdiv(1), fastdiv(K//256), fastdiv(M//128)
  sx, sy, sd = n*(K//256), M*(K//32)*9, M*n
  return (*f16,n,M,K//256,M,n,*f1,*f1,sx,sy,sd,*f1,*f1,sx,sy,sd,*f4)
def _fix_vals(n):
  f1, f16, f4 = fastdiv(1), fastdiv(K//256), fastdiv(M//128)
  return (*f16,n,M,n,*f1,M*n,*f1,M*n,*f4)

def _geometry(n):
  # llama's exact host trace deliberately keeps the full persistent service
  # geometry for Q/K/V rather than scaling CTAs with N.
  return (170,1,1), (170,4,1)

@dataclass(frozen=True)
class QKVBinding:
  ds4: object; d4: object; q4_q_main: object; q4_q_fix: object; q4_k_main: object; q4_k_fix: object; q6_main: object; q6_fix: object
  @classmethod
  def compile(cls, dev):
    ds4_src=FP16_DS4_SOURCE.replace("q8_ds4_fp16_pp512", "q8_ds4_fp16_qkv_pp512")
    ds4 = native_nv_program("q8_ds4_fp16_qkv_pp512", NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_q8_ds4_fp16_qkv_pp512_v2").compile(ds4_src), global_size=(M,8,1), local_size=(128,1,1), globals=(0,1), outs=(1,), ins=(0,))
    d4 = native_nv_program("q8_d4_fp16_qkv_pp512", NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_q8_d4_fp16_qkv_pp512_v1").compile(FP16_D4_SOURCE.replace("q8_d4_fp16_down_pp512", "q8_d4_fp16_qkv_pp512").replace("12288", "4096")), global_size=(M,8,1), local_size=(128,1,1), globals=(0,1), outs=(1,), ins=(0,))
    def main(sym, path, n): return native_nv_program(sym, (ARTIFACTS/path).read_bytes(), global_size=_geometry(n)[0], local_size=MAIN_BLOCK, globals=(0,1,2,3), outs=(2,3), ins=(0,1), vals=_vals(n), shared_mem=SHARED_BYTES)
    def fix(sym, n, path): return native_nv_program(sym, (ARTIFACTS/path).read_bytes(), global_size=_geometry(n)[1], local_size=FIXUP_BLOCK, globals=(0,1), outs=(0,), ins=(0,1), vals=_fix_vals(n))
    return cls(ds4,d4,main(Q4_MAIN,"q4k-mmq-dense.sm_120a.cubin",QN),fix(Q4_FIX,QN,"q4k-fixup-dense.sm_120a.cubin"),main(Q4_MAIN,"q4k-mmq-dense.sm_120a.cubin",KVN),fix(Q4_FIX,KVN,"q4k-fixup-dense.sm_120a.cubin"),main(Q6_MAIN,"q6k-mmq-dense.sm_120a.cubin",KVN),fix(Q6_FIX,KVN,"q6k-fixup-dense.sm_120a.cubin"))
  def new_capture(self): return QKVCapture(self)

@dataclass
class QKVCapture:
  asset: QKVBinding; trace_epoch: int = 0; cursor: int = 0
  def begin_trace(self): self.trace_epoch, self.cursor = self.trace_epoch + 1, 0
  def project_qkv(self, x, q_words, k_words, v_words, *, model_family="qwen3_8b"):
    if not self.trace_epoch: raise RuntimeError("begin_trace must establish a capture-local epoch")
    if x.dtype != dtypes.float16 or q_words.dtype != dtypes.uint32 or k_words.dtype != dtypes.uint32 or v_words.dtype not in (dtypes.uint32,dtypes.uint16): raise ValueError("QKV route requires fp16 activation and canonical packed weights")
    vtyp = "Q6_K" if v_words.dtype == dtypes.uint16 else "Q4_K"
    if not supports(model_family=model_family, role="attn_q", weight_type="Q4_K", m=M, n=QN, k=K, device=x.device) or not supports(model_family=model_family, role="attn_k", weight_type="Q4_K", m=M, n=KVN, k=K, device=x.device) or not supports(model_family=model_family, role="attn_v", weight_type=vtyp, m=M, n=KVN, k=K, device=x.device): raise ValueError("unsupported QKV route")
    self.cursor += 1; r=Tensor.empty(Q8_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device); _,r=x.uop_program(r,fxn=lambda *_:self.asset.ds4)
    outs=[]
    for w,n,main,fix in ((q_words,QN,self.asset.q4_q_main,self.asset.q4_q_fix),(k_words,KVN,self.asset.q4_k_main,self.asset.q4_k_fix)):
      o=Tensor.empty(M*n,dtype=dtypes.float32,device=x.device); s=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device); w,r,o,s=w.uop_program(r,o,s,fxn=lambda *_,p=main:p); o,s=o.uop_program(s,fxn=lambda *_,p=fix:p); outs.append(o.reshape(M,n))
    w=v_words; main,fix=(self.asset.q6_main,self.asset.q6_fix) if vtyp == "Q6_K" else (self.asset.q4_k_main,self.asset.q4_k_fix)
    vr=r
    if vtyp == "Q6_K":
      vr=Tensor.empty(D4_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device); _,vr=x.uop_program(vr,fxn=lambda *_:self.asset.d4)
    o=Tensor.empty(M*KVN,dtype=dtypes.float32,device=x.device); s=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device); w,vr,o,s=w.uop_program(vr,o,s,fxn=lambda *_,p=main:p); o,s=o.uop_program(s,fxn=lambda *_,p=fix:p); outs.append(o.reshape(M,KVN))
    return tuple(outs)
  def project_q6_v(self, x, words, *, model_family="qwen3_8b"):
    if not self.trace_epoch: raise RuntimeError("begin_trace must establish a capture-local epoch")
    if not supports(model_family=model_family,role="attn_v",weight_type="Q6_K",m=M,n=KVN,k=K,device=x.device) or x.dtype != dtypes.float16 or words.dtype != dtypes.uint16: raise ValueError("unsupported Q6 V route")
    self.cursor += 1; r=Tensor.empty(D4_RECORD_BYTES//4,dtype=dtypes.uint32,device=x.device); _,r=x.uop_program(r,fxn=lambda *_:self.asset.d4); o=Tensor.empty(M*KVN,dtype=dtypes.float32,device=x.device); s=Tensor.empty(SCRATCH_FLOATS,dtype=dtypes.float32,device=x.device); words,r,o,s=words.uop_program(r,o,s,fxn=lambda *_:self.asset.q6_main); o,s=o.uop_program(s,fxn=lambda *_:self.asset.q6_fix); return o.reshape(M,KVN)

def binding_for(device="NV"):
  if device != "NV": raise ValueError("QKV packed binding is NV-only")
  if device not in _BINDINGS: _BINDINGS[device] = QKVBinding.compile(Device[device])
  return _BINDINGS[device]
