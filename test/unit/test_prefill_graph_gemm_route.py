import os
import pytest
from types import SimpleNamespace

from extra.qk import prefill_graph_gemm_route as route
from extra.qk import prefill_schedule_spec as spec
from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp


def _walk_uops(uop):
  if not isinstance(uop, UOp):
    return
  yield uop
  for child in uop.src:
    yield from _walk_uops(child)


def _prefill_schedule(route_family: str, out_f: int, in_f: int) -> spec.PrefillGEMMScheduleSpec:
  return spec.PrefillGEMMScheduleSpec(
    m=512, n=out_f, k=in_f, route_family=route_family, tile_m=128, tile_n=256, tile_k=32, waves_m=2, waves_n=2,
    wm=4, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=256, dbuf=1, plra=0, plrab=0, pad=16, leanaddr=0,
    role="ffn_down"
  )


def test_prefill_pipe_role_selective_zero_is_retired(monkeypatch):
  route._resolve_schedule.cache_clear()
  monkeypatch.setenv("PREFILL_PIPE_ROLE_SELECTIVE", "0")
  with pytest.raises(RuntimeError, match="PREFILL_PIPE_ROLE_SELECTIVE=0 global-pipe rollback was retired"):
    route._resolve_schedule(4096, 4096)
  route._resolve_schedule.cache_clear()


def test_resolve_schedule_uses_role_shape_pipe_exclusion_policy(monkeypatch):
  route._resolve_schedule.cache_clear()
  monkeypatch.delenv("PREFILL_GEMM_PIPELINE", raising=False)
  monkeypatch.delenv("PREFILL_GEMM_CFG_12288_4096", raising=False)
  monkeypatch.delenv("PREFILL_GEMM_CFG_4096_4096", raising=False)

  assert route._resolve_schedule(12288, 4096, "ffn_gate_up")["pipe_mode"] is False
  route._resolve_schedule.cache_clear()
  assert route._resolve_schedule(12288, 4096, "attn_qo")["pipe_mode"] is True
  route._resolve_schedule.cache_clear()
  assert route._resolve_schedule(12288, 4096)["pipe_mode"] is False
  route._resolve_schedule.cache_clear()


def test_route_pf16_graph_gemm_current_lowering_path_still_wraps_ops_ins(monkeypatch):
  route._resolve_schedule.cache_clear()
  captured = {}

  def fake_describe(out_f: int, in_f: int, role: str | None = None):
    captured["described"] = (out_f, in_f, role)
    return _prefill_schedule("pipe", out_f, in_f)

  def fake_emit(_spec):
    captured["emitted"] = _spec
    return (("ins",), 1, _spec.tile_m, _spec.tile_n, _spec.threads, _spec.kernel_name)

  class _Tensor:
    def __init__(self, shape, *, base_dtype=dtypes.float16):
      self.shape = shape
      self.device = "CPU"
      self.base = UOp.placeholder(shape, base_dtype, len(shape))
    @property
    def ndim(self):
      return len(self.shape)
    def __getitem__(self, _idx):
      return self
    def reshape(self, *shape):
      self.shape = shape
      return self
    def cast(self, *_args, **_kwargs):
      return self
    def contiguous(self):
      return self
    @classmethod
    def empty(cls, *shape, **_kwargs):
      return cls(shape, base_dtype=dtypes.float16)
    def custom_kernel(self, *args, **kwargs):
      program = kwargs["fxn"](self, *args)
      captured["program"] = program
      return (self, self, self)

  lin = SimpleNamespace(
    _pf16_w=_Tensor((4096, 2560), base_dtype=dtypes.float16),
    bias=None)
  x = _Tensor((1, 512, 2560), base_dtype=dtypes.float16)

  old_tensor = route.Tensor
  try:
    route.Tensor = _Tensor
    monkeypatch.setattr(spec, "describe_prefill_schedule", fake_describe)
    monkeypatch.setattr(spec, "emit_prefill_gemm_from_spec", fake_emit)
    out = route.route_pf16_graph_gemm(lin, x)
  finally:
    route.Tensor = old_tensor

  assert out is not None
  assert captured["described"] == (4096, 2560, getattr(lin, "_prefill_graph_role", None))
  assert captured["emitted"].route_family == "pipe"
  assert any(uop.op is Ops.INS for uop in _walk_uops(captured["program"]))


def test_route_pf16_graph_gemm_pipe_primitive_opt_in_uses_generated_matmul_transport(monkeypatch):
  route._resolve_schedule.cache_clear()
  captured = {}

  def fake_describe(out_f: int, in_f: int, role: str | None = None):
    captured["described"] = (out_f, in_f, role)
    return _prefill_schedule("pipe", out_f, in_f)

  class _Tensor:
    def __init__(self, shape, *, base_dtype=dtypes.float16, label="tensor"):
      self.shape = shape
      self.device = "CPU"
      self.label = label
      self.base = UOp.placeholder(shape, base_dtype, len(shape))
    @property
    def ndim(self):
      return len(self.shape)
    def reshape(self, *shape):
      captured.setdefault("reshape", []).append((self.label, shape))
      return _Tensor(shape, label=self.label)
    def cast(self, *_args, **_kwargs):
      captured.setdefault("cast", []).append(self.label)
      return self
    def contiguous(self):
      captured.setdefault("contiguous", []).append(self.label)
      return self
    def transpose(self):
      captured["transpose"] = self.label
      return _Tensor((self.shape[1], self.shape[0]), label=f"{self.label}.T")
    def __matmul__(self, other):
      captured["matmul"] = (self.label, other.label)
      return _Tensor((self.shape[0], other.shape[1]), label="matmul")
    @classmethod
    def empty(cls, *shape, **_kwargs):
      return cls(shape)
    def custom_kernel(self, *args, **kwargs):
      raise AssertionError("generated pipe transport must not use custom_kernel")

  lin = SimpleNamespace(_pf16_w=_Tensor((4096, 2560), base_dtype=dtypes.float16, label="w"), bias=None)
  x = _Tensor((1, 512, 2560), base_dtype=dtypes.float16, label="x")

  old_tensor = route.Tensor
  env_keys = ("AMD_ISA_WAITCNT_TARGETED", "AMD_ISA_WMMA_B128_FRAG", "AMD_ISA_REG_ACCUM", "PREFILL_WMMA_CHAIN_AB_RESIDENT")
  old_env = {k: os.environ.get(k) for k in env_keys}
  try:
    route.Tensor = _Tensor
    import tinygrad.codegen.opt.postrange as pr
    old_warmstart = pr._WARMSTART_OPTS
    old_local_stage_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_KEYS", None)
    old_local_stage_deny_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_DENY_KEYS", set()).copy()
    pr._WARMSTART_OPTS = {}
    pr._WARMSTART_LOCAL_STAGE_DENY_KEYS = set()
    monkeypatch.setenv("PREFILL_WMMA_PIPE_PRIMITIVE", "1")
    for k in env_keys:
      monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(spec, "describe_prefill_schedule", fake_describe)
    monkeypatch.setattr(spec, "emit_prefill_gemm_from_spec", lambda _spec: (_ for _ in ()).throw(AssertionError("raw emitter called")))
    out = route.route_pf16_graph_gemm(lin, x)
    captured["warmstart_keys"] = set(pr._WARMSTART_OPTS)
    captured["route_env"] = {k: os.environ.get(k) for k in env_keys}
  finally:
    pr._WARMSTART_OPTS = old_warmstart
    pr._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    for k, v in old_env.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    route.Tensor = old_tensor

  assert out is not None
  assert captured["described"] == (4096, 2560, getattr(lin, "_prefill_graph_role", None))
  assert captured["matmul"] == ("x", "w.T")
  assert out.shape == (1, 512, 4096)
  assert captured["route_env"]["AMD_ISA_WAITCNT_TARGETED"] == "1"
  assert captured["route_env"]["AMD_ISA_WMMA_B128_FRAG"] == "1"
  assert captured["route_env"]["AMD_ISA_REG_ACCUM"] == "1"
  assert captured["route_env"]["PREFILL_WMMA_CHAIN_AB_RESIDENT"] == "1"
  assert (frozenset({512, 4096}), 2560) in captured["warmstart_keys"]


def test_route_pf16_graph_gemm_pipe_primitive_resource_gate_falls_back_for_attn_kv(monkeypatch):
  route._resolve_schedule.cache_clear()
  captured = {}

  def fake_describe(out_f: int, in_f: int, role: str | None = None):
    captured["described"] = (out_f, in_f, role)
    return spec.PrefillGEMMScheduleSpec(
      m=512, n=out_f, k=in_f, route_family="pipe", tile_m=128, tile_n=64, tile_k=32, waves_m=4, waves_n=1,
      wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=128, dbuf=1, plra=0, plrab=1, pad=16,
      leanaddr=0, role="attn_kv")

  def fake_emit(params: dict, name: str):
    captured["raw_emit"] = (params, name)
    return (("raw-pipe-ins",), 1, params["bm"], params["bn"], params["threads"], name)

  class _Tensor:
    def __init__(self, shape, *, base_dtype=dtypes.float16, label="tensor"):
      self.shape = shape
      self.device = "CPU"
      self.label = label
      self.base = UOp.placeholder(shape, base_dtype, len(shape))
    @property
    def ndim(self):
      return len(self.shape)
    def reshape(self, *shape):
      self.shape = shape
      return self
    def cast(self, *_args, **_kwargs):
      return self
    def contiguous(self):
      return self
    def transpose(self):
      raise AssertionError("resource-gated pipe fallback must not use generated matmul transport")
    def __matmul__(self, other):
      raise AssertionError("resource-gated pipe fallback must not use generated matmul transport")
    @classmethod
    def empty(cls, *shape, **_kwargs):
      return cls(shape)
    def custom_kernel(self, *args, **kwargs):
      captured["program"] = kwargs["fxn"](self, *args)
      return (self, self, self)

  lin = SimpleNamespace(_pf16_w=_Tensor((1024, 4096), base_dtype=dtypes.float16, label="w"), bias=None,
                        _prefill_graph_role="attn_kv")
  x = _Tensor((1, 512, 4096), base_dtype=dtypes.float16, label="x")

  old_tensor = route.Tensor
  try:
    route.Tensor = _Tensor
    monkeypatch.setenv("PREFILL_WMMA_PIPE_PRIMITIVE", "1")
    monkeypatch.setenv("PREFILL_DBUF", "1")
    monkeypatch.setenv("PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE", "0")
    monkeypatch.setattr(spec, "describe_prefill_schedule", fake_describe)
    monkeypatch.setattr(route, "_emit_schedule", fake_emit)
    out = route.route_pf16_graph_gemm(lin, x)
  finally:
    route.Tensor = old_tensor

  assert out is not None
  assert captured["described"] == (1024, 4096, "attn_kv")
  assert captured["raw_emit"][0]["n"] == 1024
  assert lin._prefill_pipe_primitive_route == "pipe_resource_gated_raw_fallback"
  assert "69632 bytes LDS" in lin._prefill_pipe_primitive_fallback_reason
  assert any(uop.op is Ops.INS for uop in _walk_uops(captured["program"]))


def test_route_pf16_graph_gemm_pipe_primitive_attn_kv_uses_generated_no_local_stage(monkeypatch):
  route._resolve_schedule.cache_clear()
  captured = {}

  def fake_describe(out_f: int, in_f: int, role: str | None = None):
    captured["described"] = (out_f, in_f, role)
    return spec.PrefillGEMMScheduleSpec(
      m=512, n=out_f, k=in_f, route_family="pipe", tile_m=128, tile_n=64, tile_k=32, waves_m=4, waves_n=1,
      wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=128, dbuf=1, plra=0, plrab=1, pad=16,
      leanaddr=0, role="attn_kv")

  class _Tensor:
    def __init__(self, shape, *, base_dtype=dtypes.float16, label="tensor"):
      self.shape = shape
      self.device = "CPU"
      self.label = label
      self.base = UOp.placeholder(shape, base_dtype, len(shape))
    @property
    def ndim(self):
      return len(self.shape)
    def reshape(self, *shape):
      return _Tensor(shape, label=self.label)
    def cast(self, *_args, **_kwargs):
      return self
    def contiguous(self):
      return self
    def transpose(self):
      captured["transpose"] = self.label
      return _Tensor((self.shape[1], self.shape[0]), label=f"{self.label}.T")
    def __matmul__(self, other):
      captured["matmul"] = (self.label, other.label)
      return _Tensor((self.shape[0], other.shape[1]), label="matmul")
    @classmethod
    def empty(cls, *shape, **_kwargs):
      return cls(shape)
    def custom_kernel(self, *args, **kwargs):
      raise AssertionError("generated no-local-stage pipe transport must not use raw custom_kernel")

  lin = SimpleNamespace(_pf16_w=_Tensor((1024, 4096), base_dtype=dtypes.float16, label="w"), bias=None,
                        _prefill_graph_role="attn_kv")
  x = _Tensor((1, 512, 4096), base_dtype=dtypes.float16, label="x")

  old_tensor = route.Tensor
  try:
    route.Tensor = _Tensor
    import tinygrad.codegen.opt.postrange as pr
    old_warmstart = pr._WARMSTART_OPTS
    old_local_stage_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_KEYS", None)
    old_local_stage_deny_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_DENY_KEYS", set()).copy()
    pr._WARMSTART_OPTS = {}
    pr._WARMSTART_LOCAL_STAGE_DENY_KEYS = set()
    monkeypatch.setenv("PREFILL_WMMA_PIPE_PRIMITIVE", "1")
    monkeypatch.setenv("PREFILL_DBUF", "1")
    monkeypatch.setattr(spec, "describe_prefill_schedule", fake_describe)
    monkeypatch.setattr(spec, "emit_prefill_gemm_from_spec", lambda _spec: (_ for _ in ()).throw(AssertionError("raw emitter called")))
    out = route.route_pf16_graph_gemm(lin, x)
    captured["warmstart_keys"] = set(pr._WARMSTART_OPTS)
    captured["local_stage_keys"] = set(pr._WARMSTART_LOCAL_STAGE_KEYS)
    captured["local_stage_deny_keys"] = set(pr._WARMSTART_LOCAL_STAGE_DENY_KEYS)
  finally:
    pr._WARMSTART_OPTS = old_warmstart
    pr._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    pr._WARMSTART_LOCAL_STAGE_DENY_KEYS = old_local_stage_deny_keys
    route.Tensor = old_tensor

  assert out is not None
  assert captured["described"] == (1024, 4096, "attn_kv")
  assert captured["matmul"] == ("x", "w.T")
  assert lin._prefill_pipe_primitive_route == "generated_pipe_no_local_stage"
  assert (frozenset({512, 1024}), 4096) in captured["warmstart_keys"]
  assert (frozenset({512, 1024}), 4096) not in captured["local_stage_keys"]
  assert (frozenset({512, 1024}), 4096) in captured["local_stage_deny_keys"]


def test_attn_kv_no_local_stage_policy_denies_composed_local_stage_key(monkeypatch):
  import tinygrad.codegen.opt.postrange as pr
  monkeypatch.setenv("PREFILL_WMMA_PIPE_PRIMITIVE", "1")
  monkeypatch.setenv("PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE", "1")
  pr.getenv.cache_clear()

  assert pr._warmstart_attn_kv_no_local_stage_key((frozenset({512, 1024}), 4096)) is True
  assert pr._warmstart_attn_kv_no_local_stage_key((frozenset({2, 4, 8, 16, 1024}), 1)) is True
  assert pr._warmstart_pipe_primitive_no_local_stage_key((frozenset({512, 4096}), 12288)) is True
  assert pr._warmstart_pipe_primitive_no_local_stage_key((frozenset({2, 4, 8, 16, 4096}), 1)) is True
  assert pr._warmstart_pipe_primitive_no_local_stage_key((frozenset({512, 12288}), 4096)) is False
  pr.getenv.cache_clear()


def test_attn_kv_no_local_stage_policy_can_be_disabled(monkeypatch):
  import tinygrad.codegen.opt.postrange as pr
  monkeypatch.setenv("PREFILL_WMMA_PIPE_PRIMITIVE", "1")
  monkeypatch.setenv("PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE", "0")
  pr.getenv.cache_clear()

  assert pr._warmstart_attn_kv_no_local_stage_key((frozenset({512, 1024}), 4096)) is False
  assert pr._warmstart_attn_kv_no_local_stage_key((frozenset({2, 4, 8, 16, 1024}), 1)) is False
  assert pr._warmstart_pipe_primitive_no_local_stage_key((frozenset({2, 4, 8, 16, 4096}), 1)) is False
  pr.getenv.cache_clear()


def test_prefill_v2_covered_linears_are_role_tagged():
  from tinygrad.llm.model import Transformer

  class _Weight:
    def __init__(self, shape):
      self.shape = shape

  tr = object.__new__(Transformer)
  tr.blk = [SimpleNamespace(
    attn_q=SimpleNamespace(weight=_Weight((4096, 4096))),
    attn_k=SimpleNamespace(weight=_Weight((1024, 4096))),
    attn_v=SimpleNamespace(weight=_Weight((1024, 4096))),
    attn_output=SimpleNamespace(weight=_Weight((4096, 4096))),
    ffn_gate=SimpleNamespace(weight=_Weight((12288, 4096))),
    ffn_up=SimpleNamespace(weight=_Weight((12288, 4096))),
    ffn_down=SimpleNamespace(weight=_Weight((4096, 12288))),
  )]

  covered = {name: getattr(tr.blk[0], name) for name in tr._PREFILL_V2_LINEARS if hasattr(tr.blk[0], name)}
  list(tr._prefill_v2_covered())

  assert covered["attn_q"]._prefill_graph_role == "attn_qo"
  assert covered["attn_k"]._prefill_graph_role == "attn_kv"
  assert covered["attn_v"]._prefill_graph_role == "attn_kv"
  assert covered["attn_output"]._prefill_graph_role == "attn_qo"
  assert covered["ffn_gate"]._prefill_graph_role == "ffn_gate_up"
  assert covered["ffn_up"]._prefill_graph_role == "ffn_gate_up"
  assert covered["ffn_down"]._prefill_graph_role == "ffn_down"
  assert covered["attn_k"].name == "attn_k"

def test_prefill_lm_head_direct_flag_excludes_output_from_resident_realize():
  # PREFILL_LM_HEAD_DIRECT=1 must drop self.output from _prefill_v2_covered so no resident _pf16_w is realized
  # for it (w stays None -> route_prefill_linear dispatches the packed q6k kernel). Default off keeps it covered.
  import os
  from tinygrad.llm.model import Transformer
  from tinygrad.helpers import getenv

  class _Weight:
    def __init__(self, shape): self.shape = shape

  def build():
    tr = object.__new__(Transformer)
    tr.blk = [SimpleNamespace()]  # no per-block linears -> block loop yields nothing; isolate the lm_head branch
    tr.output = SimpleNamespace(weight=_Weight((151936, 4096)), q6k_storage=object(),
                                prefill_packed_weight=lambda: None, name="output.weight")
    return tr

  prev = os.environ.get("PREFILL_LM_HEAD_DIRECT")
  try:
    os.environ.pop("PREFILL_LM_HEAD_DIRECT", None); getenv.cache_clear()
    tr = build(); assert any(lin is tr.output for lin, _, _ in tr._prefill_v2_covered())   # off -> covered
    os.environ["PREFILL_LM_HEAD_DIRECT"] = "1"; getenv.cache_clear()
    tr = build(); assert not any(lin is tr.output for lin, _, _ in tr._prefill_v2_covered())  # on -> excluded
  finally:
    if prev is None: os.environ.pop("PREFILL_LM_HEAD_DIRECT", None)
    else: os.environ["PREFILL_LM_HEAD_DIRECT"] = prev
    getenv.cache_clear()

class _CandidateRouteTensor:
  def __init__(self,shape): self.shape=shape; self.device="CPU"
  @property
  def ndim(self): return len(self.shape)

@pytest.mark.parametrize(("role","shape"),(
  ("ffn_gate_up",(512,12288,4096)),("ffn_down",(512,4096,12288)),
  ("attn_qo",(512,4096,4096)),("attn_kv",(512,1024,4096))))
def test_candidate_set_exact_entry_overrides_role_policy(monkeypatch,role,shape):
  identity=(role.encode().hex()+"0"*64)[:64]; admission=SimpleNamespace(canonical_identity=identity)
  class Registry:
    def get(self,profile,selected_role,selected_shape,target):
      assert profile == "qwen3_8b_q4k_m_gfx1100" and target["arch"] == "gfx1100"
      return admission if (selected_role,selected_shape) == (role,shape) else None
  monkeypatch.setattr(route,"_candidate_registry_from_env",lambda:Registry())
  monkeypatch.setattr(route,"_install_candidate_matmul",lambda x,w,n,k,spec,selected:(selected.canonical_identity,spec.route_family))
  monkeypatch.setattr(spec,"describe_prefill_schedule",lambda n,k,role=None:_prefill_schedule("pipe",n,k))
  lin=SimpleNamespace(_prefill_graph_role=role,bias=None)
  out=route.route_pf16_graph_gemm(lin,_CandidateRouteTensor((1,shape[0],shape[2])),w=_CandidateRouteTensor((shape[1],shape[2])))
  assert out == (identity,"pipe") and lin._prefill_full_kernel_candidate_identity == identity

def test_candidate_set_missing_exact_role_preserves_existing_emitter(monkeypatch):
  class Registry:
    def get(self,*_args): return None
  monkeypatch.setattr(route,"_candidate_registry_from_env",lambda:Registry())
  monkeypatch.setattr(spec,"describe_prefill_schedule",lambda n,k,role=None:_prefill_schedule("pipe",n,k))
  marker=object(); monkeypatch.setattr(spec,"emit_prefill_gemm_from_spec",lambda _spec:None)
  lin=SimpleNamespace(_prefill_graph_role="attn_qo",bias=None)
  assert route.route_pf16_graph_gemm(lin,_CandidateRouteTensor((1,512,4096)),w=_CandidateRouteTensor((2048,4096))) is None
  assert not hasattr(lin,"_prefill_full_kernel_candidate_identity")

def test_four_role_candidate_specs_have_distinct_warmstart_keys():
  rows=(("ffn_gate_up",12288,4096),("ffn_down",4096,12288),("attn_qo",4096,4096),("attn_kv",1024,4096))
  keys={route._primitive_warmstart_key(_prefill_schedule("pipe",n,k)) for _role,n,k in rows}
  assert len(keys) == 4

def _census_admission(role,n,k,identity):
  return SimpleNamespace(canonical_identity=identity,normalized_payload={"workload":{
    "profile":"qwen3_8b_q4k_m_gfx1100","role":role,"shape":{"m":512,"n":n,"k":k},
    "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}}})

def test_candidate_route_census_requires_actual_exact_bindings_and_counts_reuse():
  admissions=tuple(_census_admission(role,n,k,str(i)*64) for i,(role,n,k) in enumerate((
    ("ffn_gate_up",12288,4096),("ffn_down",4096,12288),("attn_qo",4096,4096),("attn_kv",1024,4096)),1))
  entries=tuple(SimpleNamespace(exact_key=("qwen3_8b_q4k_m_gfx1100",a.normalized_payload["workload"]["role"],512,
    a.normalized_payload["workload"]["shape"]["n"],a.normalized_payload["workload"]["shape"]["k"],"AMD","gfx1100",32)) for a in admissions)
  registry=SimpleNamespace(candidate_set=SimpleNamespace(entries=entries),admissions=admissions)
  with route.candidate_route_census() as collector:
    for admission in admissions: route._record_candidate_route(admission)
    route._record_candidate_route(admissions[2])
  report=route.finalize_candidate_route_census(collector,registry)
  assert report["passed"] and report["selected_entry_count"] == report["expected_entry_count"] == 4
  assert next(x for x in report["selected"] if x["role"] == "attn_qo")["bindings"] == 2
  with route.candidate_route_census() as incomplete: route._record_candidate_route(admissions[0])
  failed=route.finalize_candidate_route_census(incomplete,registry)
  assert not failed["passed"] and len(failed["missing"]) == 3

def test_candidate_route_census_context_does_not_leak_between_runs():
  admission=_census_admission("attn_kv",1024,4096,"a"*64)
  with route.candidate_route_census() as first: route._record_candidate_route(admission)
  with route.candidate_route_census() as second: pass
  route._record_candidate_route(admission)
  assert len(first["selected"]) == 1 and second["selected"] == {}


def test_route_pf16_graph_gemm_lds_primitive_opt_in_uses_existing_generated_lds_transport(monkeypatch):
  route._resolve_schedule.cache_clear()
  captured = {}

  def fake_describe(out_f: int, in_f: int, role: str | None = None):
    captured["described"] = (out_f, in_f, role)
    return spec.PrefillGEMMScheduleSpec(
      m=512, n=out_f, k=in_f, route_family="lds", tile_m=128, tile_n=128, tile_k=32,
      waves_m=4, waves_n=2, wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=256,
      dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0, role="ffn_gate_up")

  class _Tensor:
    def __init__(self, shape, *, base_dtype=dtypes.float16, label="tensor"):
      self.shape = shape
      self.device = "CPU"
      self.label = label
      self.base = UOp.placeholder(shape, base_dtype, len(shape))
    @property
    def ndim(self):
      return len(self.shape)
    def reshape(self, *shape):
      captured.setdefault("reshape", []).append((self.label, shape))
      return _Tensor(shape, label=self.label)
    def cast(self, *_args, **_kwargs):
      captured.setdefault("cast", []).append(self.label)
      return self
    def contiguous(self):
      captured.setdefault("contiguous", []).append(self.label)
      return self
    def transpose(self):
      captured["transpose"] = self.label
      return _Tensor((self.shape[1], self.shape[0]), label=f"{self.label}.T")
    def __matmul__(self, other):
      captured["matmul"] = (self.label, other.label)
      return _Tensor((self.shape[0], other.shape[1]), label="matmul")
    @classmethod
    def empty(cls, *shape, **_kwargs):
      return cls(shape)
    def custom_kernel(self, *args, **kwargs):
      raise AssertionError("generated LDS transport must not use custom_kernel")

  lin = SimpleNamespace(_pf16_w=_Tensor((12288, 4096), base_dtype=dtypes.float16, label="w"), bias=None,
                        _prefill_graph_role="ffn_gate_up")
  x = _Tensor((1, 512, 4096), base_dtype=dtypes.float16, label="x")

  old_tensor = route.Tensor
  env_keys = ("AMD_ISA_WMMA_B128_FRAG", "AMD_ISA_REG_ACCUM", "PREFILL_TC_LOCAL_STAGE",
              "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY",
              "PREFILL_LDS_PACK_WITHLOCAL_B128", "PREFILL_DBUF")
  old_env = {k: os.environ.get(k) for k in env_keys}
  try:
    route.Tensor = _Tensor
    import tinygrad.codegen.opt.postrange as pr
    old_warmstart = pr._WARMSTART_OPTS
    old_local_stage_keys = getattr(pr, "_WARMSTART_LOCAL_STAGE_KEYS", None)
    pr._WARMSTART_OPTS = {}
    monkeypatch.setenv("PREFILL_WMMA_LDS_PRIMITIVE", "1")
    for k in env_keys:
      monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(spec, "describe_prefill_schedule", fake_describe)
    monkeypatch.setattr(spec, "emit_prefill_gemm_from_spec", lambda _spec: (_ for _ in ()).throw(AssertionError("raw emitter called")))
    out = route.route_pf16_graph_gemm(lin, x)
    captured["warmstart"] = pr._WARMSTART_OPTS
    captured["route_env"] = {k: os.environ.get(k) for k in env_keys}
  finally:
    pr._WARMSTART_OPTS = old_warmstart
    pr._WARMSTART_LOCAL_STAGE_KEYS = old_local_stage_keys
    for k, v in old_env.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    route.Tensor = old_tensor

  assert out is not None
  assert captured["described"] == (12288, 4096, "ffn_gate_up")
  assert captured["matmul"] == ("x", "w.T")
  assert out.shape == (1, 512, 12288)
  assert captured["route_env"]["AMD_ISA_WMMA_B128_FRAG"] == "1"
  assert captured["route_env"]["AMD_ISA_REG_ACCUM"] == "1"
  assert captured["route_env"]["PREFILL_TC_LOCAL_STAGE"] == "both"
  assert captured["route_env"]["PREFILL_TC_LOCAL_STAGE_WITH_LOCAL"] == "1"
  assert captured["route_env"]["PREFILL_TC_LOCAL_STAGE_B_TILEKEY"] == "1"
  assert captured["route_env"]["PREFILL_LDS_PACK_WITHLOCAL_B128"] == "1"
  assert captured["route_env"]["PREFILL_DBUF"] is None
  opts = captured["warmstart"][(frozenset({512, 12288}), 4096)]
  assert [o.op.name for o in opts] == ["TC", "UPCAST", "UPCAST", "UNROLL"]
  assert opts[1].arg == 2
  assert opts[2].arg == 4


def test_s10_lds_primitive_route_trace_does_not_silently_classify_as_raw_oracle(monkeypatch):
  route._resolve_schedule.cache_clear()
  monkeypatch.setenv("PREFILL_WMMA_LDS_PRIMITIVE", "1")

  trace = route.prefill_lds_primitive_route_trace(12288, 4096, role="ffn_gate_up")

  assert trace["schema"] == "prefill-s10-lds-route-trace.v1"
  assert trace["role"] == "ffn_gate_up"
  assert trace["route_family"] == "lds"
  assert trace["schedule_spec"]["route_family"] == "lds"
  assert trace["lds_spec"]["n"] == 12288
  assert trace["lds_spec"]["k"] == 4096
  assert trace["selected_surface"] == "generated_transport"
  assert trace["fallback_reason"] is None
  assert trace["classification"] == "compiler_primitive_spec_owned__generated_transport"
  assert trace["classification"] != "legacy_raw_oracle"
  assert trace["calls_build_gemm_lds2"] is False
  assert trace["build_gemm_lds2_called"] is False

  fallback = route.prefill_lds_primitive_route_trace(12288, 4096, role="ffn_gate_up", primitive_opt_in=False)
  assert fallback["selected_surface"] == "fallback_raw_oracle"
  assert fallback["fallback_reason"] == "PREFILL_WMMA_LDS_PRIMITIVE not enabled"
  assert fallback["classification"] == "legacy_raw_oracle"
  assert fallback["calls_build_gemm_lds2"] is True
