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
