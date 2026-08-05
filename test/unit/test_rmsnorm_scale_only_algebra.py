import numpy as np


def _rms_affine(x, weight, eps=1e-6):
  scale=np.float32(1.0)/np.sqrt(np.mean(x.astype(np.float32)**2,dtype=np.float32)+np.float32(eps))
  # This is the production fp16 round point: it cannot be moved after GEMV.
  return ((x.astype(np.float32)*scale).astype(np.float16)*weight.astype(np.float16)).astype(np.float16), scale


def test_scale_only_consumer_must_apply_per_element_affine_before_q4_dot():
  rng=np.random.default_rng(20260805)
  x=rng.normal(0,.2,4096).astype(np.float16)
  norm_weight=rng.normal(1,.1,4096).astype(np.float16)
  row=rng.normal(0,.05,4096).astype(np.float16)
  normalized,scale=_rms_affine(x,norm_weight)
  reference=np.dot(row.astype(np.float32),normalized.astype(np.float32))
  # A scalar-only post-dot factor is invalid because affine RMSNorm has a
  # nonuniform weight and an fp16 round point before the consumer.
  invalid=np.dot(row.astype(np.float32),x.astype(np.float32))*scale
  assert abs(float(reference-invalid)) > 1e-3
  raw_affine=((x.astype(np.float32)*scale).astype(np.float16)*norm_weight).astype(np.float16)
  np.testing.assert_array_equal(raw_affine,normalized)
