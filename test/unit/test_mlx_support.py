from tinygrad.runtime.autogen import mlx5
from tinygrad.runtime.support.mlx.mlxdev import MLXDev, MLXQP, ifc_get, ifc_set


def test_mlx_runtime_package_and_generated_binding_are_present():
  assert MLXDev.__name__ == "MLXDev" and MLXQP.__name__ == "MLXQP"
  assert mlx5.MLX5_CMD_OP_ENABLE_HCA > 0


def test_mlx_ifc_codec_round_trips_unaligned_big_endian_bits():
  payload = bytearray(4)
  ifc_set(payload, 5, 13, 0x1234)
  assert ifc_get(payload, 5, 13) == 0x1234
