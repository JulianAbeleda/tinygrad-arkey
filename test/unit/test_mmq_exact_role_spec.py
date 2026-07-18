import json

import pytest

from extra.qk.mmq_exact_role_spec import (
  DEFAULT_INVENTORY, exact_role_spec, exact_role_spec_from_shape, load_exact_role_specs,
)


def test_inventory_admitted_exact_roles_derive_program_geometry_grid_and_epochs():
  rows = {row.role: row for row in load_exact_role_specs()}
  assert set(rows) == {"attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"}
  expected = {
    "attn_kv": ((512, 1024, 5120), (512, 1024, 256), (8, 4, 1), 20),
    "attn_qo": ((512, 5120, 5120), (512, 5120, 256), (40, 4, 1), 20),
    "ffn_down": ((512, 5120, 17408), (512, 5120, 256), (40, 4, 1), 68),
    "ffn_gate_up": ((512, 17408, 5120), (512, 17408, 256), (136, 4, 1), 20),
  }
  for role, (shape, program_shape, grid, epochs) in expected.items():
    assert rows[role].shape == shape
    assert rows[role].program.shape == program_shape
    assert rows[role].program.grid == grid
    assert rows[role].epochs == epochs
  assert rows["attn_qo"].program == rows["ffn_down"].program


def test_exact_role_lookup_rejects_role_shape_or_inventory_mismatch():
  with pytest.raises(ValueError, match="differs from admitted"):
    exact_role_spec("attn_kv", shape=(512, 5120, 5120))
  with pytest.raises(ValueError, match="expected one admitted"):
    exact_role_spec("made_up_role")
  artifact = json.loads(DEFAULT_INVENTORY.read_text())
  artifact["bindings"] = artifact["bindings"][1:]
  with pytest.raises(ValueError, match="lacks an exact binding"):
    load_exact_role_specs(artifact)
  with pytest.raises(ValueError, match="expected one admitted"):
    exact_role_spec_from_shape((512, 5120, 256))
