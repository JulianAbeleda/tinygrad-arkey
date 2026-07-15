from extra.qk.mmq_q4_role_contracts import Q4_ROLES, Q4_ROLE_CONTRACTS, q4_role_contract, q4_role_matrix


def test_q4_role_matrix_has_independent_full_role_contracts():
  matrix = q4_role_matrix()
  assert tuple(matrix) == Q4_ROLES
  assert {tuple(row["shape"].values()) for row in matrix.values()} == {
    (512, 17408, 5120), (512, 5120, 17408), (512, 5120, 5120), (512, 1024, 5120)}
  assert len({row["candidate_identity"] for row in matrix.values()}) == 4


def test_each_role_contract_is_explicit_and_research_only():
  for role in Q4_ROLES:
    contract = q4_role_contract(role)
    assert contract.to_dict()["edge_axes"] == ["m", "n", "k"]
    assert contract.weight_layout == "q4_k_blocks_n_k"
    assert contract.activation_layout == "q8_1_ds4_m_k"
    assert contract.research_only and contract.route == "direct_packed"
    assert contract.candidate().descriptor.abi["role"] == role


def test_role_contracts_reject_route_promotion():
  contract = q4_role_contract("ffn_down")
  try:
    type(contract)(**(contract.to_dict() | {"shape": contract.shape, "route": "generated"}))
  except ValueError as exc:
    assert "direct-packed" in str(exc)
  else:
    raise AssertionError("promoted route accepted")
