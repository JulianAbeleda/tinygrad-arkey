import json
from pathlib import Path

LEDGER = json.loads(Path(__file__).with_name("dense_tensor_accounting_ledger.json").read_text())

def test_fixed_block_divisibility_and_bytes():
    for t in LEDGER["tensors"]:
        assert t["cols"] % 256 == 0
        for r in t["candidates"].values():
            assert r["payload_bytes"] % 1 == 0
            assert r["padded_bytes"] >= r["payload_bytes"]
            assert r["padded_bytes"] % 256 == 0

def test_r2_is_lower_than_r1_for_every_tensor():
    for t in LEDGER["tensors"]:
        assert t["candidates"]["R2"]["payload_bytes"] < t["candidates"]["R1"]["payload_bytes"]

def test_tied_weights_are_not_silently_aliased():
    names = {t["tensor_name"] for t in LEDGER["tensors"]}
    assert "token_embd.weight" in names and "output.weight" in names

def test_roles_are_split_and_have_fixtures():
    assert set(LEDGER["representative_tensors"]) == {"Q","K","V","O","gate","up","down","vocab"}

def test_source_block_arithmetic():
    for t in LEDGER["tensors"]:
        block = 144 if t["source_quant"] == "Q4_K" else 210
        assert t["source_payload_bytes"] == t["rows"] * (t["cols"] // 256) * block

def test_streamed_excludes_embedding():
    total = sum(t["candidates"]["R1"]["padded_bytes"] for t in LEDGER["tensors"] if t["role"] != "embedding")
    assert LEDGER["streamed_projection_aggregate"]["R1"]["padded_bytes"] == total

def test_source_stream_and_population_exposure_are_separate():
    streamed = [t for t in LEDGER["tensors"] if t["role"] != "embedding"]
    assert LEDGER["source_streamed_projection_aggregate"]["padded_bytes"] == sum(t["source_padded_bytes"] for t in streamed)
    assert LEDGER["population_exposure"]["R2"]["tensor_count"] == sum(t["source_quant"] == "Q4_K" for t in streamed)
    assert LEDGER["population_exposure"]["R3"]["tensor_count"] == sum(t["source_quant"] == "Q6_K" for t in streamed)
    for exposure in LEDGER["population_exposure"].values():
        assert exposure["mixed_streamed_padded_bytes"] == (LEDGER["source_streamed_projection_aggregate"]["padded_bytes"] +
          exposure["candidate_minus_source_bytes"])
