import pytest

from flsys.ledger import EvmLedger, SimulatedLedger

H1, H2 = "11" * 32, "22" * 32
C1, C2 = b"\x01" * 32, b"\x02" * 32


@pytest.fixture(params=["sim", "evm"])
def ledger(request):
    return SimulatedLedger() if request.param == "sim" else EvmLedger()


def test_submit_and_verify(ledger):
    assert ledger.submit_model_update(1, C1, 1, H1).ok
    assert ledger.verify_integrity(1, C1, H1)
    assert not ledger.verify_integrity(1, C1, H2)  # hash lain
    assert not ledger.verify_integrity(1, C2, H1)  # klien lain
    assert not ledger.verify_integrity(2, C1, H1)  # ronde lain


def test_duplicate_rejected(ledger):
    assert ledger.submit_model_update(1, C1, 1, H1).ok
    assert not ledger.submit_model_update(1, C1, 1, H2).ok
    assert ledger.verify_integrity(1, C1, H1)  # catatan asli tidak tertimpa


def test_zero_hash_rejected(ledger):
    assert not ledger.submit_model_update(1, C1, 1, "00" * 32).ok
    assert not ledger.submit_global_model(1, 1, "00" * 32).ok


def test_global_model(ledger):
    assert ledger.submit_global_model(3, 3, H1).ok
    assert ledger.verify_global(3, H1) and not ledger.verify_global(3, H2)
    assert not ledger.submit_global_model(3, 3, H2).ok


def test_audit_trail(ledger):
    ledger.submit_model_update(7, C1, 7, H1)
    ledger.submit_model_update(7, C2, 7, H2)
    tr = ledger.get_audit_trail(7)
    assert [t["client_id"] for t in tr] == [C1.hex(), C2.hex()]
    assert [t["hash"] for t in tr] == [H1, H2]
    assert ledger.get_audit_trail(8) == []


def test_evm_only_gateway_can_write():
    L = EvmLedger()
    other = L.w3.eth.accounts[1]
    with pytest.raises(Exception):
        L.c.functions.submitModelUpdate(1, C1, 1, bytes.fromhex(H1)).transact({"from": other})
    assert not L.verify_integrity(1, C1, H1)


def test_evm_reports_gas():
    L = EvmLedger()
    r = L.submit_model_update(1, C1, 1, H1)
    assert r.ok and r.gas and r.gas > 21000


def test_simulated_chain_detects_tampering():
    L = SimulatedLedger()
    L.submit_model_update(1, C1, 1, H1)
    L.submit_model_update(1, C2, 1, H2)
    assert L.chain_is_valid()
    L.chain[0]["payload"]["hash"] = H2
    assert not L.chain_is_valid()
