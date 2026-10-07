import math

import numpy as np
import pytest

from flsys import dp
from flsys import model as M
from flsys.identity import ClientIdentity, verify_signature
from flsys.serialize import from_canonical_bytes, hash_params, to_canonical_bytes


def test_roundtrip_and_determinism():
    w = M.init_params(3)
    b = to_canonical_bytes(w)
    assert len(b) == 8 + 4 * 1057 and b[:4] == b"FLM1"
    assert np.array_equal(from_canonical_bytes(b), w)
    assert hash_params(w) == hash_params(w.copy())


def test_hash_sensitive_to_one_bit():
    w = M.init_params(3)
    b = bytearray(to_canonical_bytes(w))
    b[100] ^= 1
    from flsys.serialize import sha256_hex

    assert sha256_hex(bytes(b)) != hash_params(w)


def test_corrupt_payload_rejected_by_decoder():
    b = to_canonical_bytes(M.init_params(0))
    with pytest.raises(ValueError):
        from_canonical_bytes(b[:-4])
    with pytest.raises(ValueError):
        from_canonical_bytes(b"XXXX" + b[4:])


def test_golden_vector_for_dart():
    """Vektor emas: implementasi Dart harus menghasilkan hash yang sama persis."""
    w = np.arange(1057, dtype=np.float32) / 1057.0 - 0.5
    h = hash_params(w)
    assert len(h) == 64
    import json
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "results" / "golden_hash.json"
    p.write_text(
        json.dumps({"rule": "w[i] = i/1057 - 0.5 (float32), i=0..1056", "sha256": h}, indent=2)
    )


def test_clip_and_noise():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 1057).astype(np.float32)
    c, n0 = dp.clip_update(d, 1.0)
    assert np.linalg.norm(c) <= 1.0 + 1e-5 and n0 > 1.0
    small = (d / (10 * np.linalg.norm(d))).astype(np.float32)
    assert np.allclose(dp.clip_update(small, 1.0)[0], small)  # tidak diubah bila norma < C
    noisy = np.stack(
        [
            dp.add_gaussian_noise(np.zeros(1057, np.float32), 2.0, 0.5, np.random.default_rng(i))
            for i in range(200)
        ]
    )
    assert abs(noisy.std() - 1.0) < 0.05  # sigma*C = 1.0


def test_epsilon_monotonic():
    assert dp.rdp_epsilon(0, 10) == math.inf
    e1, e2 = dp.rdp_epsilon(2.0, 30), dp.rdp_epsilon(1.0, 30)
    assert e1 < e2  # noise lebih besar -> eps lebih kecil
    assert dp.rdp_epsilon(1.0, 10) < dp.rdp_epsilon(
        1.0, 50
    )  # ronde lebih banyak -> eps lebih besar


def test_signature():
    a, b = ClientIdentity(b"a"), ClientIdentity(b"b")
    h = "ab" * 32
    s = a.sign(5, h)
    assert verify_signature(a.public_key, 5, h, s)
    assert not verify_signature(a.public_key, 6, h, s)  # ronde lain
    assert not verify_signature(a.public_key, 5, "cd" * 32, s)  # hash lain
    assert not verify_signature(b.public_key, 5, h, s)  # kunci lain
    assert len(a.client_id) == 32
