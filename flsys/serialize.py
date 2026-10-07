"""Serialisasi KANONIK + hash. Harus identik byte-per-byte antara Python dan Dart.

Format: magic "FLM1" (4 byte ASCII) | n (uint32 little-endian) | n x float32 little-endian.
Urutan parameter = urutan flat model.py (W1, b1, W2, b2, W3, b3; W row-major).
Hash = SHA-256 atas seluruh byte tersebut.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

MAGIC = b"FLM1"


def to_canonical_bytes(flat: np.ndarray) -> bytes:
    a = np.ascontiguousarray(flat, dtype="<f4")
    return MAGIC + struct.pack("<I", a.size) + a.tobytes()


def from_canonical_bytes(b: bytes) -> np.ndarray:
    if len(b) < 8 or b[:4] != MAGIC:
        raise ValueError("Header payload tidak valid")
    (n,) = struct.unpack("<I", b[4:8])
    if len(b) != 8 + 4 * n:
        raise ValueError("Panjang payload tidak cocok dengan header")
    return np.frombuffer(b[8:], dtype="<f4").astype(np.float32)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_params(flat: np.ndarray) -> str:
    return sha256_hex(to_canonical_bytes(flat))
