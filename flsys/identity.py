"""Identitas klien: keypair Ed25519. client_id = SHA-256(public key) (pseudonim).
Klien menandatangani (round_id || hash); gateway memverifikasi sebelum menulis ke chain.
Di aplikasi nyata kunci disimpan di Android Keystore; di sini disimulasikan."""

from __future__ import annotations

import hashlib
import struct

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _msg(round_id: int, hash_hex: str) -> bytes:
    return struct.pack(">Q", round_id) + bytes.fromhex(hash_hex)


class ClientIdentity:
    def __init__(self, seed: bytes | None = None):
        if seed is not None:
            self._sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed).digest())
        else:
            self._sk = Ed25519PrivateKey.generate()
        self.public_key: bytes = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.client_id: bytes = hashlib.sha256(self.public_key).digest()  # 32 byte

    def sign(self, round_id: int, hash_hex: str) -> bytes:
        return self._sk.sign(_msg(round_id, hash_hex))


def verify_signature(public_key: bytes, round_id: int, hash_hex: str, sig: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(sig, _msg(round_id, hash_hex))
        return True
    except (InvalidSignature, ValueError):
        return False
