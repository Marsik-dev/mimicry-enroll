"""
Шифрование SSH private key через биометрический код НПБК.
Использует Grasshopper (Кузнечик) ГОСТ Р 34.12-2015 в режиме CTR.
"""
from __future__ import annotations

import hashlib
import os

import gostcrypto
import numpy as np

_KDF_ITERATIONS = 100_000
_KDF_DKLEN = 32
_CTR_IV_LEN = 8
_MAGIC = b"NPBK\x00"


def _derive_key(reference_code: np.ndarray, salt: bytes) -> bytes:
    code_bytes = np.packbits(reference_code.astype(np.uint8)).tobytes()
    return hashlib.pbkdf2_hmac("sha256", code_bytes, salt, _KDF_ITERATIONS, dklen=_KDF_DKLEN)


def encrypt_key(reference_code: np.ndarray, pem_bytes: bytes, salt: bytes) -> bytes:
    """Зашифровать SSH private key (PEM) кодом НПБК."""
    key = _derive_key(reference_code, salt)
    iv = os.urandom(_CTR_IV_LEN)
    plaintext = bytearray(_MAGIC + pem_bytes)
    cipher = gostcrypto.gostcipher.new(
        "kuznechik", bytearray(key), gostcrypto.gostcipher.MODE_CTR, init_vect=bytearray(iv)
    )
    return iv + bytes(cipher.encrypt(plaintext))


def decrypt_key(reference_code: np.ndarray, ciphertext_blob: bytes, salt: bytes) -> bytes:
    """Расшифровать SSH private key. Возбуждает ValueError при неверном коде."""
    if len(ciphertext_blob) <= _CTR_IV_LEN:
        raise ValueError("ciphertext_blob too short")
    iv = ciphertext_blob[:_CTR_IV_LEN]
    ciphertext = bytearray(ciphertext_blob[_CTR_IV_LEN:])
    key = _derive_key(reference_code, salt)
    cipher = gostcrypto.gostcipher.new(
        "kuznechik", bytearray(key), gostcrypto.gostcipher.MODE_CTR, init_vect=bytearray(iv)
    )
    plaintext = bytes(cipher.decrypt(ciphertext))
    if not plaintext.startswith(_MAGIC):
        raise ValueError("Biometric key mismatch — НПБК rejected")
    return plaintext[len(_MAGIC):]
