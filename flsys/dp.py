"""Differential Privacy tingkat-klien: clip norma L2 lalu tambah noise Gaussian.

Update dari klien i: delta_i = w_i - w_global.
  1) clip:   delta_i <- delta_i * min(1, C / ||delta_i||_2)
  2) noise:  delta_i <- delta_i + N(0, (sigma * C)^2 I)
Dilakukan DI KLIEN sebelum serialisasi/hash/kirim.

Akuntansi privasi (accountant) memakai RDP mekanisme Gaussian:
  RDP orde a per ronde = a / (2 sigma^2)   (sensitivitas = C, adjacency add/remove-satu-klien)
  komposisi T ronde = T * a / (2 sigma^2)
  eps(delta) = min_a [ T*a/(2 sigma^2) + log(1/delta)/(a-1) ]
CATATAN: tanpa amplifikasi subsampling (semua klien ikut tiap ronde) -> batas konservatif.
Claim epsilon di naskah harus menyebut: pendekatan RDP, delta, sensitivitas=C, asumsi adjacency.
"""

from __future__ import annotations

import math

import numpy as np


def clip_update(delta: np.ndarray, clip_norm: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(delta))
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return (delta * scale).astype(np.float32), norm


def add_gaussian_noise(
    delta: np.ndarray, clip_norm: float, noise_multiplier: float, rng: np.random.Generator
) -> np.ndarray:
    if noise_multiplier <= 0:
        return delta
    noise = rng.normal(0.0, noise_multiplier * clip_norm, size=delta.shape)
    return (delta + noise).astype(np.float32)


def rdp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float:
    """Epsilon (eps, delta)-DP untuk `rounds` ronde Gaussian. inf jika sigma<=0."""
    if noise_multiplier <= 0:
        return math.inf
    best = math.inf
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))
    for a in orders:
        eps = rounds * a / (2 * noise_multiplier**2) + math.log(1 / delta) / (a - 1)
        best = min(best, eps)
    return best
