"""Flat (Frobenius) geometry of the ambient space Sym(n)."""
from __future__ import annotations
import numpy as np


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    D = np.asarray(A, float) - np.asarray(B, float)
    return np.sqrt(np.sum(D * D, axis=(-2, -1)))
