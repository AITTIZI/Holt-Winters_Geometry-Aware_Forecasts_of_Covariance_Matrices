"""Riemannian geometry of symmetric positive-definite matrices."""

from .airm import (
    airm_distance,
    airm_distance_sq,
    airm_exp,
    airm_geodesic,
    airm_inner,
    airm_log,
    airm_norm,
    airm_transport,
    le_distance,
    le_distance_sq,
    le_mean,
    normalized_coordinates,
)
from .euclidean import frobenius_distance
from .so3 import expm_so3, hat, is_rotation, orthogonality_error, random_rotation, vee
from .spd import (
    SPDViolationError,
    assert_spd,
    eigh,
    expm_sym,
    from_eig,
    invsqrtm_spd,
    is_spd,
    is_symmetric,
    logm_spd,
    positive_mask,
    powm_spd,
    REPAIR_FLOOR,
    random_spd,
    repair_spd,
    spd_factor,
    spectral_factor,
    sqrt_invsqrt_spd,
    sqrtm_spd,
    sym,
    transpose,
)
