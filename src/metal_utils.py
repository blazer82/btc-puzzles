"""
Utilities for converting between Python data types and the data structures
required by the Metal kernels (e.g., 'ge' and 'gej' structs).
"""
import numpy as np
from coincurve.keys import PublicKey


def _int_to_limbs(value: int) -> np.ndarray:
    """Converts a 256-bit integer to a 5x52-bit limb array."""
    limbs = []
    for _ in range(5):
        limbs.append(value & 0xFFFFFFFFFFFFF)
        value >>= 52
    return np.array(limbs, dtype=np.uint64)


def point_to_ge_struct(point: PublicKey) -> np.ndarray:
    """Converts a PublicKey to the 'ge' (affine) struct format (11 ulongs)."""
    struct = np.zeros(11, dtype=np.uint64)
    if point is None:
        struct[10] = 1  # infinity
        return struct

    x, y = point.point()
    struct[0:5] = _int_to_limbs(x)
    struct[5:10] = _int_to_limbs(y)
    struct[10] = 0  # not infinity
    return struct


def point_to_gej_struct(point: PublicKey, z_int: int = 1) -> np.ndarray:
    """Converts a PublicKey to the 'gej' (Jacobian) struct format (16 ulongs)."""
    struct = np.zeros(16, dtype=np.uint64)
    if point is None:
        struct[15] = 1  # infinity
        return struct

    x, y = point.point()
    struct[0:5] = _int_to_limbs(x)
    struct[5:10] = _int_to_limbs(y)
    struct[10:15] = _int_to_limbs(z_int)
    struct[15] = 0  # not infinity
    return struct


def ge_struct_to_point(ge_struct: np.ndarray) -> (int, int):
    """Converts a 'ge' struct numpy array back to an (x, y) tuple."""
    x = 0
    y = 0
    # Reconstruct the 256-bit integers from the 5x52-bit limbs
    for i in range(4, -1, -1):
        x = (x << 52) | int(ge_struct[i])
        y = (y << 52) | int(ge_struct[i + 5])
    return x, y


def ge_struct_to_x_coord(ge_struct: np.ndarray) -> int:
    """Extracts the x-coordinate from a 'ge' struct numpy array."""
    x = 0
    # Reconstruct the 256-bit integer from the 5x52-bit limbs
    for i in range(4, -1, -1):
        x = (x << 52) | int(ge_struct[i])
    return x
