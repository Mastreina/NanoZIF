"""
Geometry helpers for rounded truncated tetrahedron (RTT).

This file reproduces and slightly generalizes the vertex and sweep-radius
utilities from the upstream floppy-box script, without importing the script
itself (to avoid side effects). Keep third-party/upstream code separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def truncated_tetrahedron_vertices(cube_length: float, truncation: float) -> np.ndarray:
    """
    Generate vertices for a truncated tetrahedron built from a tetrahedron
    embedded in a cube of side `cube_length` and truncation factor `truncation`.

    This mirrors the math used in the upstream `fbmc.py`.
    Returns an array of shape (12, 3) for truncation > 0, or (4, 3) if == 0.
    """
    original_vertices = np.array(
        [
            (0.5 * cube_length, -0.5 * cube_length, -0.5 * cube_length),
            (-0.5 * cube_length, 0.5 * cube_length, -0.5 * cube_length),
            (-0.5 * cube_length, -0.5 * cube_length, 0.5 * cube_length),
            (0.5 * cube_length, 0.5 * cube_length, 0.5 * cube_length),
        ],
        dtype=float,
    )

    if truncation == 0:
        return original_vertices.copy()

    vertices = []
    for i in range(len(original_vertices)):
        for j in range(len(original_vertices)):
            if i != j:
                v = original_vertices[i] + 0.5 * truncation * (
                    original_vertices[j] - original_vertices[i]
                )
                vertices.append(v)
    return np.array(vertices, dtype=float)


def sweep_radius(cube_length: float, roundness: float) -> float:
    """
    Calculate the sweep radius for a rounded tetrahedron consistent with upstream.
    roundness in (0, 1). Larger -> more rounding.
    """
    return (np.sqrt(3.0) / 2.0) * cube_length * roundness / (1.0 - roundness)


@dataclass
class RTTParams:
    cube_length: float
    truncation: float
    roundness: float

    @property
    def edge_length(self) -> float:
        return (2.0 ** 0.5) * self.cube_length

    @property
    def vertices(self) -> np.ndarray:
        return truncated_tetrahedron_vertices(self.cube_length, self.truncation)

    @property
    def r_sweep(self) -> float:
        return sweep_radius(self.cube_length, self.roundness)
