"""
Geometry helpers for rounded truncated tetrahedron (RTT).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .shapes import truncated_tetrahedron_vertices

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
    def vertices(self) -> NDArray[np.floating]:
        return truncated_tetrahedron_vertices(self.cube_length, self.truncation)

    @property
    def r_sweep(self) -> float:
        return sweep_radius(self.cube_length, self.roundness)
