"""
Vertex generation functions for various polyhedra.
"""

from __future__ import annotations

import math
from itertools import product, permutations
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, HalfspaceIntersection

def cube_vertices(edge_length: float = 1.0) -> NDArray[np.floating]:
    """Generate vertices for a cube centered at origin."""
    pts = []
    half = edge_length / 2.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append((sx * half, sy * half, sz * half))
    return np.array(pts, dtype=float)

def octa_vertices(edge_length: float = 1.0) -> NDArray[np.floating]:
    """Generate vertices for an octahedron."""
    # Standard octahedron with vertices on axes at distance R has edge length sqrt(2)*R
    # If we want specific edge length, R = edge / sqrt(2)
    r = edge_length / math.sqrt(2.0)
    return np.array(
        [
            (r, 0.0, 0.0), (-r, 0.0, 0.0),
            (0.0, r, 0.0), (0.0, -r, 0.0),
            (0.0, 0.0, r), (0.0, 0.0, -r),
        ],
        dtype=float,
    )

def rhombic_dodecahedron_vertices(diameter: float = 1.0) -> NDArray[np.floating]:
    """
    Generate vertices for a rhombic dodecahedron.
    Diameter is the distance between opposite vertices (axis-aligned ones).
    """
    half = diameter / 2.0
    diag = half / 2.0
    verts = []
    # 6 vertices on axes
    for axis in range(3):
        for sign in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0]
            v[axis] = sign * half
            verts.append(tuple(v))
    # 8 vertices on diagonals
    for sx, sy, sz in product((-1.0, 1.0), repeat=3):
        verts.append((sx * diag, sy * diag, sz * diag))
    return np.array(verts, dtype=float)

def truncated_cube_vertices(edge_length: float, truncation_ratio: float) -> NDArray[np.floating]:
    """
    Generate vertices for a truncated cube.
    edge_length: The edge length of the resulting truncated cube? 
                 Or the original cube? 
                 Based on `truncated_cube_52.py`, it seems to be related to the final shape.
                 The logic uses permutations of (1, 1, ratio).
    """
    # Logic adapted from src/truncated_cube_52.py
    base_perms = {perm for perm in permutations((1.0, 1.0, truncation_ratio))}
    coords = set()
    for perm in base_perms:
        for sx, sy, sz in product((-1.0, 1.0), repeat=3):
            coords.add((sx * perm[0], sy * perm[1], sz * perm[2]))
    
    # Scale logic from original file: scale = edge_nm / sqrt(8.0)
    # This implies edge_length is the edge length of the *original* cube before truncation?
    # Or maybe the edge length of the triangular faces?
    # Let's keep the scaling consistent with the input parameter name being "edge_length"
    # as used in the original code context.
    scale = edge_length / math.sqrt(8.0)
    return np.array([(scale * x, scale * y, scale * z) for x, y, z in coords], dtype=float)

def truncated_tetrahedron_vertices(cube_length: float, truncation: float) -> NDArray[np.floating]:
    """
    Generate vertices for a truncated tetrahedron built from a tetrahedron
    embedded in a cube of side `cube_length`.
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

def truncated_rd_vertices(truncation: float = 0.68) -> NDArray[np.floating]:
    """
    Truncated Rhombic Dodecahedron.
    truncation = 0 -> Original RD.
    """
    trunc = float(np.clip(truncation, 0.0, 0.9))
    # Base RD with diameter 2.0 (radius 1.0) for normalized calculation
    base = rhombic_dodecahedron_vertices(diameter=2.0)
    hull = ConvexHull(base)
    halfspaces = []
    for eq in hull.equations:
        halfspaces.append(eq.copy())
    
    # Cut corners
    for v in base:
        r = float(np.linalg.norm(v))
        if r < 1e-9:
            continue
        u = v / r
        offset = r * (1.0 - trunc)
        # Plane equation: n.x + d = 0 -> n.x - offset = 0 -> d = -offset
        # But scipy HalfspaceIntersection expects Ax + b <= 0? 
        # No, HalfspaceIntersection takes equations [A; b] such that Ax + b <= 0.
        # We want to keep the interior.
        # If normal u points outward, then u.x <= offset -> u.x - offset <= 0.
        halfspaces.append(np.array([u[0], u[1], u[2], -offset], dtype=float))
        
    hs = np.vstack(halfspaces)
    hs_int = HalfspaceIntersection(hs, np.zeros(3, dtype=float))
    verts = hs_int.intersections
    
    # Clean up vertices using ConvexHull
    hull_trunc = ConvexHull(verts)
    verts = hull_trunc.points[hull_trunc.vertices]
    
    # Normalize to unit radius (max distance)
    verts /= np.linalg.norm(verts, axis=1).max()
    return verts
