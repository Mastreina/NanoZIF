"""
Core geometry and math utilities.
Includes vector normalization, quaternion operations, and sphere sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Vector Utilities
# -----------------------------------------------------------------------------

def normalize(vec: np.ndarray) -> np.ndarray:
    """Return the normalized vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        # For safety in some contexts, we might return vec or raise error.
        # Existing code behavior varies, but raising error is safer for pure math.
        # However, reversible_compression_mc returns vec if 0.
        # We will stick to returning vec if 0 to avoid breaking that logic,
        # or check if we can unify.
        # Let's follow the safer path: if it's 0, it stays 0 (no direction).
        return vec
    return vec / norm

def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Sample a random unit vector uniformly from the sphere."""
    # Method 1: Normal distribution
    vec = rng.normal(size=3)
    return normalize(vec)

def orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct an orthonormal basis (u, v, n) given a normal vector n."""
    n = normalize(normal)
    if abs(n[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    u = normalize(np.cross(helper, n))
    v = np.cross(n, u)
    return u, v, n

def rotate_basis(u: np.ndarray, v: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the tangent basis (u, v) around the normal by angle."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u_rot = cos_a * u + sin_a * v
    v_rot = -sin_a * u + cos_a * v
    return u_rot, v_rot

def gnomonic_project_point(
    point: np.ndarray,
    basis: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[float, float]:
    """Project a sphere point to the tangent plane (gnomonic projection)."""
    u, v, n = basis
    dot = float(np.dot(point, n))
    if dot <= 0:
        # Below horizon
        return math.inf, math.inf
    scale = 1.0 / dot
    sx = scale * float(np.dot(point, u))
    sy = scale * float(np.dot(point, v))
    return sx, sy

def lift_to_sphere(
    plane_xy: Tuple[float, float],
    basis: Tuple[np.ndarray, np.ndarray, np.ndarray],
    radius: float,
) -> np.ndarray:
    """Lift a point from the tangent plane back to the sphere."""
    sx, sy = plane_xy
    u, v, n = basis
    tangent_point = radius * n
    candidate = tangent_point + radius * (sx * u + sy * v)
    return radius * normalize(candidate)

# -----------------------------------------------------------------------------
# Quaternion Utilities
# -----------------------------------------------------------------------------

def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create a quaternion from a rotation axis and angle."""
    axis_n = normalize(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis_n[0] * s, axis_n[1] * s, axis_n[2] * s], dtype=float)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)

def quat_conj(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]

def quat_from_two_unit_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Minimal rotation quaternion mapping unit vector u -> v.
    Handles opposite vectors by picking an arbitrary perpendicular axis.
    """
    u = normalize(u)
    v = normalize(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-12:
        # 180-degree rotation: pick any axis perpendicular to u
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, u)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(u, axis))
        return quat_from_axis_angle(axis, np.pi)
    axis = normalize(np.cross(u, v))
    angle = np.arccos(dot)
    return quat_from_axis_angle(axis, angle)

# -----------------------------------------------------------------------------
# Sphere Sampling
# -----------------------------------------------------------------------------

def fibonacci_sphere(samples: int, radius: float = 1.0) -> np.ndarray:
    """Generate Fibonacci sphere sampling points."""
    samples = max(1, int(samples))
    indices = np.arange(samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1)
    return radius * points

def geodesic_sphere_points(subdiv_level: int) -> np.ndarray:
    """
    Generate (approx.) uniformly distributed points on unit sphere using
    recursive subdivision of an icosahedron.
    """
    level = max(0, int(subdiv_level))
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1,  phi), (0, 1,  phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    verts = [normalize(np.array(v, dtype=float)) for v in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    verts = list(verts)
    faces = [tuple(face) for face in faces]

    def midpoint(i: int, j: int, cache: Dict[Tuple[int, int], int]) -> int:
        key = tuple(sorted((i, j)))
        if key in cache:
            return cache[key]
        vi = verts[i]
        vj = verts[j]
        vm = normalize((vi + vj) * 0.5)
        verts.append(vm)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for _ in range(level):
        new_faces = []
        cache: Dict[Tuple[int, int], int] = {}
        for tri in faces:
            a = midpoint(tri[0], tri[1], cache)
            b = midpoint(tri[1], tri[2], cache)
            c = midpoint(tri[2], tri[0], cache)
            new_faces.extend([
                (tri[0], a, c),
                (tri[1], b, a),
                (tri[2], c, b),
                (a, b, c),
            ])
        faces = new_faces

    return np.array(verts, dtype=float)

@dataclass(frozen=True)
class SpherePoint:
    """Spherical coordinates description."""
    theta: float  # 0..pi
    phi: float    # -pi..pi

    @staticmethod
    def from_vector(vec: np.ndarray) -> "SpherePoint":
        vec_norm = normalize(vec)
        theta = math.acos(np.clip(vec_norm[2], -1.0, 1.0))
        phi = math.atan2(vec_norm[1], vec_norm[0])
        return SpherePoint(theta=theta, phi=phi)

    def to_vector(self, radius: float) -> np.ndarray:
        sin_t = math.sin(self.theta)
        return radius * np.array(
            [sin_t * math.cos(self.phi), sin_t * math.sin(self.phi), math.cos(self.theta)]
        )
