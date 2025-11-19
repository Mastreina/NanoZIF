"""
Convex Spheropolyhedron geometry and overlap tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from .core import rotate_vec_by_quat

@dataclass
class ConvexSpheropolyhedron:
    base_vertices: NDArray[np.floating]  # shape (M, 3), in body frame
    sweep_radius: float
    # Derived geometry
    face_normals: NDArray[np.floating]  # (F, 3), unit outward normals (triangulated faces)
    edges: NDArray[np.floating]         # (E, 3), unoriented edge vectors (unit direction)
    faces: NDArray[np.int_]

    @classmethod
    def from_vertices(cls, vertices: NDArray[np.floating], sweep_radius: float) -> "ConvexSpheropolyhedron":
        # Build convex hull to get triangular faces and edges
        hull = ConvexHull(vertices)
        # Face normals from plane equations: n.x + d = 0, n points outward
        # scipy gives equations normalized s.t. ||n||=1
        face_normals = hull.equations[:, :3]

        # Collect unique edges from simplices
        edge_set = set()
        for tri in hull.simplices:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in ((i, j), (j, k), (k, i)):
                e = tuple(sorted((int(a), int(b))))
                edge_set.add(e)
        edges = []
        for a, b in edge_set:
            evec = vertices[b] - vertices[a]
            nrm = np.linalg.norm(evec)
            if nrm > 0:
                edges.append(evec / nrm)
        edges = np.array(edges, dtype=float)

        return cls(
            base_vertices=np.array(vertices, dtype=float),
            sweep_radius=float(sweep_radius),
            face_normals=face_normals,
            edges=edges,
            faces=hull.simplices.copy(),
        )

    # World-geometry utilities
    def rotated_vertices(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.stack([rotate_vec_by_quat(v, q) for v in self.base_vertices], axis=0)

    def world_vertices(self, q: NDArray[np.floating], pos: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.rotated_vertices(q) + pos[None, :]

    def world_face_normals(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.stack([rotate_vec_by_quat(n, q) for n in self.face_normals], axis=0)

    def world_edges(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.stack([rotate_vec_by_quat(e, q) for e in self.edges], axis=0)

    def world_faces(self, q: NDArray[np.floating], pos: NDArray[np.floating]) -> NDArray[np.floating]:
        verts = self.world_vertices(q, pos)
        return verts[self.faces]


def project_interval(points: NDArray[np.floating], axis: NDArray[np.floating]) -> Tuple[float, float]:
    vals = points @ axis
    return float(vals.min()), float(vals.max())


def sat_overlap_spheropolyhedra(
    shape_a: ConvexSpheropolyhedron,
    pos_a: NDArray[np.floating],
    quat_a: NDArray[np.floating],
    shape_b: ConvexSpheropolyhedron,
    pos_b: NDArray[np.floating],
    quat_b: NDArray[np.floating],
    eps: float = 1e-12,
) -> bool:
    """
    SAT test with rounding: treat as Minkowski-expanded by r_a + r_b.
    If any separating axis exists, return False (no overlap). Otherwise True.
    """
    r_total = shape_a.sweep_radius + shape_b.sweep_radius

    verts_a = shape_a.world_vertices(quat_a, pos_a)
    verts_b = shape_b.world_vertices(quat_b, pos_b)

    # Candidate axes: face normals of A and B, and cross of edges
    axes: List[NDArray[np.floating]] = []
    axes.extend(shape_a.world_face_normals(quat_a))
    axes.extend(shape_b.world_face_normals(quat_b))

    edges_a = shape_a.world_edges(quat_a)
    edges_b = shape_b.world_edges(quat_b)

    for ea in edges_a:
        for eb in edges_b:
            c = np.cross(ea, eb)
            nrm = np.linalg.norm(c)
            if nrm > 1e-14:
                axes.append(c / nrm)

    # Test all axes using separation distance with rounding
    for ax in axes:
        a_min, a_max = project_interval(verts_a, ax)
        b_min, b_max = project_interval(verts_b, ax)
        # separation distance between intervals along axis
        sep = max(0.0, max(a_min - b_max, b_min - a_max))
        if sep > (r_total + 1e-12):
            return False
    return True
