"""
可逆压缩 MC：在球面上模拟凸体（如圆角截角四面体）的单层压缩过程。

思路：粒子中心限制在半径 R 的球面上，通过切平面平移 + 小幅转动尝试移动，
并周期性缩小 R；若出现重叠则拒绝，从而逐步逼近高密度状态。

Notes
- This is an initial, research-grade prototype for geometry + MC mechanics.
- Overlap test uses SAT (separating axis theorem) on convex polyhedra and accounts
  for sweep radii (Minkowski sum with a ball). Geometry is computed from a convex
  hull of provided vertices (requires scipy).
- For performance (large N), you should add neighbor lists / cell lists on the
  sphere; here we keep O(N^2) checks for simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Callable, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from .geometry_rtt import RTTParams


# -----------------------------
# Quaternion / rotation helpers
# -----------------------------

def normalize(v: NDArray[np.floating]) -> NDArray[np.floating]:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def quat_from_axis_angle(axis: NDArray[np.floating], angle: float) -> NDArray[np.floating]:
    axis_n = normalize(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis_n[0] * s, axis_n[1] * s, axis_n[2] * s], dtype=float)


def quat_mul(q1: NDArray[np.floating], q2: NDArray[np.floating]) -> NDArray[np.floating]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_conj(q: NDArray[np.floating]) -> NDArray[np.floating]:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def rotate_vec_by_quat(v: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.floating]:
    # Convert v to quaternion (0, v) and compute q * v * q_conj
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]


def quat_from_two_unit_vectors(u: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
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


def geodesic_sphere_points(subdiv_level: int) -> NDArray[np.floating]:
    """
    Generate (approx.) uniformly distributed points on unit sphere using
    recursive subdivision of an icosahedron.
    Returns array shape (M, 3).
    """
    level = max(0, int(subdiv_level))
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1,  phi, 0),
        (1,  phi, 0),
        (-1, -phi, 0),
        (1, -phi, 0),
        (0, -1,  phi),
        (0, 1,  phi),
        (0, -1, -phi),
        (0, 1, -phi),
        (phi, 0, -1),
        (phi, 0, 1),
        (-phi, 0, -1),
        (-phi, 0, 1),
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

# ------------------------------------
# Convex spheropolyhedron (poly + sweep)
# ------------------------------------

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


# -----------------------------
# Particle and MC configuration
# -----------------------------

@dataclass
class Particle:
    pos: NDArray[np.floating]  # on sphere radius R
    quat: NDArray[np.floating] # [w, x, y, z]


@dataclass
class ReversibleCompressionConfig:
    radius: float
    translate_sigma: float  # typical arc length on sphere (in units of length)
    rotate_sigma: float     # typical rotation angle (rad)
    shrink_factor: float    # multiplicative radius shrink per attempt (e.g., 0.999)
    shrink_interval: int    # MC steps between shrink attempts
    max_shrink_fail: int    # stop after this many failed shrinks
    # Optional orientation constraint: align a body-frame vector to radial direction
    align_body_vec: Optional[Tuple[float, float, float]] = None
    restrict_orientation_azimuthal: bool = True  # if align enabled, only rotate about radial
    # Initialization / relaxation controls
    init_method: str = "fibonacci"  # or "geodesic"
    init_geodesic_level: int = 2
    init_jitter: float = 0.0
    init_relax_steps: int = 0
    block_relax_size: int = 0
    block_relax_attempts: int = 3
    # Annealing of proposal steps / shrink factor
    translate_sigma_final: Optional[float] = None
    rotate_sigma_final: Optional[float] = None
    shrink_factor_final: Optional[float] = None
    anneal_steps: int = 0
    # Order-parameter guided acceptance
    target_neighbor_angle_deg: Optional[float] = None
    order_strength: float = 0.0
    order_k_neighbors: int = 6


class ReversibleCompressionMC:
    def __init__(
        self,
        shape: ConvexSpheropolyhedron,
        config: ReversibleCompressionConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.shape = shape
        self.cfg = config
        self.rng = np.random.default_rng() if rng is None else rng
        self.particles: List[Particle] = []
        self.accepted_moves = 0
        self.total_moves = 0
        self.shrink_fail_count = 0
        self._translate_sigma0 = config.translate_sigma
        self._rotate_sigma0 = config.rotate_sigma
        self._shrink_factor0 = config.shrink_factor
        self._footprint_angle = 0.0
        self._update_footprint_angle()

    def initialize_on_sphere(
        self,
        n_particles: int,
        method: Optional[str] = None,
        jitter: Optional[float] = None,
        relax_steps: Optional[int] = None,
        geodesic_level: Optional[int] = None,
    ) -> None:
        """
        Initialize particle centers on the sphere using either Fibonacci or geodesic tilings,
        with optional jitter and post-initialization relaxation.
        """
        R = self.cfg.radius
        method = (method or self.cfg.init_method or "fibonacci").lower()
        if method == "geodesic":
            level = geodesic_level if geodesic_level is not None else self.cfg.init_geodesic_level
            base_pts = geodesic_sphere_points(level)
            if base_pts.shape[0] < n_particles:
                # pad with fibonacci sequence if requested N exceeds geodesic nodes
                extra_needed = n_particles - base_pts.shape[0]
                fib = self._fibonacci_points(extra_needed)
                centers = np.vstack([base_pts, fib])
            else:
                # pick quasi-uniform subset
                idx = np.linspace(0, base_pts.shape[0] - 1, n_particles, dtype=int)
                centers = base_pts[idx]
        else:
            centers = self._fibonacci_points(n_particles)
        jit = jitter if jitter is not None else self.cfg.init_jitter
        if jit and jit > 0.0:
            noise = self.rng.normal(size=centers.shape)
            centers = np.array([normalize(pt + jit * n) for pt, n in zip(centers, noise)])
        centers = centers[:n_particles] * R
        self.particles = []
        for i in range(n_particles):
            pos = centers[i]
            if self.cfg.align_body_vec is not None:
                u_body = normalize(np.array(self.cfg.align_body_vec, dtype=float))
                nhat = normalize(pos)
                q_align = quat_from_two_unit_vectors(u_body, nhat)
                # random azimuthal rotation about nhat
                phi = self.rng.uniform(0.0, 2.0 * np.pi)
                q_spin = quat_from_axis_angle(nhat, phi)
                q0 = quat_mul(q_spin, q_align)
            else:
                q0 = np.array([1.0, 0.0, 0.0, 0.0])
            self.particles.append(Particle(pos=pos, quat=q0))
        relax = relax_steps if relax_steps is not None else self.cfg.init_relax_steps
        if relax and relax > 0:
            self.global_relax(relax, neighbor_angle_deg=25.0)

    def _fibonacci_points(self, n_particles: int) -> NDArray[np.floating]:
        n_particles = max(1, int(n_particles))
        index = np.arange(0, n_particles, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * index / n_particles)
        theta = np.pi * (1 + 5 ** 0.5) * index
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.stack([x, y, z], axis=1)

    def _propose_move(self, p: Particle) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        R = self.cfg.radius
        # Tangent-plane displacement
        d = self.rng.normal(size=3)
        nhat = normalize(p.pos)
        d_tan = d - np.dot(d, nhat) * nhat
        nrm = np.linalg.norm(d_tan)
        if nrm < 1e-16:
            # resample small displacement
            d_tan = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), nhat) * nhat
            nrm = np.linalg.norm(d_tan)
        step = self.cfg.translate_sigma
        p_new = normalize(p.pos + (step / nrm) * d_tan) * R

        # Orientation proposal
        if self.cfg.align_body_vec is None:
            # free rotation in SO(3)
            axis = normalize(self.rng.normal(size=3))
            angle = self.cfg.rotate_sigma * self.rng.normal()
            dq = quat_from_axis_angle(axis, angle)
            q_new = quat_mul(dq, p.quat)
            q_new = q_new / np.linalg.norm(q_new)
        else:
            # keep align_body_vec aligned to radial: correct drift then spin about nhat
            u_body = normalize(np.array(self.cfg.align_body_vec, dtype=float))
            nhat_old = normalize(p.pos)
            nhat_new = normalize(p_new)
            # quaternion to map current aligned direction to new radial
            u_world_old = rotate_vec_by_quat(u_body, p.quat)
            q_corr = quat_from_two_unit_vectors(u_world_old, nhat_new)
            q_aligned = quat_mul(q_corr, p.quat)
            q_aligned = q_aligned / np.linalg.norm(q_aligned)
            # azimuthal rotation about nhat_new
            if self.cfg.restrict_orientation_azimuthal:
                delta_phi = self.cfg.rotate_sigma * self.rng.normal()
                q_spin = quat_from_axis_angle(nhat_new, delta_phi)
                q_new = quat_mul(q_spin, q_aligned)
                q_new = q_new / np.linalg.norm(q_new)
            else:
                # allow small wobble but re-align after
                axis = normalize(self.rng.normal(size=3))
                angle = self.cfg.rotate_sigma * self.rng.normal()
                dq = quat_from_axis_angle(axis, angle)
                q_tmp = quat_mul(dq, q_aligned)
                u_world_tmp = rotate_vec_by_quat(u_body, q_tmp)
                q_fix = quat_from_two_unit_vectors(u_world_tmp, nhat_new)
                q_new = quat_mul(q_fix, q_tmp)
                q_new = q_new / np.linalg.norm(q_new)
        return p_new, q_new

    def _overlaps_any(self, idx: int, pos: NDArray[np.floating], quat: NDArray[np.floating]) -> bool:
        for j, pj in enumerate(self.particles):
            if j == idx:
                continue
            if sat_overlap_spheropolyhedra(
                self.shape, pos, quat, self.shape, pj.pos, pj.quat
            ):
                return True
        return False

    def _random_point_on_sphere(self, radius: float) -> NDArray[np.floating]:
        vec = normalize(self.rng.normal(size=3))
        return vec * radius

    def _random_quaternion(self, nhat: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.cfg.align_body_vec is None:
            axis = normalize(self.rng.normal(size=3))
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            quat = quat_from_axis_angle(axis, angle)
        else:
            u_body = normalize(np.array(self.cfg.align_body_vec, dtype=float))
            q_align = quat_from_two_unit_vectors(u_body, nhat)
            phi = self.rng.uniform(0.0, 2.0 * np.pi)
            q_spin = quat_from_axis_angle(nhat, phi)
            quat = quat_mul(q_spin, q_align)
        return quat / np.linalg.norm(quat)

    def _overlaps_configuration(self, pos: NDArray[np.floating], quat: NDArray[np.floating]) -> bool:
        for pj in self.particles:
            if sat_overlap_spheropolyhedra(
                self.shape, pos, quat, self.shape, pj.pos, pj.quat
            ):
                return True
        return False

    def _overlaps_with_overrides(
        self,
        idx: int,
        pos: NDArray[np.floating],
        quat: NDArray[np.floating],
        overrides: Optional[Dict[int, Particle]] = None,
    ) -> bool:
        if overrides is None:
            overrides = {}
        for j, pj in enumerate(self.particles):
            if j == idx:
                continue
            other = overrides.get(j, pj)
            if sat_overlap_spheropolyhedra(
                self.shape,
                pos,
                quat,
                self.shape,
                other.pos,
                other.quat,
            ):
                return True
        return False

    def _order_active(self) -> bool:
        return (
            self.cfg.order_strength > 0.0
            and self.cfg.target_neighbor_angle_deg is not None
            and len(self.particles) > max(3, self.cfg.order_k_neighbors)
        )

    def _order_penalty(
        self,
        idx: int,
        pos: NDArray[np.floating],
        overrides: Optional[Dict[int, Particle]] = None,
    ) -> float:
        if not self._order_active():
            return 0.0
        overrides = overrides or {}
        center_dir = normalize(pos)
        angles: List[float] = []
        for j, pj in enumerate(self.particles):
            if j == idx:
                continue
            other = overrides.get(j, pj)
            ang = math.acos(float(np.clip(np.dot(normalize(other.pos), center_dir), -1.0, 1.0)))
            angles.append(ang)
        if not angles:
            return 0.0
        angles.sort()
        k = max(1, min(self.cfg.order_k_neighbors, len(angles)))
        target = math.radians(float(self.cfg.target_neighbor_angle_deg))
        subset = np.array(angles[:k], dtype=float)
        diff = subset - target
        return float(np.mean(diff * diff))

    def _order_penalty_new(self, pos: NDArray[np.floating]) -> float:
        if not self._order_active():
            return 0.0
        center_dir = normalize(pos)
        angles = [
            math.acos(float(np.clip(np.dot(normalize(pj.pos), center_dir), -1.0, 1.0)))
            for pj in self.particles
        ]
        if not angles:
            return 0.0
        angles.sort()
        k = max(1, min(self.cfg.order_k_neighbors, len(angles)))
        target = math.radians(float(self.cfg.target_neighbor_angle_deg))
        subset = np.array(angles[:k], dtype=float)
        diff = subset - target
        return float(np.mean(diff * diff))

    def _metropolis_accept(self, old: float, new: float) -> bool:
        if new <= old:
            return True
        beta = max(1e-9, float(self.cfg.order_strength))
        return self.rng.random() < math.exp(-beta * (new - old))

    def _order_accept_existing(
        self,
        idx: int,
        new_pos: NDArray[np.floating],
        overrides: Optional[Dict[int, Particle]] = None,
    ) -> bool:
        if not self._order_active():
            return True
        overrides = dict(overrides) if overrides else {}
        current_particle = overrides.get(idx, self.particles[idx])
        old_pen = self._order_penalty(idx, current_particle.pos, overrides)
        overrides[idx] = Particle(pos=new_pos, quat=current_particle.quat)
        new_pen = self._order_penalty(idx, new_pos, overrides)
        return self._metropolis_accept(old_pen, new_pen)

    def _order_accept_new(self, pos: NDArray[np.floating]) -> bool:
        if not self._order_active():
            return True
        penalty = self._order_penalty_new(pos)
        baseline = 0.0
        return self._metropolis_accept(baseline, penalty)

    def _neighbor_indices(self, idx: int, angle_rad: float) -> List[int]:
        if angle_rad <= 0.0:
            return []
        center_dir = normalize(self.particles[idx].pos)
        out: List[int] = []
        for j, pj in enumerate(self.particles):
            if j == idx:
                continue
            ang = math.acos(float(np.clip(np.dot(normalize(pj.pos), center_dir), -1.0, 1.0)))
            if ang <= angle_rad:
                out.append(j)
        return out

    def _local_relax(self, center_idx: int, steps: int, neighbor_angle_rad: float, parallel_proposals: int = 1) -> None:
        if steps <= 0 or not self.particles:
            return
        indices = [center_idx] + self._neighbor_indices(center_idx, neighbor_angle_rad)
        unique_indices = np.array(sorted(set(indices)), dtype=int)
        if unique_indices.size == 0:
            return
        for _ in range(steps):
            pick = int(unique_indices[self.rng.integers(0, unique_indices.size)])
            particle = self.particles[pick]
            if parallel_proposals <= 1:
                new_pos, new_quat = self._propose_move(particle)
                self.total_moves += 1
                if self._overlaps_any(pick, new_pos, new_quat):
                    continue
                if not self._order_accept_existing(pick, new_pos):
                    continue
                self.particles[pick] = Particle(pos=new_pos, quat=new_quat)
                self.accepted_moves += 1
            else:
                candidates = [self._propose_move(particle) for _ in range(parallel_proposals)]
                self.total_moves += parallel_proposals

                def _check(candidate):
                    pos, quat = candidate
                    return candidate if not self._overlaps_any(pick, pos, quat) else None

                accepted_candidate: Optional[Tuple[NDArray[np.floating], NDArray[np.floating]]] = None
                with ThreadPoolExecutor(max_workers=parallel_proposals) as executor:
                    future_map = {executor.submit(_check, cand): cand for cand in candidates}
                    for future in as_completed(future_map):
                        res = future.result()
                        if res is None:
                            continue
                        if self._order_accept_existing(pick, res[0]):
                            accepted_candidate = res
                            break

                if accepted_candidate is not None:
                    new_pos, new_quat = accepted_candidate
                    self.particles[pick] = Particle(pos=new_pos, quat=new_quat)
                    self.accepted_moves += 1

    def try_insert_random_particle(
        self,
        insertion_attempts: int = 10,
        relax_steps: int = 100,
        neighbor_angle_deg: float = 25.0,
        parallel_proposals: int = 1,
    ) -> bool:
        attempts = max(1, int(insertion_attempts))
        neighbor_angle_rad = math.radians(max(0.0, float(neighbor_angle_deg)))
        for _ in range(attempts):
            pos = self._random_point_on_sphere(self.cfg.radius)
            nhat = normalize(pos)
            quat = self._random_quaternion(nhat)
            if self._overlaps_configuration(pos, quat):
                continue
            if not self._order_accept_new(pos):
                continue
            self.particles.append(Particle(pos=pos, quat=quat))
            if relax_steps > 0 and len(self.particles) > 1:
                self._local_relax(len(self.particles) - 1, relax_steps, neighbor_angle_rad, parallel_proposals=parallel_proposals)
            return True
        return False

    def _block_relax_random(self, block_size: int, neighbor_angle_rad: float, attempts: int) -> None:
        if block_size <= 1 or len(self.particles) < 2:
            return
        center = int(self.rng.integers(0, len(self.particles)))
        candidates = [center] + self._neighbor_indices(center, neighbor_angle_rad)
        unique = np.array(sorted(set(candidates)), dtype=int)
        if unique.size <= 1:
            return
        if unique.size <= block_size:
            block = unique.tolist()
        else:
            block = self.rng.choice(unique, size=block_size, replace=False).tolist()
        self._attempt_block_move(block, attempts)

    def _attempt_block_move(self, indices: List[int], attempts: int) -> None:
        if not indices:
            return
        attempts = max(1, attempts)
        for _ in range(attempts):
            overrides: Dict[int, Particle] = {}
            success = True
            for idx in indices:
                base_particle = overrides.get(idx, self.particles[idx])
                new_pos, new_quat = self._propose_move(base_particle)
                self.total_moves += 1
                if self._overlaps_with_overrides(idx, new_pos, new_quat, overrides):
                    success = False
                    break
                if not self._order_accept_existing(idx, new_pos, overrides):
                    success = False
                    break
                overrides[idx] = Particle(pos=new_pos, quat=new_quat)
            if success:
                for idx, updated in overrides.items():
                    self.particles[idx] = updated
                    self.accepted_moves += 1
                break

    def global_relax(
        self,
        sweeps: int,
        neighbor_angle_deg: float = 25.0,
        block_size: Optional[int] = None,
    ) -> None:
        if sweeps <= 0 or not self.particles:
            return
        neighbor_angle_rad = math.radians(max(0.0, neighbor_angle_deg))
        block_size = block_size if block_size is not None else self.cfg.block_relax_size
        for _ in range(sweeps):
            if (
                block_size
                and block_size > 1
                and len(self.particles) >= block_size
                and self.rng.random() < 0.4
            ):
                self._block_relax_random(block_size, neighbor_angle_rad, self.cfg.block_relax_attempts)
            else:
                pick = int(self.rng.integers(0, len(self.particles)))
                self._local_relax(pick, 1, neighbor_angle_rad, parallel_proposals=1)

    def step(self) -> None:
        if not self.particles:
            raise RuntimeError("Particles not initialized")
        i = self.rng.integers(0, len(self.particles))
        pi = self.particles[i]
        new_pos, new_quat = self._propose_move(pi)
        self.total_moves += 1
        if self._overlaps_any(i, new_pos, new_quat):
            return
        if not self._order_accept_existing(i, new_pos):
            return
        self.particles[i] = Particle(pos=new_pos, quat=new_quat)
        self.accepted_moves += 1

    def try_shrink(self) -> bool:
        R_old = self.cfg.radius
        R_new = R_old * self.cfg.shrink_factor
        scaled = []
        for p in self.particles:
            scaled.append(Particle(pos=normalize(p.pos) * R_new, quat=p.quat))

        # Check overlaps on the new radius
        for a in range(len(scaled)):
            for b in range(a + 1, len(scaled)):
                if sat_overlap_spheropolyhedra(
                    self.shape, scaled[a].pos, scaled[a].quat,
                    self.shape, scaled[b].pos, scaled[b].quat,
                ):
                    # Reject shrink
                    self.shrink_fail_count += 1
                    return False

        # Accept shrink
        self.cfg.radius = R_new
        self.particles = scaled
        self.shrink_fail_count = 0
        self._update_footprint_angle()
        return True

    def _apply_annealing(self, step: int) -> None:
        if self.cfg.anneal_steps and self.cfg.anneal_steps > 0:
            ratio = min(1.0, step / float(self.cfg.anneal_steps))
            if self.cfg.translate_sigma_final is not None:
                self.cfg.translate_sigma = (1 - ratio) * self._translate_sigma0 + ratio * self.cfg.translate_sigma_final
            if self.cfg.rotate_sigma_final is not None:
                self.cfg.rotate_sigma = (1 - ratio) * self._rotate_sigma0 + ratio * self.cfg.rotate_sigma_final
            if self.cfg.shrink_factor_final is not None:
                self.cfg.shrink_factor = (1 - ratio) * self._shrink_factor0 + ratio * self.cfg.shrink_factor_final

    def run(self, n_steps: int, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        for step in range(1, n_steps + 1):
            self._apply_annealing(step)
            self.step()
            if (step % self.cfg.shrink_interval) == 0:
                ok = self.try_shrink()
                if (not ok) and (self.shrink_fail_count >= self.cfg.max_shrink_fail):
                    if progress_callback:
                        progress_callback(step, n_steps)
                    break
            if progress_callback:
                progress_callback(step, n_steps)

    # Simple observables
    @property
    def acceptance(self) -> float:
        return (self.accepted_moves / self.total_moves) if self.total_moves else 0.0

    @property
    def surface_number_density(self) -> float:
        R = self.cfg.radius
        return len(self.particles) / (4.0 * np.pi * R * R)

    def _update_footprint_angle(self) -> None:
        if not hasattr(self.shape, "base_vertices") or self.cfg.radius <= 0:
            self._footprint_angle = 0.0
            return
        verts = self.shape.base_vertices
        if verts.size == 0:
            self._footprint_angle = 0.0
            return
        max_body = float(np.linalg.norm(verts, axis=1).max()) + float(self.shape.sweep_radius)
        ratio = min(1.0, max(0.0, max_body / max(1e-9, self.cfg.radius)))
        self._footprint_angle = float(np.arcsin(ratio))

    def estimate_surface_coverage(self, samples: int = 20000, seed: int = 1234) -> float:
        if not self.particles or self._footprint_angle <= 0.0:
            return 0.0
        rng = np.random.default_rng(seed)
        centers = np.array([normalize(p.pos) for p in self.particles], dtype=float)
        hits = 0
        total = max(1, int(samples))
        batch = min(4096, total)
        for start in range(0, total, batch):
            m = min(batch, total - start)
            vecs = rng.normal(size=(m, 3))
            vecs = np.array([normalize(v) for v in vecs])
            dots = np.clip(vecs @ centers.T, -1.0, 1.0)
            ang = np.arccos(dots)
            covered = (ang <= self._footprint_angle).any(axis=1)
            hits += int(np.count_nonzero(covered))
        return hits / float(total)


def build_rtt_particle(cube_length: float, truncation: float, roundness: float) -> ConvexSpheropolyhedron:
    params = RTTParams(cube_length=cube_length, truncation=truncation, roundness=roundness)
    verts = params.vertices
    return ConvexSpheropolyhedron.from_vertices(verts, sweep_radius=params.r_sweep)
