"""
硬多面体球形约束 MC（依据技术报告规范）。

特性
- 支持多类凸多面体（Cube / Octa / Rhombic Dodecahedron / Truncated RD）。
- 粒子在球形硬壁内进行局域平移+旋转 MC，带有自适应步长。
- 准静态压缩：按步长 α 缩小球半径 → 松弛若干 sweep → 若失败则回退并调小 α。
- 广相采用包围球 + 三维网格粗筛，窄相使用 SAT。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, HalfspaceIntersection

from .reversible_compression_mc import (
    ConvexSpheropolyhedron,
    sat_overlap_spheropolyhedra,
    normalize,
    quat_from_axis_angle,
    quat_mul,
)

Array3 = NDArray[np.floating]
Quat = NDArray[np.floating]

EPS = 1e-9


def _poly_from_vertices(vertices: Array3) -> Tuple[ConvexSpheropolyhedron, float]:
    hull = ConvexHull(vertices)
    poly = ConvexSpheropolyhedron.from_vertices(vertices, sweep_radius=0.0)
    r_bound = float(np.linalg.norm(vertices, axis=1).max())
    return poly, r_bound


def cube_vertices() -> Array3:
    pts = []
    scale = 1.0 / math.sqrt(3.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append((sx * scale, sy * scale, sz * scale))
    return np.array(pts, dtype=float)


def octa_vertices() -> Array3:
    return np.array(
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
        dtype=float,
    )


def rhombic_dodecahedron_vertices() -> Array3:
    pts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            vec = [0.0, 0.0, 0.0]
            vec[axis] = 2.0 * sign
            pts.append(tuple(vec))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append((sx, sy, sz))
    pts = np.array(pts, dtype=float)
    pts /= np.linalg.norm(pts, axis=1).max()  # 归一化外接球半径
    return pts


def truncated_rd_vertices(truncation: float = 0.68) -> Array3:
    """
    通过对 rhombic dodecahedron 在每个顶点方向引入平行截平面获得 TRD。
    truncation = 0 → 原始 RD, 0.68 ~ 实验默认。
    """
    trunc = float(np.clip(truncation, 0.0, 0.9))
    base = rhombic_dodecahedron_vertices()
    hull = ConvexHull(base)
    halfspaces: List[Array3] = []
    for eq in hull.equations:
        halfspaces.append(eq.copy())
    for v in base:
        r = float(np.linalg.norm(v))
        if r < 1e-9:
            continue
        u = v / r
        offset = r * (1.0 - trunc)
        halfspaces.append(np.array([u[0], u[1], u[2], -offset], dtype=float))
    hs = np.vstack(halfspaces)
    hs_int = HalfspaceIntersection(hs, np.zeros(3, dtype=float))
    verts = hs_int.intersections
    hull_trunc = ConvexHull(verts)
    verts = hull_trunc.points[hull_trunc.vertices]
    verts = hull_trunc.points  # use all unique vertices
    verts /= np.linalg.norm(verts, axis=1).max()
    return verts


SHAPE_BUILDERS = {
    "cube": cube_vertices,
    "octa": octa_vertices,
    "rd": rhombic_dodecahedron_vertices,
    "trd": truncated_rd_vertices,
}


@dataclass
class PolyhedronModel:
    body: ConvexSpheropolyhedron
    bounding_radius: float


def build_polyhedron(name: str, truncation: float | None = None) -> PolyhedronModel:
    key = name.lower()
    if key not in SHAPE_BUILDERS:
        raise ValueError(f"未知形状: {name}")
    if key == "trd":
        verts = SHAPE_BUILDERS[key](truncation if truncation is not None else 0.68)
    else:
        verts = SHAPE_BUILDERS[key]()
    body, r_bound = _poly_from_vertices(verts)
    return PolyhedronModel(body=body, bounding_radius=r_bound)


@dataclass
class SimulationConfig:
    delta_translate: float = 0.05
    delta_rotate_deg: float = 5.0
    pre_sweeps: int = 500
    relax_sweeps: int = 200
    max_sweeps: int = 2000
    compress_alpha: float = 5e-4
    min_alpha: float = 5e-6
    alpha_scale_up: float = 1.2
    alpha_scale_down: float = 0.5
    max_failed_compressions: int = 8
    neighbor_cut_deg: float = 25.0
    surface_threshold: float = 0.98
    radial_bins: int = 48


@dataclass
class Particle:
    pos: Array3
    quat: Quat


class CellList:
    def __init__(self, radius: float, cell_size: float) -> None:
        self.radius = radius
        self.cell_size = max(cell_size, 1e-3)
        self.inv = 1.0 / self.cell_size
        self.n_cells = int(math.ceil(2.0 * radius * self.inv)) + 2
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}
        self.membership: Dict[int, Tuple[int, int, int]] = {}

    def _cell_index(self, pos: Array3) -> Tuple[int, int, int]:
        shifted = (pos + self.radius) * self.inv
        return (
            int(math.floor(shifted[0])),
            int(math.floor(shifted[1])),
            int(math.floor(shifted[2])),
        )

    def rebuild(self, particles: List[Particle]) -> None:
        self.cells.clear()
        self.membership.clear()
        for idx, particle in enumerate(particles):
            cell = self._cell_index(particle.pos)
            self.cells.setdefault(cell, []).append(idx)
            self.membership[idx] = cell

    def update(self, idx: int, pos: Array3) -> None:
        old = self.membership.get(idx)
        new = self._cell_index(pos)
        if old == new:
            return
        if old and old in self.cells:
            self.cells[old].remove(idx)
        self.cells.setdefault(new, []).append(idx)
        self.membership[idx] = new

    def neighbors(self, pos: Array3) -> Iterable[int]:
        base = self._cell_index(pos)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = (base[0] + dx, base[1] + dy, base[2] + dz)
                    if cell in self.cells:
                        yield from self.cells[cell]


class HardPolyhedraConfinementMC:
    def __init__(
        self,
        polyhedron: PolyhedronModel,
        radius: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.shape = polyhedron
        self.radius = float(radius)
        self.rng = np.random.default_rng() if rng is None else rng
        self.particles: List[Particle] = []
        self.moves_attempted = 0
        self.moves_accepted = 0
        self.cell_list = CellList(self.radius, cell_size=2.2 * self.shape.bounding_radius)

    # -------- Initialization -------------------------------------------------

    def random_initialization(self, n: int, max_attempts: int = 50_000) -> None:
        self.particles = []
        attempts = 0
        while len(self.particles) < n and attempts < max_attempts:
            attempts += 1
            pos = self._random_position()
            quat = self._random_quaternion()
            if not self._inside_sphere(pos, quat):
                continue
            if self._overlaps_configuration(pos, quat):
                continue
            self.particles.append(Particle(pos=pos, quat=quat))
        if len(self.particles) < n:
            raise RuntimeError(
                f"无法放置全部粒子 (N={n}). 成功 {len(self.particles)} 次, 尝试 {attempts}。"
            )
        self.cell_list = CellList(self.radius, cell_size=2.2 * self.shape.bounding_radius)
        self.cell_list.rebuild(self.particles)

    def _random_position(self) -> Array3:
        # 拒绝采样：均匀取球内点（半径需要留出包围球）
        limit = self.radius - self.shape.bounding_radius - 1e-3
        if limit <= 0:
            raise RuntimeError("球半径不足以容纳粒子。")
        while True:
            vec = self.rng.normal(size=3)
            vec = normalize(vec)
            r = (self.rng.random() ** (1.0 / 3.0)) * limit
            pos = vec * r
            return pos

    def _random_quaternion(self) -> Quat:
        axis = normalize(self.rng.normal(size=3))
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        quat = quat_from_axis_angle(axis, angle)
        return quat / np.linalg.norm(quat)

    # -------- Geometry checks ------------------------------------------------

    def _inside_sphere(self, pos: Array3, quat: Quat) -> bool:
        if np.linalg.norm(pos) + self.shape.bounding_radius > self.radius + 1e-7:
            return False
        verts = self.shape.body.world_vertices(quat, pos)
        norms = np.linalg.norm(verts, axis=1)
        return float(norms.max()) <= self.radius + 1e-8

    def _overlaps_configuration(self, pos: Array3, quat: Quat, ignore_idx: int | None = None) -> bool:
        if not self.cell_list.cells:
            # during init fallback对所有粒子
            candidates = range(len(self.particles))
        else:
            candidates = self.cell_list.neighbors(pos)
        for j in candidates:
            if ignore_idx is not None and j == ignore_idx:
                continue
            other = self.particles[j]
            center_dist = np.linalg.norm(pos - other.pos)
            if center_dist > (2.0 * self.shape.bounding_radius + 1e-6):
                continue
            if sat_overlap_spheropolyhedra(
                self.shape.body,
                pos,
                quat,
                self.shape.body,
                other.pos,
                other.quat,
            ):
                return True
        return False

    def detect_any_overlap(self) -> bool:
        for i, particle in enumerate(self.particles):
            if self._overlaps_configuration(particle.pos, particle.quat, ignore_idx=i):
                return True
        return False

    # -------- Monte Carlo moves ---------------------------------------------

    def mc_sweep(self, cfg: SimulationConfig) -> None:
        accepts = 0
        for _ in range(len(self.particles)):
            acc = self._attempt_move(cfg.delta_translate, math.radians(cfg.delta_rotate_deg))
            if acc:
                accepts += 1
        self.moves_attempted += len(self.particles)
        self.moves_accepted += accepts
        acceptance = accepts / max(1, len(self.particles))
        if acceptance < 0.25:
            cfg.delta_translate *= 0.9
            cfg.delta_rotate_deg *= 0.9
        elif acceptance > 0.55:
            cfg.delta_translate *= 1.05
            cfg.delta_rotate_deg *= 1.05
        cfg.delta_translate = float(np.clip(cfg.delta_translate, 1e-3, 0.5))
        cfg.delta_rotate_deg = float(np.clip(cfg.delta_rotate_deg, 0.1, 30.0))

    def _attempt_move(self, delta_t: float, delta_r: float) -> bool:
        if not self.particles:
            return False
        idx = int(self.rng.integers(0, len(self.particles)))
        particle = self.particles[idx]
        disp = self.rng.uniform(-1.0, 1.0, size=3)
        disp = normalize(disp) * delta_t * self.rng.random()
        pos_new = particle.pos + disp
        axis = normalize(self.rng.normal(size=3))
        angle = delta_r * self.rng.uniform(-1.0, 1.0)
        dq = quat_from_axis_angle(axis, angle)
        quat_new = quat_mul(dq, particle.quat)
        quat_new = quat_new / np.linalg.norm(quat_new)
        if not self._inside_sphere(pos_new, quat_new):
            return False
        if self._overlaps_configuration(pos_new, quat_new, ignore_idx=idx):
            return False
        self.particles[idx] = Particle(pos=pos_new, quat=quat_new)
        self.cell_list.update(idx, pos_new)
        return True

    # -------- Compression loop ----------------------------------------------

    def run_density_schedule(
        self,
        cfg: SimulationConfig,
        target_radius: float | None = None,
        progress_hook: Optional[callable] = None,
    ) -> None:
        # 预平衡
        for sweep_idx in range(cfg.pre_sweeps):
            self.mc_sweep(cfg)
            if progress_hook:
                progress_hook("pre", sweep_idx + 1, cfg.pre_sweeps, self.radius)

        failed = 0
        sweeps_done = 0
        while sweeps_done < cfg.max_sweeps:
            sweeps_done += 1
            success = self._compress_once(cfg)
            if progress_hook:
                progress_hook("compress", sweeps_done, cfg.max_sweeps, self.radius)
            if success:
                failed = 0
                cfg.compress_alpha = min(cfg.compress_alpha * cfg.alpha_scale_up, 2e-3)
            else:
                failed += 1
                cfg.compress_alpha = max(cfg.compress_alpha * cfg.alpha_scale_down, cfg.min_alpha)
                if failed >= cfg.max_failed_compressions:
                    break
            if target_radius and self.radius <= target_radius:
                break
        # 最后再做一次松弛
        for sweep_idx in range(cfg.relax_sweeps):
            self.mc_sweep(cfg)

    def _compress_once(self, cfg: SimulationConfig) -> bool:
        old_radius = self.radius
        new_radius = old_radius * (1.0 - cfg.compress_alpha)
        if new_radius <= self.shape.bounding_radius * 1.05:
            return False
        backup_particles = [
            Particle(pos=np.copy(p.pos), quat=np.copy(p.quat)) for p in self.particles
        ]
        scale = new_radius / old_radius
        for idx, p in enumerate(self.particles):
            new_pos = p.pos * scale
            self.particles[idx] = Particle(pos=new_pos, quat=p.quat.copy())
        self.radius = new_radius
        self.cell_list = CellList(self.radius, cell_size=2.2 * self.shape.bounding_radius)
        self.cell_list.rebuild(self.particles)
        for _ in range(cfg.relax_sweeps):
            self.mc_sweep(cfg)
        if self.detect_any_overlap():
            # 回退
            self.particles = backup_particles
            self.radius = old_radius
            self.cell_list = CellList(self.radius, cell_size=2.2 * self.shape.bounding_radius)
            self.cell_list.rebuild(self.particles)
            return False
        return True

    # -------- Diagnostics ----------------------------------------------------

    def radial_profile(self, bins: int = 48) -> Tuple[np.ndarray, np.ndarray]:
        centers = np.array([np.linalg.norm(p.pos) for p in self.particles])
        hist, edges = np.histogram(centers, bins=bins, range=(0.0, self.radius))
        return 0.5 * (edges[1:] + edges[:-1]), hist

    def surface_particles(self, threshold: float = 0.98) -> List[int]:
        ids: List[int] = []
        limit = self.radius * float(np.clip(threshold, 0.8, 0.999))
        for idx, particle in enumerate(self.particles):
            verts = self.shape.body.world_vertices(particle.quat, particle.pos)
            if np.linalg.norm(verts, axis=1).max() >= limit:
                ids.append(idx)
        return ids

    def surface_order_parameter(self, mode: str = "hex", threshold: float = 0.98) -> float:
        surf = self.surface_particles(threshold=threshold)
        if not surf:
            return 0.0
        order_n = 6 if mode == "hex" else 4
        values: List[complex] = []
        for idx in surf:
            pi = self.particles[idx]
            center_dir = normalize(pi.pos)
            tangent_x = normalize(np.cross(center_dir, np.array([0.0, 0.0, 1.0])))
            if np.linalg.norm(tangent_x) < 1e-6:
                tangent_x = normalize(np.cross(center_dir, np.array([0.0, 1.0, 0.0])))
            tangent_y = np.cross(center_dir, tangent_x)
            angles: List[float] = []
            for jdx in surf:
                if jdx == idx:
                    continue
                pj = self.particles[jdx]
                ang = math.acos(
                    float(np.clip(np.dot(normalize(pj.pos), center_dir), -1.0, 1.0))
                )
                if ang > math.radians(30):
                    continue
                proj = pj.pos - np.dot(pj.pos, center_dir) * center_dir
                proj = proj / (np.linalg.norm(proj) + 1e-12)
                x = float(np.dot(proj, tangent_x))
                y = float(np.dot(proj, tangent_y))
                angles.append(math.atan2(y, x))
            if not angles:
                continue
            psi = sum(np.exp(1j * order_n * np.array(angles))) / len(angles)
            values.append(psi)
        if not values:
            return 0.0
        return float(np.mean(np.abs(values)))


__all__ = [
    "SimulationConfig",
    "PolyhedronModel",
    "build_polyhedron",
    "HardPolyhedraConfinementMC",
]
