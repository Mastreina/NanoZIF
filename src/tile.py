"""定义可选多面体贴片的数据结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Dict, List, Tuple

import numpy as np

from .geometry import gnomonic_project_point, lift_to_sphere, orthonormal_basis, rotate_basis


def _point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """射线法判断点是否在二维多边形内（包含边界）。"""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if intersects:
            inside = not inside
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if (
            abs(cross) <= 1e-12
            and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
        ):
            return True
    return inside


SHAPE_LIBRARY = {
    "square": {"sides": 4, "height_factor": 1.0, "top_scale": 1.0},
    "hexagon": {"sides": 6, "height_factor": 0.25, "top_scale": 1.0},
    "hexagonal_prism": {"sides": 6, "height_factor": 1.0, "top_scale": 1.0},
    "octagon": {"sides": 8, "height_factor": 0.4, "top_scale": 1.0},
    "truncated_octahedron": {"sides": 8, "height_factor": 1.1, "top_scale": 0.6},
    "truncated_cube": {"sides": 8, "height_factor": None, "top_scale": None},
}


@dataclass
class RegularPolygonPatch:
    """球面贴片模型，可生成不同粒子的三维几何。"""

    radius: float
    edge_length: float
    normal: np.ndarray
    rotation: float
    sides: int
    shape: str
    _basis: Tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False, repr=False)
    _plane_vertices: np.ndarray = field(init=False, repr=False)
    _bounding_radius: float = field(init=False, repr=False)
    _sample_cache: Dict[int, np.ndarray] = field(init=False, repr=False)
    _height: float | None = field(init=False, repr=False)
    _top_scale: float | None = field(init=False, repr=False)
    _mesh_vertices: np.ndarray = field(init=False, repr=False)
    _mesh_triangles: Tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False, repr=False)
    _canonical_truncated_cube: np.ndarray | None = field(init=False, repr=False, default=None)
    _truncated_faces: Dict[str, List[List[int]]] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.sides < 3:
            raise ValueError("sides必须>=3")
        self.normal = np.array(self.normal, dtype=float)
        norm = np.linalg.norm(self.normal)
        if norm == 0:
            raise ValueError("法向量不可为零向量")
        self.normal /= norm
        self.rotation = float(self.rotation) % (2.0 * math.pi)
        self.shape = self.shape.lower()
        cfg = SHAPE_LIBRARY.get(self.shape)
        if cfg is None:
            raise ValueError(f"未知的贴片形状: {self.shape}")
        if cfg["sides"] != self.sides:
            raise ValueError("形状与边数不匹配")
        self._height = (
            None if cfg["height_factor"] is None else self.edge_length * float(cfg["height_factor"])
        )
        self._top_scale = None if cfg["top_scale"] is None else float(cfg["top_scale"])
        self._sample_cache = {}
        self._setup_static_polygon()
        self._update_basis()
        self._rebuild_polyhedron()

    @property
    def center(self) -> np.ndarray:
        return self.radius * self.normal

    def local_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._basis

    def copy_with(
        self, *, normal: np.ndarray | None = None, rotation: float | None = None
    ) -> "RegularPolygonPatch":
        new_normal = normal if normal is not None else self.normal
        new_rotation = rotation if rotation is not None else self.rotation
        return RegularPolygonPatch(
            radius=self.radius,
            edge_length=self.edge_length,
            normal=new_normal,
            rotation=new_rotation,
            sides=self.sides,
            shape=self.shape,
        )

    def update_orientation(self, normal: np.ndarray, rotation: float | None = None) -> None:
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        if rotation is not None:
            self.rotation = float(rotation) % (2.0 * math.pi)
        self._update_basis()
        self._rebuild_polyhedron()

    def contains_point(self, point: np.ndarray) -> bool:
        sx, sy = gnomonic_project_point(point, self._basis)
        if math.isinf(sx) or math.isinf(sy):
            return False
        if sx * sx + sy * sy > (self._bounding_radius + 1e-6) ** 2:
            return False
        return _point_in_polygon((sx, sy), self._plane_vertices)

    def sample_surface_points(self, resolution: int) -> np.ndarray:
        if resolution <= 0:
            return np.zeros((0, 3), dtype=float)
        cached = self._sample_cache.get(resolution)
        if cached is not None:
            return cached
        grid = np.linspace(-self._bounding_radius, self._bounding_radius, resolution)
        samples: List[np.ndarray] = []
        for sx in grid:
            for sy in grid:
                if _point_in_polygon((sx, sy), self._plane_vertices):
                    samples.append(lift_to_sphere((sx, sy), self._basis, self.radius))
        if not samples:
            samples = [lift_to_sphere(tuple(v), self._basis, self.radius) for v in self._plane_vertices]
        array = np.array(samples, dtype=float)
        self._sample_cache[resolution] = array
        return array

    def polygon_vertices(self) -> np.ndarray:
        return np.array(
            [lift_to_sphere(tuple(v), self._basis, self.radius) for v in self._plane_vertices],
            dtype=float,
        )

    def to_plane_polygon(self) -> np.ndarray:
        return np.array(self._plane_vertices, copy=True)

    def polyhedron_mesh(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._mesh_vertices, self._mesh_triangles

    def overlaps_with(
        self,
        other: "RegularPolygonPatch",
        resolution: int = 7,
        tolerance: float = 0.0,
    ) -> bool:
        tol = max(0.0, min(1.0, tolerance))
        samples_self = self.sample_surface_points(resolution)
        if samples_self.size:
            allowed_hits = int(math.floor(tol * samples_self.shape[0]))
            hits = 0
            for point in samples_self:
                if other.contains_point(point):
                    hits += 1
                    if hits > allowed_hits:
                        return True
        samples_other = other.sample_surface_points(resolution)
        if samples_other.size:
            allowed_hits = int(math.floor(tol * samples_other.shape[0]))
            hits = 0
            for point in samples_other:
                if self.contains_point(point):
                    hits += 1
                    if hits > allowed_hits:
                        return True
        return False

    # ---- 内部工具 ----
    def _setup_static_polygon(self) -> None:
        if self.shape == "truncated_cube":
            self._init_truncated_cube_static()
        else:
            self._init_generic_static()

    def _init_generic_static(self) -> None:
        circumradius = self.edge_length / (2.0 * math.sin(math.pi / self.sides))
        scaled_radius = circumradius / self.radius
        angles = np.linspace(0.0, 2.0 * math.pi, self.sides, endpoint=False)
        self._plane_vertices = np.stack(
            (scaled_radius * np.cos(angles), scaled_radius * np.sin(angles)), axis=1
        )
        self._bounding_radius = float(np.max(np.linalg.norm(self._plane_vertices, axis=1)))
        self._sample_cache.clear()
        self._canonical_truncated_cube = None
        self._truncated_faces = None

    def _init_truncated_cube_static(self) -> None:
        sqrt2 = math.sqrt(2.0)
        canonical: List[Tuple[float, float, float]] = []
        for perm in set(permutations([1 + sqrt2, 1, 1])):
            for signs in product([-1, 1], repeat=3):
                x, y, z = (sign * val for sign, val in zip(signs, perm))
                canonical.append((y, z, -(x - 1)))
        canonical = sorted(set(canonical))
        canonical = np.array(canonical, dtype=float)
        scale = self.edge_length / 2.0
        canonical *= scale

        eps = 1e-6
        base_idx = np.where(np.abs(canonical[:, 2]) < eps)[0]
        base_idx = base_idx[np.argsort(np.arctan2(canonical[base_idx, 1], canonical[base_idx, 0]))]
        top_idx = np.where(np.abs(canonical[:, 2] - self.edge_length) < eps)[0]
        top_idx = top_idx[np.argsort(np.arctan2(canonical[top_idx, 1], canonical[top_idx, 0]))]
        xpos_idx = np.where(np.abs(canonical[:, 0] - self.edge_length / 2.0) < eps)[0]
        xpos_idx = xpos_idx[np.argsort(np.arctan2(canonical[xpos_idx, 2], canonical[xpos_idx, 1]))]
        xneg_idx = np.where(np.abs(canonical[:, 0] + self.edge_length / 2.0) < eps)[0]
        xneg_idx = xneg_idx[np.argsort(np.arctan2(canonical[xneg_idx, 2], canonical[xneg_idx, 1]))]
        ypos_idx = np.where(np.abs(canonical[:, 1] - self.edge_length / 2.0) < eps)[0]
        ypos_idx = ypos_idx[np.argsort(np.arctan2(canonical[ypos_idx, 2], canonical[ypos_idx, 0]))]
        yneg_idx = np.where(np.abs(canonical[:, 1] + self.edge_length / 2.0) < eps)[0]
        yneg_idx = yneg_idx[np.argsort(np.arctan2(canonical[yneg_idx, 2], canonical[yneg_idx, 0]))]

        base_coords = canonical[base_idx, :2]
        self._plane_vertices = base_coords / self.radius
        self._bounding_radius = float(np.max(np.linalg.norm(self._plane_vertices, axis=1)))
        self._sample_cache.clear()
        self._height = None
        self._top_scale = None

        # 构建三角面索引
        edges = set()
        for i in range(len(canonical)):
            for j in range(i + 1, len(canonical)):
                dist = np.linalg.norm(canonical[i] - canonical[j])
                if abs(dist - self.edge_length) < 1e-5:
                    edges.add((i, j))
                    edges.add((j, i))
        triangles_set = set()
        for i in range(len(canonical)):
            for j in range(i + 1, len(canonical)):
                if (i, j) not in edges:
                    continue
                for k in range(j + 1, len(canonical)):
                    if (i, k) in edges and (j, k) in edges:
                        tri = tuple(sorted((i, j, k)))
                        triangles_set.add(tri)
        triangle_faces = sorted(triangles_set)

        octagon_faces = [
            base_idx.tolist(),
            top_idx.tolist(),
            xpos_idx.tolist(),
            xneg_idx.tolist(),
            ypos_idx.tolist(),
            yneg_idx.tolist(),
        ]

        self._canonical_truncated_cube = canonical
        self._truncated_faces = {
            "octagons": octagon_faces,
            "triangles": triangle_faces,
            "base_set": set(base_idx.tolist()),
        }

    def _update_basis(self) -> None:
        u, v, n = orthonormal_basis(self.normal)
        u_rot, v_rot = rotate_basis(u, v, self.rotation)
        self._basis = (u_rot, v_rot, n)

    def _rebuild_polyhedron(self) -> None:
        if self.shape == "truncated_cube":
            self._build_truncated_cube_geometry()
        else:
            self._build_generic_polyhedron()

    def _build_generic_polyhedron(self) -> None:
        basis_u, basis_v, basis_n = self._basis
        center = self.center
        base_vertices = [lift_to_sphere(tuple(v), self._basis, self.radius) for v in self._plane_vertices]
        vertices = [center] + base_vertices
        i_list: List[int] = []
        j_list: List[int] = []
        k_list: List[int] = []

        if self._height is not None and self._top_scale is not None:
            top_center = center + self._height * basis_n
            top_vertices = []
            for sx, sy in self._plane_vertices:
                offset = self.radius * (self._top_scale * sx * basis_u + self._top_scale * sy * basis_v)
                top_vertices.append(top_center + offset)
            vertices.extend(top_vertices)
            vertices.append(top_center)

            n = len(self._plane_vertices)
            base_offset = 1
            top_offset = 1 + n
            top_center_idx = top_offset + n

            for idx in range(n):
                i_list.append(0)
                j_list.append(base_offset + idx)
                k_list.append(base_offset + (idx + 1) % n)

            for idx in range(n):
                b1 = base_offset + idx
                b2 = base_offset + (idx + 1) % n
                t1 = top_offset + idx
                t2 = top_offset + (idx + 1) % n
                i_list.extend([b1, b1])
                j_list.extend([b2, t2])
                k_list.extend([t2, t1])

            for idx in range(n):
                i_list.append(top_center_idx)
                j_list.append(top_offset + (idx + 1) % n)
                k_list.append(top_offset + idx)
        else:
            n = len(self._plane_vertices)
            for idx in range(n):
                i_list.append(0)
                j_list.append(idx + 1)
                k_list.append((idx + 1) % n + 1)

        self._mesh_vertices = np.array(vertices, dtype=float)
        self._mesh_triangles = (
            np.array(i_list, dtype=int),
            np.array(j_list, dtype=int),
            np.array(k_list, dtype=int),
        )

    def _build_truncated_cube_geometry(self) -> None:
        if self._canonical_truncated_cube is None or self._truncated_faces is None:
            raise RuntimeError("truncated cube 数据未初始化")
        basis_u, basis_v, basis_n = self._basis
        center = self.center
        canonical = self._canonical_truncated_cube
        faces = self._truncated_faces
        base_set = faces["base_set"]

        world_vertices: List[np.ndarray] = []
        for idx, (px, py, pz) in enumerate(canonical):
            if idx in base_set:
                sx = px / self.radius
                sy = py / self.radius
                world_point = lift_to_sphere((sx, sy), self._basis, self.radius)
            else:
                world_point = center + px * basis_u + py * basis_v + pz * basis_n
            world_vertices.append(world_point)
        world_vertices = np.array(world_vertices, dtype=float)

        triangle_list: List[Tuple[int, int, int]] = []
        for face in faces["octagons"]:
            anchor = face[0]
            for i in range(1, len(face) - 1):
                triangle_list.append((anchor, face[i], face[i + 1]))
        triangle_list.extend(faces["triangles"])

        i_idx = np.array([tri[0] for tri in triangle_list], dtype=int)
        j_idx = np.array([tri[1] for tri in triangle_list], dtype=int)
        k_idx = np.array([tri[2] for tri in triangle_list], dtype=int)

        self._mesh_vertices = world_vertices
        self._mesh_triangles = (i_idx, j_idx, k_idx)
