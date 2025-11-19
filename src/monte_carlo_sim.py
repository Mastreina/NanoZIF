"""蒙特卡洛铺展主循环实现。"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .geometry import fibonacci_sphere, random_unit_vector
from .tile import RegularPolygonPatch


def _random_tangent_direction(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """生成与法向量正交的随机单位向量。"""
    trial = rng.normal(size=3)
    trial -= np.dot(trial, normal) * normal
    norm = np.linalg.norm(trial)
    if norm == 0:
        return _random_tangent_direction(normal, rng)
    return trial / norm


def _perturb_normal(normal: np.ndarray, step: float, rng: np.random.Generator) -> np.ndarray:
    """在切平面方向旋转一定角度，返回新法向量。"""
    axis = _random_tangent_direction(normal, rng)
    cos_step = math.cos(step)
    sin_step = math.sin(step)
    return cos_step * normal + sin_step * axis


@dataclass
class MonteCarloConfig:
    radius: float = 2.55  # nm
    tile_side: float = 0.6  # nm
    shape: str = "square"
    insertion_probability: float = 0.55
    max_steps: int = 200_000
    stall_steps: int = 20_000
    translation_step: float = math.radians(3.0)
    rotation_step: float = math.radians(5.0)
    overlap_resolution: int = 7
    overlap_tolerance: float = 0.0
    coverage_samples: int = 20_000
    rng_seed: Optional[int] = None
    cooperative_insert_attempts: int = 8
    cooperative_move_scale: float = 1.25
    post_insert_relax_steps: int = 80
    relax_translation_scale: float = 0.45
    relax_rotation_scale: float = 0.45
    neighbor_relax_angle: float = math.radians(32.0)
    energy_contact_scale: float = 0.9
    energy_target_scale: float = 1.1
    energy_temperature_start: float = 1.2
    energy_temperature_end: float = 0.05
    energy_batches: int = 3
    energy_moves_per_batch: int = 120
    energy_acceptance_softness: float = 1.0
    force_insert_enabled: bool = True
    force_insert_iterations: int = 300
    force_insert_step: float = 0.25
    force_insert_rotation_sigma: float = math.radians(6.0)
    force_insert_neighbor_limit: int = 24
    checkpoint_interval: int = 0
    gap_sample_size: int = 48
    gap_insert_bias: float = 0.85
    targeted_insert_trigger: int = 4_000
    targeted_insert_attempts: int = 4
    targeted_samples: int = 6_000
    global_relax_trigger: int = 12_000
    initial_layout: str = "none"
    initial_count: int = 0
    initial_caspar_h: int = 1
    initial_caspar_k: int = 0


@dataclass
class SimulationResult:
    tiles: List[RegularPolygonPatch]
    coverage: float
    coverage_points: np.ndarray
    coverage_mask: np.ndarray


class SphereSurfaceMonteCarlo:
    """负责球面正方体铺展的蒙特卡洛求解器。"""

    def __init__(
        self,
        config: MonteCarloConfig,
        *,
        checkpoint_path: Optional[Path] = None,
        resume_state: Optional[dict] = None,
    ) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.rng_seed)
        self.tiles: List[RegularPolygonPatch] = []
        self.checkpoint_path = checkpoint_path
        self._steps_completed = 0
        self._resume_steps_without_change = 0
        self._shape = config.shape.lower()
        self._sides = self._resolve_sides(self._shape)
        reference_tile = RegularPolygonPatch(
            radius=self.cfg.radius,
            edge_length=self.cfg.tile_side,
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            rotation=0.0,
            sides=self._sides,
            shape=self._shape,
        )
        plane_polygon = reference_tile.to_plane_polygon()
        self._plane_polygon_area = self._polygon_area_2d(plane_polygon)
        self._approx_tile_area = max(1e-9, self._plane_polygon_area * (self.cfg.radius ** 2))
        vertices = reference_tile.polygon_vertices()
        corner_angles = [
            math.acos(
                np.clip(
                    float(np.dot(reference_tile.normal, vertex / self.cfg.radius)),
                    -1.0,
                    1.0,
                )
            )
            for vertex in vertices
        ]
        self._corner_angle = max(corner_angles)
        self._exclusion_angle = 2.0 * self._corner_angle
        self._cos_exclusion = math.cos(self._exclusion_angle)
        self._neighbor_cos = math.cos(self.cfg.neighbor_relax_angle)
        self._contact_distance = max(1e-9, self.cfg.tile_side * self.cfg.energy_contact_scale)
        self._target_distance = max(self._contact_distance * 1.01, self.cfg.tile_side * self.cfg.energy_target_scale)
        self._contact_angle = self._contact_distance / self.cfg.radius
        self._target_angle = self._target_distance / self.cfg.radius
        targeted_samples = max(0, self.cfg.targeted_samples)
        if targeted_samples > 0 and self.cfg.radius > 0:
            self._target_points_cache = fibonacci_sphere(targeted_samples, self.cfg.radius)
        else:
            self._target_points_cache = np.zeros((0, 3), dtype=float)
        self._last_targeted_insert_step = -10**9

        if resume_state is not None:
            self._load_state(resume_state)
        else:
            self._seed_initial_layout()

    @staticmethod
    def _resolve_sides(shape: str) -> int:
        shape_map = {
            "square": 4,
            "quad": 4,
            "hexagon": 6,
            "hexagonal": 6,
            "hexagonal_prism": 6,
            "hex": 6,
            "octagon": 8,
            "truncated_octahedron": 8,
            "truncated_cube": 8,
        }
        if shape not in shape_map:
            raise ValueError(f"不支持的形状: {shape}")
        return shape_map[shape]

    @staticmethod
    def _polygon_area_2d(points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    # 状态管理 --------------------------------------------------------------
    @staticmethod
    def load_checkpoint(path: Path) -> dict:
        data = np.load(path, allow_pickle=False)
        rng_state_raw = data["rng_state"]
        rng_state_json = str(rng_state_raw.item() if hasattr(rng_state_raw, "item") else rng_state_raw)
        shape_sides = None
        if "shape_sides" in data.files:
            raw = data["shape_sides"]
            shape_sides = int(raw.item() if hasattr(raw, "item") else raw)
        shape_name = None
        if "shape_name" in data.files:
            raw_name = data["shape_name"]
            shape_name = str(raw_name.item() if hasattr(raw_name, "item") else raw_name)
        state = {
            "radius": float(data["radius"]),
            "tile_side": float(data["tile_side"]),
            "normals": np.asarray(data["normals"], dtype=float),
            "rotations": np.asarray(data["rotations"], dtype=float),
            "step": int(data["step"]),
            "steps_without_change": int(data["steps_without_change"]),
            "rng_state": json.loads(rng_state_json),
            "shape_sides": shape_sides,
            "shape_name": shape_name,
        }
        return state

    def _load_state(self, state: dict) -> None:
        radius = float(state.get("radius", self.cfg.radius))
        tile_side = float(state.get("tile_side", self.cfg.tile_side))
        if abs(radius - self.cfg.radius) > 1e-9 or abs(tile_side - self.cfg.tile_side) > 1e-9:
            raise ValueError("Checkpoint配置与当前模拟参数不匹配")
        shape_sides_state = state.get("shape_sides")
        shape_sides = int(shape_sides_state) if shape_sides_state is not None else self._sides
        if shape_sides != self._sides:
            raise ValueError("Checkpoint形状与当前配置不匹配")
        shape_name = state.get("shape_name")
        if shape_name is not None and shape_name.lower() != self._shape:
            raise ValueError("Checkpoint形状名称与当前配置不匹配")
        normals = np.asarray(state.get("normals", []), dtype=float)
        rotations = np.asarray(state.get("rotations", []), dtype=float)
        self.tiles = [
            RegularPolygonPatch(
                radius=self.cfg.radius,
                edge_length=self.cfg.tile_side,
                normal=normal,
                rotation=float(rotation),
                sides=self._sides,
                shape=self._shape,
            )
            for normal, rotation in zip(normals, rotations)
        ]
        self._steps_completed = int(state.get("step", 0))
        self._resume_steps_without_change = int(state.get("steps_without_change", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _save_checkpoint(self, step: int, steps_without_change: int) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        normals = np.array([tile.normal for tile in self.tiles], dtype=float)
        rotations = np.array([tile.rotation for tile in self.tiles], dtype=float)
        rng_state_json = json.dumps(self.rng.bit_generator.state)
        np.savez_compressed(
            self.checkpoint_path,
            radius=self.cfg.radius,
            tile_side=self.cfg.tile_side,
            shape_sides=self._sides,
            shape_name=np.array(self._shape, dtype="U"),
            normals=normals,
            rotations=rotations,
            step=step,
            steps_without_change=steps_without_change,
            rng_state=np.array(rng_state_json, dtype="U"),
        )

    # 插入和移动 ------------------------------------------------------------
    def _tile_overlaps(self, candidate: RegularPolygonPatch, collect_indices: bool = False):
        overlaps: List[int] = []
        for idx, tile in enumerate(self.tiles):
            if tile is candidate:
                continue
            dot = float(np.dot(tile.normal, candidate.normal))
            if dot <= self._cos_exclusion:
                continue
            if candidate.overlaps_with(
                tile,
                resolution=self.cfg.overlap_resolution,
                tolerance=self.cfg.overlap_tolerance,
            ):
                if not collect_indices:
                    return True
                overlaps.append(idx)
        return overlaps if collect_indices else False

    def _pair_energy(self, tile_a: RegularPolygonPatch, tile_b: RegularPolygonPatch) -> float:
        dot = float(np.dot(tile_a.normal, tile_b.normal))
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)
        distance = self.cfg.radius * angle
        if distance <= 1e-12:
            return 0.0
        if distance < self._contact_distance:
            gap = (self._contact_distance - distance) / self._contact_distance
            return 4.0 * gap * gap
        diff = distance - self._target_distance
        return (diff / self._target_distance) ** 2

    def _tile_energy_at(self, tile: RegularPolygonPatch) -> float:
        total = 0.0
        for other in self.tiles:
            if other is tile:
                continue
            total += self._pair_energy(tile, other)
        return total

    def _candidate_energy(self, candidate: RegularPolygonPatch) -> float:
        total = 0.0
        for other in self.tiles:
            total += self._pair_energy(candidate, other)
        return total

    def _estimate_initial_count(self) -> int:
        if self.cfg.initial_count > 0:
            return self.cfg.initial_count
        sphere_area = 4.0 * math.pi * (self.cfg.radius ** 2)
        if self._approx_tile_area <= 0.0:
            return 0
        coverage_factor = 0.95
        raw = coverage_factor * sphere_area / self._approx_tile_area
        return max(1, int(round(raw)))

    def _generate_fibonacci_normals(self, count: int) -> List[np.ndarray]:
        if count <= 0:
            return []
        points = fibonacci_sphere(count, 1.0)
        normals: List[np.ndarray] = []
        for vec in points:
            norm = np.linalg.norm(vec)
            if norm <= 1e-9:
                continue
            normals.append(vec / norm)
        return normals

    @staticmethod
    def _icosahedron_geometry() -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        vertices = np.array(
            [
                (-1, phi, 0),
                (1, phi, 0),
                (-1, -phi, 0),
                (1, -phi, 0),
                (0, -1, phi),
                (0, 1, phi),
                (0, -1, -phi),
                (0, 1, -phi),
                (phi, 0, -1),
                (phi, 0, 1),
                (-phi, 0, -1),
                (-phi, 0, 1),
            ],
            dtype=float,
        )
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        faces: List[Tuple[int, int, int]] = [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ]
        return vertices, faces

    def _generate_caspar_klug_normals(self, h: int, k: int) -> List[np.ndarray]:
        frequency = max(1, h + k)
        vertices, faces = self._icosahedron_geometry()
        candidate_points: List[np.ndarray] = []
        for face in faces:
            A, B, C = (vertices[idx] for idx in face)
            for i in range(frequency + 1):
                for j in range(frequency + 1 - i):
                    k_share = frequency - i - j
                    point = (i * B + j * C + k_share * A) / max(1, frequency)
                    candidate_points.append(point)
        normals: List[np.ndarray] = []
        seen: set[Tuple[int, int, int]] = set()
        for point in candidate_points:
            norm = np.linalg.norm(point)
            if norm <= 1e-9:
                continue
            vec = point / norm
            key = tuple(int(round(coord * 1e6)) for coord in vec)
            if key in seen:
                continue
            seen.add(key)
            normals.append(vec)
        return normals

    def _seed_initial_layout(self) -> None:
        layout = self.cfg.initial_layout.strip().lower()
        if layout == "none":
            return
        target = self._estimate_initial_count()
        normals: List[np.ndarray]
        if layout == "fibonacci":
            normals = self._generate_fibonacci_normals(target)
        elif layout == "caspar_klug":
            normals = self._generate_caspar_klug_normals(
                max(0, self.cfg.initial_caspar_h),
                max(0, self.cfg.initial_caspar_k),
            )
            if target > 0 and len(normals) > target:
                order = self.rng.permutation(len(normals))
                normals = [normals[int(idx)] for idx in order[:target]]
        else:
            return
        if not normals:
            return
        order = self.rng.permutation(len(normals))
        inserted = 0
        for raw_idx in order:
            idx = int(raw_idx)
            normal_vec = normals[idx]
            rotation = self.rng.uniform(0.0, 2.0 * math.pi)
            tile = RegularPolygonPatch(
                radius=self.cfg.radius,
                edge_length=self.cfg.tile_side,
                normal=normal_vec,
                rotation=rotation,
                sides=self._sides,
                shape=self._shape,
            )
            if self._tile_overlaps(tile):
                continue
            self.tiles.append(tile)
            inserted += 1
        if inserted:
            self._annealed_relaxation()

    def _tile_normals_matrix(self) -> np.ndarray:
        if not self.tiles:
            return np.zeros((0, 3), dtype=float)
        return np.array([tile.normal for tile in self.tiles], dtype=float)

    def _select_gap_oriented_normal(self) -> np.ndarray:
        if not self.tiles:
            return random_unit_vector(self.rng)
        normals = self._tile_normals_matrix()
        sample_count = max(1, self.cfg.gap_sample_size)
        best_candidate: np.ndarray | None = None
        best_score = -2.0
        for _ in range(sample_count):
            candidate = random_unit_vector(self.rng)
            dots = normals @ candidate
            max_dot = float(np.max(dots)) if dots.size else -1.0
            score = -max_dot
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate if best_candidate is not None else random_unit_vector(self.rng)

    def _attempt_insert(self, *, forced_normal: np.ndarray | None = None) -> bool:
        if forced_normal is not None:
            normal_vec = np.array(forced_normal, dtype=float)
            norm = np.linalg.norm(normal_vec)
            if norm <= 1e-12:
                return False
            normal = normal_vec / norm
        else:
            use_gap = (
                bool(self.tiles)
                and self.cfg.gap_sample_size > 1
                and self.cfg.gap_insert_bias > 0.0
                and self.rng.random() < self.cfg.gap_insert_bias
            )
            normal = self._select_gap_oriented_normal() if use_gap else random_unit_vector(self.rng)
        rotation = self.rng.uniform(0.0, 2.0 * math.pi)
        tile = RegularPolygonPatch(
            radius=self.cfg.radius,
            edge_length=self.cfg.tile_side,
            normal=normal,
            rotation=rotation,
            sides=self._sides,
            shape=self._shape,
        )
        overlaps = self._tile_overlaps(tile, collect_indices=True)
        if overlaps:
            if self.cfg.cooperative_insert_attempts <= 0:
                return False
            if not self._attempt_cooperative_insert(tile, overlaps):
                return False
            if self._tile_overlaps(tile):
                if not self.cfg.force_insert_enabled:
                    return False
                return self._force_insert(tile)
        self.tiles.append(tile)
        new_idx = len(self.tiles) - 1
        self._local_relaxation(new_idx)
        self._annealed_relaxation(new_idx)
        return True

    def _attempt_targeted_insert(self) -> bool:
        points = getattr(self, "_target_points_cache", None)
        if points is None or points.size == 0:
            return False
        mask = np.zeros(points.shape[0], dtype=bool)
        for tile in self.tiles:
            inside = np.array([tile.contains_point(point) for point in points], dtype=bool)
            mask |= inside
            if mask.all():
                break
        if mask.all():
            return False
        uncovered = points[~mask]
        if uncovered.size == 0:
            return False
        idx = int(self.rng.integers(0, uncovered.shape[0]))
        direction = uncovered[idx]
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            return False
        normal = direction / norm
        return self._attempt_insert(forced_normal=normal)

    def _force_insert(self, tile: RegularPolygonPatch) -> bool:
        if not self.cfg.force_insert_enabled:
            return False
        self.tiles.append(tile)
        new_idx = len(self.tiles) - 1
        neighbors = self._find_neighbors(new_idx)
        if self.cfg.force_insert_neighbor_limit > 0 and len(neighbors) > self.cfg.force_insert_neighbor_limit:
            neighbors = list(
                self.rng.choice(neighbors, size=self.cfg.force_insert_neighbor_limit, replace=False)
            )
        working = [new_idx] + neighbors
        saved_tiles = {idx: self.tiles[idx].copy_with() for idx in working if idx != new_idx}
        success = False

        for _ in range(max(1, self.cfg.force_insert_iterations)):
            overlaps = self._tile_overlaps(self.tiles[new_idx], collect_indices=True)
            if not overlaps:
                success = True
                break
            active = list({new_idx, *working, *overlaps})
            for idx in active:
                if idx != new_idx and idx not in saved_tiles:
                    saved_tiles[idx] = self.tiles[idx].copy_with()

            forces = {idx: np.zeros(3, dtype=float) for idx in active}
            for idx in active:
                tile_i = self.tiles[idx]
                normal_i = tile_i.normal
                neighbors_i = self._find_neighbors(idx)
                if self.cfg.force_insert_neighbor_limit > 0 and len(neighbors_i) > self.cfg.force_insert_neighbor_limit:
                    neighbors_i = list(
                        self.rng.choice(neighbors_i, size=self.cfg.force_insert_neighbor_limit, replace=False)
                    )
                for j in neighbors_i:
                    if j == idx:
                        continue
                    tile_j = self.tiles[j]
                    normal_j = tile_j.normal
                    dot = np.clip(float(np.dot(normal_i, normal_j)), -1.0, 1.0)
                    angle = math.acos(dot)
                    distance = self.cfg.radius * angle
                    if distance <= 1e-9:
                        continue
                    diff = normal_i - normal_j
                    tangent = diff - np.dot(diff, normal_i) * normal_i
                    norm_tangent = np.linalg.norm(tangent)
                    if norm_tangent == 0.0:
                        continue
                    tangent /= norm_tangent
                    if distance < self._contact_distance:
                        weight = (self._contact_distance - distance) / self._contact_distance
                    elif distance > self._target_distance:
                        weight = -0.2 * (distance - self._target_distance) / self._target_distance
                    else:
                        weight = 0.0
                    forces[idx] += weight * tangent

            moved = False
            for idx in active:
                force = forces[idx]
                normal = self.tiles[idx].normal
                force_tangent = force - np.dot(force, normal) * normal
                norm_force = np.linalg.norm(force_tangent)
                if norm_force <= 1e-9:
                    continue
                step = self.cfg.force_insert_step * force_tangent / norm_force
                new_normal = normal + step
                norm_new = np.linalg.norm(new_normal)
                if norm_new <= 1e-9:
                    continue
                new_normal /= norm_new
                new_rotation = (
                    self.tiles[idx].rotation + self.cfg.force_insert_rotation_sigma * self.rng.normal()
                ) % (2.0 * math.pi)
                self.tiles[idx] = self.tiles[idx].copy_with(normal=new_normal, rotation=new_rotation)
                moved = True

            if not moved:
                break

        if success:
            self._local_relaxation(new_idx)
            self._annealed_relaxation(new_idx)
            return True

        self.tiles.pop()
        for idx, original in saved_tiles.items():
            if idx < len(self.tiles):
                self.tiles[idx] = original
        return False

    def _attempt_move(self) -> bool:
        if not self.tiles:
            return False
        idx = self.rng.integers(0, len(self.tiles))
        return self._attempt_scaled_move(int(idx), 1.0, 1.0)

    def _attempt_scaled_move(
        self,
        idx: int,
        translation_scale: float,
        rotation_scale: float,
        *,
        use_energy: bool = False,
        temperature: float = 1.0,
    ) -> bool:
        if idx < 0 or idx >= len(self.tiles):
            return False
        current = self.tiles[idx]
        translation_base = self.cfg.translation_step * max(translation_scale, 0.0)
        rotation_base = self.cfg.rotation_step * max(rotation_scale, 0.0)
        translation_angle = translation_base * (0.5 + self.rng.random()) if translation_base > 0 else 0.0
        rotation_delta = rotation_base * self.rng.normal() if rotation_base > 0 else 0.0
        if translation_angle == 0.0 and rotation_delta == 0.0:
            return False
        new_normal = (
            current.normal
            if translation_angle == 0.0
            else _perturb_normal(current.normal, translation_angle, self.rng)
        )
        new_rotation = (current.rotation + rotation_delta) % (2.0 * math.pi)
        proposal = current.copy_with(normal=new_normal, rotation=new_rotation)

        current_energy = self._tile_energy_at(current) if use_energy else 0.0
        removed = self.tiles.pop(idx)
        try:
            if self._tile_overlaps(proposal):
                return False
            if use_energy:
                proposal_energy = self._candidate_energy(proposal)
                delta = proposal_energy - current_energy
                if delta > 0.0:
                    base_temp = max(temperature, 1e-6)
                    effective_temp = max(
                        base_temp * max(self.cfg.energy_acceptance_softness, 1e-6),
                        1e-6,
                    )
                    if self.rng.random() >= math.exp(-delta / effective_temp):
                        return False
            self.tiles.insert(idx, proposal)
            return True
        finally:
            if len(self.tiles) <= idx or self.tiles[idx] is not proposal:
                self.tiles.insert(idx, removed)

    def _attempt_cooperative_insert(self, candidate: RegularPolygonPatch, overlaps: List[int]) -> bool:
        current_overlaps = list(overlaps)
        for _ in range(max(0, self.cfg.cooperative_insert_attempts)):
            improved = False
            if not current_overlaps:
                return True
            order = self.rng.permutation(current_overlaps)
            for raw_idx in order:
                idx = int(raw_idx)
                success = self._attempt_scaled_move(
                    idx,
                    self.cfg.relax_translation_scale * self.cfg.cooperative_move_scale,
                    self.cfg.relax_rotation_scale * self.cfg.cooperative_move_scale,
                )
                if success:
                    improved = True
            current_overlaps = self._tile_overlaps(candidate, collect_indices=True)
            if not current_overlaps:
                return True
            if not improved:
                break
        return False

    def _find_neighbors(self, center_idx: int) -> List[int]:
        if center_idx < 0 or center_idx >= len(self.tiles):
            return []
        center = self.tiles[center_idx]
        neighbors: List[int] = []
        for idx, tile in enumerate(self.tiles):
            if idx == center_idx:
                continue
            if float(np.dot(center.normal, tile.normal)) >= self._neighbor_cos:
                neighbors.append(idx)
        return neighbors

    def _local_relaxation(self, center_idx: int) -> None:
        if self.cfg.post_insert_relax_steps <= 0:
            return
        neighbors = self._find_neighbors(center_idx)
        pool = neighbors + [center_idx] if neighbors else [center_idx]
        if not pool:
            return
        for _ in range(self.cfg.post_insert_relax_steps):
            idx = int(self.rng.choice(pool))
            move_scale = 0.5 + 0.5 * self.rng.random()
            self._attempt_scaled_move(
                idx,
                self.cfg.relax_translation_scale * move_scale,
                self.cfg.relax_rotation_scale * move_scale,
            )

    def _annealed_relaxation(self, center_idx: Optional[int] = None) -> None:
        if self.cfg.energy_batches <= 0 or self.cfg.energy_moves_per_batch <= 0 or not self.tiles:
            return
        if center_idx is None:
            pool = list(range(len(self.tiles)))
        else:
            neighbors = self._find_neighbors(center_idx)
            pool = list({center_idx, *neighbors})
        if not pool:
            pool = list(range(len(self.tiles)))
        start_temp = max(1e-6, self.cfg.energy_temperature_start)
        end_temp = max(1e-6, self.cfg.energy_temperature_end)
        if self.cfg.energy_batches == 1:
            temps = [end_temp]
        else:
            ratio = end_temp / start_temp if start_temp > 0 else 1.0
            temps = [
                start_temp * (ratio ** (batch / max(1, self.cfg.energy_batches - 1)))
                for batch in range(self.cfg.energy_batches)
            ]
        for temp in temps:
            for _ in range(self.cfg.energy_moves_per_batch):
                idx = int(self.rng.choice(pool))
                move_scale = 0.4 + 0.6 * self.rng.random()
                self._attempt_scaled_move(
                    idx,
                    self.cfg.relax_translation_scale * move_scale,
                    self.cfg.relax_rotation_scale * move_scale,
                    use_energy=True,
                    temperature=temp,
                )

    # 估计覆盖率 ------------------------------------------------------------
    def _estimate_coverage(self) -> SimulationResult:
        if not self.tiles:
            points = fibonacci_sphere(self.cfg.coverage_samples, self.cfg.radius)
            mask = np.zeros(points.shape[0], dtype=bool)
            return SimulationResult(tiles=[], coverage=0.0, coverage_points=points, coverage_mask=mask)

        points = fibonacci_sphere(self.cfg.coverage_samples, self.cfg.radius)
        mask = np.zeros(points.shape[0], dtype=bool)
        for tile in self.tiles:
            inside = np.apply_along_axis(tile.contains_point, 1, points)
            mask |= inside
        coverage = mask.mean()
        return SimulationResult(
            tiles=list(self.tiles),
            coverage=coverage,
            coverage_points=points,
            coverage_mask=mask,
        )

    # 主入口 ---------------------------------------------------------------
    def run(self, show_progress: bool = False) -> SimulationResult:
        if self._steps_completed >= self.cfg.max_steps:
            return self._estimate_coverage()

        steps_without_change = self._resume_steps_without_change
        progress_bar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError:  # pragma: no cover - 仅在缺少依赖时触发
                tqdm = None  # type: ignore
            if tqdm is not None:
                progress_bar = tqdm(
                    total=self.cfg.max_steps,
                    initial=self._steps_completed,
                    desc=f"Monte Carlo (tiles={len(self.tiles)})",
                    unit="step",
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.stdout,
                )

        for step in range(self._steps_completed, self.cfg.max_steps):
            move_type = "insert" if self.rng.random() < self.cfg.insertion_probability else "move"
            if move_type == "insert":
                changed = self._attempt_insert()
            else:
                changed = self._attempt_move()

            if not changed:
                steps_without_change += 1
                triggered = (
                    self.cfg.targeted_insert_trigger > 0
                    and steps_without_change >= self.cfg.targeted_insert_trigger
                    and (step - self._last_targeted_insert_step) >= self.cfg.targeted_insert_trigger
                )
                if triggered:
                    self._last_targeted_insert_step = step
                    targeted_success = False
                    attempts = max(1, self.cfg.targeted_insert_attempts)
                    for _ in range(attempts):
                        if self._attempt_targeted_insert():
                            targeted_success = True
                            break
                    if targeted_success:
                        changed = True
                        steps_without_change = 0
                if not changed and steps_without_change >= self.cfg.stall_steps:
                    self._steps_completed = step + 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                    break
            else:
                steps_without_change = 0

            if changed and progress_bar is not None and hasattr(progress_bar, "set_description"):
                progress_bar.set_description(f"Monte Carlo (tiles={len(self.tiles)})", refresh=False)

            self._steps_completed = step + 1
            if progress_bar is not None:
                progress_bar.update(1)
            if (
                self.cfg.global_relax_trigger > 0
                and self._steps_completed % self.cfg.global_relax_trigger == 0
            ):
                self._annealed_relaxation()
            if (
                self.cfg.checkpoint_interval > 0
                and self.checkpoint_path is not None
                and (self._steps_completed % self.cfg.checkpoint_interval == 0)
            ):
                self._save_checkpoint(self._steps_completed, steps_without_change)

        if progress_bar is not None:
            progress_bar.close()

        self._annealed_relaxation()

        if self.checkpoint_path is not None:
            self._save_checkpoint(self._steps_completed, steps_without_change)
        self._resume_steps_without_change = 0

        return self._estimate_coverage()
