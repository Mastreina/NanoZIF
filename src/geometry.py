"""几何工具函数，负责球面与切平面的坐标转换。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    """返回单位化后的向量。"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("无法归一化零向量")
    return vec / norm


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """在球面上均匀采样单位向量。"""
    phi = rng.uniform(0.0, 2.0 * math.pi)
    z = rng.uniform(-1.0, 1.0)
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([r_xy * math.cos(phi), r_xy * math.sin(phi), z], dtype=float)


def orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定法向量，构造正交归一基(u, v, n)。"""
    n = normalize(normal)
    # 选择与n不平行的向量作为辅助
    if abs(n[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    u = normalize(np.cross(helper, n))
    v = np.cross(n, u)
    return u, v, n


def rotate_basis(u: np.ndarray, v: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """绕法向量旋转切平面基，返回新的(u, v)。"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u_rot = cos_a * u + sin_a * v
    v_rot = -sin_a * u + cos_a * v
    return u_rot, v_rot


def gnomonic_project_point(
    point: np.ndarray,
    basis: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[float, float]:
    """将球面点投影到切平面，返回切平面坐标(sx, sy)。"""
    u, v, n = basis
    dot = float(np.dot(point, n))
    if dot <= 0:
        # 在切平面以下，使用一个大值代表不可达
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
    """将切平面坐标(以球半径归一化)投影回球面坐标。"""
    sx, sy = plane_xy
    u, v, n = basis
    tangent_point = radius * n
    candidate = tangent_point + radius * (sx * u + sy * v)
    return radius * normalize(candidate)


def fibonacci_sphere(samples: int, radius: float) -> np.ndarray:
    """生成Fibonacci球面采样点，用于覆盖度估计。"""
    indices = np.arange(samples, dtype=float)
    phi = (1 + math.sqrt(5)) / 2  # 黄金比例
    theta = 2 * math.pi * indices / phi
    z = 1 - 2 * (indices + 0.5) / samples
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    points = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)
    return radius * points


@dataclass(frozen=True)
class SpherePoint:
    """球面点的极坐标描述。"""

    theta: float  # 0..pi
    phi: float  # -pi..pi

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
