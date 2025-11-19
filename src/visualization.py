"""封装结果可视化：二维投影与三维展示。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import SpherePoint
from .monte_carlo_sim import SimulationResult


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _convert_vertices_to_lon_lat(vertices: np.ndarray) -> np.ndarray:
    """将球面坐标转换为(经度, 纬度)角度制。"""
    vertices_norm = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    theta = np.arccos(np.clip(vertices_norm[:, 2], -1.0, 1.0))
    phi = np.arctan2(vertices_norm[:, 1], vertices_norm[:, 0])
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)
    return np.stack((lon, lat), axis=1)


def plot_equirectangular(result: SimulationResult, output: Path) -> None:
    """生成二维等距反射投影图。"""
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_title("Sphere Coverage (Equirectangular Projection)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    cmap = cm.get_cmap("viridis")
    for idx, tile in enumerate(result.tiles):
        vertices = tile.polygon_vertices()
        lon_lat = _convert_vertices_to_lon_lat(vertices)
        polygon = patches.Polygon(
            lon_lat,
            closed=True,
            facecolor=cmap(idx / max(1, len(result.tiles))),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.6,
        )
        ax.add_patch(polygon)
        center = SpherePoint.from_vector(tile.center)
        ax.plot(
            [math.degrees(center.phi)],
            [90.0 - math.degrees(center.theta)],
            marker="o",
            markersize=2.5,
            color="black",
        )

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_3d(result: SimulationResult, output: Path) -> None:
    """生成三维展示图，包含球体、单元面片与覆盖采样点。"""
    _ensure_parent(output)
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Sphere Coverage 3D View")

    radius = np.linalg.norm(result.tiles[0].center) if result.tiles else 1.0

    # 绘制球体
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#bfbfbf", alpha=1.0, rstride=2, cstride=2, linewidth=0)

    # 绘制采样点
    if result.coverage_points.size:
        covered_points = result.coverage_points[result.coverage_mask]
        uncovered_points = result.coverage_points[~result.coverage_mask]
        if covered_points.size:
            ax.scatter(
                covered_points[:, 0],
                covered_points[:, 1],
                covered_points[:, 2],
                c="#b0b0c0",
                s=4,
                alpha=0.3,
                label="Covered samples",
            )
        if uncovered_points.size:
            ax.scatter(
                uncovered_points[:, 0],
                uncovered_points[:, 1],
                uncovered_points[:, 2],
                c="#d0d0d0",
                s=4,
                alpha=0.3,
                label="Uncovered samples",
            )

    # 绘制各贴片
    for tile in result.tiles:
        vertices = tile.polygon_vertices()
        poly = Poly3DCollection(
            [vertices],
            facecolor="#9e9e9e",
            edgecolor="#101010",
            linewidths=0.7,
            alpha=0.9,
        )
        ax.add_collection3d(poly)

    ax.legend(loc="upper left", fontsize=8)
    ax.set_axis_off()
    fig.savefig(output, dpi=300)
    plt.close(fig)
