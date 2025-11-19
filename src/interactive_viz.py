"""基于Plotly生成可交互的三维可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from .monte_carlo_sim import SimulationResult


def _sphere_surface(radius: float, resolution: Tuple[int, int] = (60, 120)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成球面坐标网格，用于绘制底层球体。"""
    theta_res, phi_res = resolution
    theta = np.linspace(1e-6, np.pi - 1e-6, theta_res)
    phi = np.linspace(0.0, 2.0 * np.pi, phi_res)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)
    return x, y, z


def _sphere_wireframe(radius: float, lat_lines: int = 9, lon_lines: int = 18) -> List[go.Scatter3d]:
    traces: List[go.Scatter3d] = []
    latitudes = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, lat_lines)
    longitudes = np.linspace(0.0, 2.0 * np.pi, lon_lines, endpoint=False)

    for lat in latitudes:
        phi = np.linspace(0.0, 2.0 * np.pi, 360)
        x = radius * np.cos(lat) * np.cos(phi)
        y = radius * np.cos(lat) * np.sin(phi)
        z = radius * np.sin(lat) * np.ones_like(phi)
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "#d0d0d0", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for lon in longitudes:
        theta = np.linspace(0.0, np.pi, 360)
        x = radius * np.sin(theta) * np.cos(lon)
        y = radius * np.sin(theta) * np.sin(lon)
        z = radius * np.cos(theta)
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "#d0d0d0", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def _tile_mesh(tile) -> go.Mesh3d:
    vertices, (idx_i, idx_j, idx_k) = tile.polyhedron_mesh()
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=idx_i,
        j=idx_j,
        k=idx_k,
        opacity=0.92,
        color="#9c9c9c",
        name="Patch",
        lighting=dict(ambient=0.25, diffuse=0.65, specular=0.4, roughness=0.35, fresnel=0.15),
        lightposition=dict(x=0.5, y=0.25, z=1.2),
        showscale=False,
    )
    return mesh


def _tile_edges(tile) -> go.Scatter3d:
    """返回多边形边界折线。"""
    vertices = tile.polygon_vertices()
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for idx in range(len(vertices)):
        a = vertices[idx]
        b = vertices[(idx + 1) % len(vertices)]
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#101010", "width": 2.4},
        name="Patch edges",
        hoverinfo="skip",
    )


def build_interactive_html(result: SimulationResult, output_path: Path) -> None:
    """生成交互式HTML文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    radius = np.linalg.norm(result.tiles[0].center) if result.tiles else 1.0

    sphere_x, sphere_y, sphere_z = _sphere_surface(radius)
    fig = go.Figure()
    fig.add_surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        colorscale=[[0.0, "#c8c8c8"], [1.0, "#c8c8c8"]],
        opacity=1.0,
        showscale=False,
        hoverinfo="skip",
        name="Sphere",
    )
    for trace in _sphere_wireframe(radius):
        fig.add_trace(trace)

    # 添加覆盖采样点
    if result.coverage_points.size:
        covered = result.coverage_points[result.coverage_mask]
        uncovered = result.coverage_points[~result.coverage_mask]
        if covered.size:
            fig.add_trace(
                go.Scatter3d(
                    x=covered[:, 0],
                    y=covered[:, 1],
                    z=covered[:, 2],
                    mode="markers",
                    marker={"size": 2.5, "color": "#8c8ca8", "opacity": 0.4},
                    name="Covered samples",
                )
            )
        if uncovered.size:
            fig.add_trace(
                go.Scatter3d(
                    x=uncovered[:, 0],
                    y=uncovered[:, 1],
                    z=uncovered[:, 2],
                    mode="markers",
                    marker={"size": 2.5, "color": "#b3b3b3", "opacity": 0.5},
                    name="Uncovered samples",
                )
            )

    # 添加正方形面片
    for tile in result.tiles:
        fig.add_trace(_tile_mesh(tile))
        fig.add_trace(_tile_edges(tile))

    fig.update_layout(
        title="Interactive Sphere Coverage",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        hovermode="closest",
    )
    fig.write_html(str(output_path), auto_open=False, include_plotlyjs="cdn")
