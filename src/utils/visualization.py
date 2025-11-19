"""
Visualization utilities for Monte Carlo simulations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Any, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import plotly.graph_objects as go
    from plotly import colors
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..geometry.core import SpherePoint

if TYPE_CHECKING:
    from ..simulation.tiling import SimulationResult

# -----------------------------------------------------------------------------
# Generic Polyhedra Visualization (Reversible / Hard Poly)
# -----------------------------------------------------------------------------

def collect_positions(particles: List[Any]) -> np.ndarray:
    if particles:
        return np.stack([p.pos for p in particles], axis=0)
    return np.zeros((0, 3), dtype=float)

def render_static_polyhedra(
    particles: List[Any],
    shape: Any,  # ConvexSpheropolyhedron or similar
    radius: float,
    png_path: str,
    title: str = "Simulation State"
) -> None:
    if not particles:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere wireframe
    u = np.linspace(0.0, 2.0 * math.pi, 50)
    v = np.linspace(0.0, math.pi, 25)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgrey", linewidth=0.3, alpha=0.5)

    # Draw particles
    faces = shape.faces
    cmap = plt.get_cmap("tab20")
    n_colors = 20
    
    for idx, particle in enumerate(particles):
        verts = shape.world_vertices(particle.quat, particle.pos)
        poly = Poly3DCollection(verts[faces], alpha=0.55, linewidths=0.2)
        color = cmap(idx % n_colors)
        poly.set_facecolor(color)
        poly.set_edgecolor("k")
        ax.add_collection3d(poly)

    centers = collect_positions(particles)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="k", s=12)

    max_range = radius * 1.2
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-max_range, max_range)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

def render_interactive_polyhedra(
    particles: List[Any],
    shape: Any,
    radius: float,
    html_path: str,
    title: str = "Simulation State"
) -> None:
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed, skipping interactive render.")
        return
    if not particles:
        return

    traces = []
    # Sphere surface
    u = np.linspace(0.0, 2.0 * math.pi, 60)
    v = np.linspace(0.0, math.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    sphere = go.Surface(
        x=x, y=y, z=z,
        opacity=0.3,
        showscale=False,
        name="sphere",
        surfacecolor=np.zeros_like(z),
        colorscale=[[0, "#87CEFA"], [1, "#87CEFA"]],
    )
    traces.append(sphere)

    palette = colors.qualitative.Plotly
    faces = shape.faces
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    for idx, particle in enumerate(particles):
        verts = shape.world_vertices(particle.quat, particle.pos)
        colour = palette[idx % len(palette)]
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i, j=j, k=k,
                opacity=0.6,
                color=colour,
                name=f"particle {idx}",
            )
        )

    centers = collect_positions(particles)
    traces.append(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            marker=dict(size=4, color="black"),
            name="centers",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(html_path, include_plotlyjs="cdn")

# -----------------------------------------------------------------------------
# Tiling Visualization
# -----------------------------------------------------------------------------

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _convert_vertices_to_lon_lat(vertices: np.ndarray) -> np.ndarray:
    """Convert sphere coordinates to (longitude, latitude) in degrees."""
    vertices_norm = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    theta = np.arccos(np.clip(vertices_norm[:, 2], -1.0, 1.0))
    phi = np.arctan2(vertices_norm[:, 1], vertices_norm[:, 0])
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)
    return np.stack((lon, lat), axis=1)

def plot_equirectangular(result: "SimulationResult", output: Path) -> None:
    """Generate 2D equirectangular projection plot."""
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

def plot_3d(result: "SimulationResult", output: Path) -> None:
    """Generate 3D plot with sphere, tiles, and coverage samples."""
    _ensure_parent(output)
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Sphere Coverage 3D View")

    radius = np.linalg.norm(result.tiles[0].center) if result.tiles else 1.0

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#bfbfbf", alpha=1.0, rstride=2, cstride=2, linewidth=0)

    # Draw samples
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

    # Draw tiles
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

def _sphere_surface(radius: float, resolution: Tuple[int, int] = (60, 120)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                x=x, y=y, z=z,
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
                x=x, y=y, z=z,
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
        i=idx_i, j=idx_j, k=idx_k,
        opacity=0.92,
        color="#9c9c9c",
        name="Patch",
        lighting=dict(ambient=0.25, diffuse=0.65, specular=0.4, roughness=0.35, fresnel=0.15),
        lightposition=dict(x=0.5, y=0.25, z=1.2),
        showscale=False,
    )
    return mesh

def _tile_edges(tile) -> go.Scatter3d:
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
        x=xs, y=ys, z=zs,
        mode="lines",
        line={"color": "#101010", "width": 2.4},
        name="Patch edges",
        hoverinfo="skip",
    )

def build_interactive_html(result: "SimulationResult", output_path: Path) -> None:
    """Generate interactive HTML for tiling result."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed, skipping interactive render.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    radius = np.linalg.norm(result.tiles[0].center) if result.tiles else 1.0

    sphere_x, sphere_y, sphere_z = _sphere_surface(radius)
    fig = go.Figure()
    fig.add_surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        colorscale=[[0.0, "#c8c8c8"], [1.0, "#c8c8c8"]],
        opacity=1.0,
        showscale=False,
        hoverinfo="skip",
        name="Sphere",
    )
    for trace in _sphere_wireframe(radius):
        fig.add_trace(trace)

    # Add coverage samples
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

    # Add tiles
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
