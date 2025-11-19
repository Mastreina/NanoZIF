#!/usr/bin/env python3
"""
简单的 NPZ 可视化脚本：读取 `spherical_polyhedra_mc` 输出并绘制散点 + 球面。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import plotly.graph_objects as go
except ImportError:  # plotly 可选
    go = None


def render_png(
    positions: np.ndarray,
    radius: float,
    png_path: str,
    sphere_steps: int,
    marker_size: float,
    sphere_alpha: float,
    cmap_name: str,
) -> None:
    dist = np.linalg.norm(positions, axis=1)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap((dist - dist.min()) / max(1e-9, dist.max() - dist.min()))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    steps = max(8, sphere_steps)
    u = np.linspace(0.0, 2.0 * np.pi, steps)
    v = np.linspace(0.0, np.pi, steps // 2)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgrey", linewidth=0.3, alpha=sphere_alpha)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=marker_size, depthshade=False)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()

    out_dir = os.path.dirname(png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def render_html(
    positions: np.ndarray,
    radius: float,
    html_path: str,
    sphere_steps: int,
    marker_size: float,
    sphere_alpha: float,
) -> None:
    if go is None:
        raise RuntimeError("Plotly not installed, cannot export HTML. Please `pip install plotly`.")
    steps = max(8, sphere_steps)
    u = np.linspace(0.0, 2.0 * np.pi, steps)
    v = np.linspace(0.0, np.pi, steps // 2)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=[[0, "#dddddd"], [1, "#dddddd"]],
        showscale=False,
        opacity=float(np.clip(sphere_alpha, 0.0, 1.0)),
    )
    scatter = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=np.linalg.norm(positions, axis=1), colorscale="Viridis"),
    )
    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title="Hard polyhedra in spherical confinement",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    out_dir = os.path.dirname(html_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")

def register_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--state", required=True, help="NPZ file path (containing positions/quaternions/radius)")
    parser.add_argument("--png", required=True, help="Output PNG path")
    parser.add_argument("--html", default=None, help="If provided, export Plotly HTML 3D interactive plot")
    parser.add_argument("--sphere-steps", type=int, default=64, help="Sphere mesh resolution")
    parser.add_argument("--marker-size", type=float, default=8.0)
    parser.add_argument("--sphere-alpha", type=float, default=0.12)
    parser.add_argument("--cmap", type=str, default="viridis", help="Scatter colormap")

def load_state(state_path: str) -> tuple[np.ndarray, float]:
    data = np.load(state_path)
    positions = data["positions"]
    radius = float(data["radius"]) if "radius" in data else float(np.linalg.norm(positions, axis=1).max())
    return positions, radius

def run(args: argparse.Namespace):
    positions, radius = load_state(args.state)
    render_png(
        positions=positions,
        radius=radius,
        png_path=args.png,
        sphere_steps=args.sphere_steps,
        marker_size=args.marker_size,
        sphere_alpha=args.sphere_alpha,
        cmap_name=args.cmap,
    )
    if args.html:
        render_html(
            positions=positions,
            radius=radius,
            html_path=args.html,
            sphere_steps=args.sphere_steps,
            marker_size=args.marker_size * 0.6,
            sphere_alpha=args.sphere_alpha,
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Hard Polyhedra MC NPZ output")
    register_arguments(parser)
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
