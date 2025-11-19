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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="可视化球内硬多面体 MC 的 NPZ 输出")
    ap.add_argument("--state", required=True, help="NPZ 文件路径（包含 positions/quaternions/radius）")
    ap.add_argument("--png", required=True, help="输出 PNG 路径")
    ap.add_argument("--html", default=None, help="若提供，则导出 Plotly HTML 三维交互图")
    ap.add_argument("--sphere-steps", type=int, default=64, help="球面网格分辨率")
    ap.add_argument("--marker-size", type=float, default=8.0)
    ap.add_argument("--sphere-alpha", type=float, default=0.12)
    ap.add_argument("--cmap", type=str, default="viridis", help="散点 colormap")
    return ap


def load_state(state_path: str) -> tuple[np.ndarray, float]:
    data = np.load(state_path)
    positions = data["positions"]
    radius = float(data["radius"]) if "radius" in data else float(np.linalg.norm(positions, axis=1).max())
    return positions, radius


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
        raise RuntimeError("Plotly 未安装，无法导出 HTML。请先 `pip install plotly`。")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
