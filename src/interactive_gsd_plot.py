#!/usr/bin/env python3
"""Generate an interactive Plotly HTML view of the GSD configuration."""

from __future__ import annotations

import argparse
import math
from itertools import product, permutations
from pathlib import Path

import numpy as np
import gsd.hoomd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def sphere_mesh(radius: float, resolution: int = 36):
    theta = np.linspace(0.0, np.pi, resolution)
    phi = np.linspace(0.0, 2.0 * np.pi, resolution * 2)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    x = radius * np.sin(th) * np.cos(ph)
    y = radius * np.sin(th) * np.sin(ph)
    z = radius * np.cos(th)

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    faces_i = []
    faces_j = []
    faces_k = []
    n_theta, n_phi = th.shape
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            idx0 = i * n_phi + j
            idx1 = idx0 + 1
            idx2 = idx0 + n_phi
            idx3 = idx2 + 1
            faces_i.extend([idx0, idx0])
            faces_j.extend([idx1, idx2])
            faces_k.extend([idx3, idx3])
    return verts, faces_i, faces_j, faces_k


def add_wireframe(fig: go.Figure, radius: float, lat_lines: int = 9, lon_lines: int = 18):
    latitudes = np.linspace(-np.pi / 2 + 1e-4, np.pi / 2 - 1e-4, lat_lines)
    longitudes = np.linspace(0.0, 2.0 * np.pi, lon_lines, endpoint=False)

    for lat in latitudes:
        phi = np.linspace(0.0, 2.0 * np.pi, 360)
        x = radius * np.cos(lat) * np.cos(phi)
        y = radius * np.cos(lat) * np.sin(phi)
        z = radius * np.sin(lat) * np.ones_like(phi)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="#bbbbbb", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for lon in longitudes:
        theta = np.linspace(0.0, np.pi, 360)
        x = radius * np.sin(theta) * np.cos(lon)
        y = radius * np.sin(theta) * np.sin(lon)
        z = radius * np.cos(theta)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="#bbbbbb", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_z_to_vec(vec: np.ndarray) -> np.ndarray:
    target = vec / np.linalg.norm(vec)
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(z_axis, target), -1.0, 1.0)
    if np.isclose(dot, 1.0):
        return np.array([1.0, 0.0, 0.0, 0.0])
    if np.isclose(dot, -1.0):
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z_axis, target)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def rhombic_dodeca_vertices(diameter_nm: float) -> np.ndarray:
    half = diameter_nm / 2.0
    diag = half / 2.0
    verts = []
    for sign in (-1.0, 1.0):
        verts.append((sign * half, 0.0, 0.0))
        verts.append((0.0, sign * half, 0.0))
        verts.append((0.0, 0.0, sign * half))
    for sx, sy, sz in product((-1.0, 1.0), repeat=3):
        verts.append((sx * diag, sy * diag, sz * diag))
    return np.asarray(verts, dtype=float)


def truncated_cube_vertices(edge_nm: float, trunc_ratio: float = 3.0) -> np.ndarray:
    base_perms = {perm for perm in permutations((1.0, 1.0, trunc_ratio))}
    coords = set()
    for perm in base_perms:
        for sx, sy, sz in product((-1.0, 1.0), repeat=3):
            coords.add((sx * perm[0], sy * perm[1], sz * perm[2]))
    scale = edge_nm / math.sqrt(8.0)
    return scale * np.asarray(list(coords), dtype=float)


def hex_prism_vertices(flat_diameter_nm: float, height_nm: float | None = None) -> np.ndarray:
    """Regular hexagonal prism. `flat_diameter` is distance between opposite vertices."""
    if height_nm is None:
        height_nm = flat_diameter_nm
    R = flat_diameter_nm / 2.0
    angles = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    xy = np.array([[R * math.cos(a), R * math.sin(a)] for a in angles])
    half_h = 0.5 * height_nm
    verts = []
    for z in (-half_h, half_h):
        for x, y in xy:
            verts.append((x, y, z))
    return np.asarray(verts, dtype=float)


def generate_vertices(
    shape: str,
    size: float,
    hex_height: float | None = None,
    trunc_ratio: float = 3.0,
) -> np.ndarray:
    if shape == "cube":
        half = 0.5 * size
        return np.asarray(
            [(sx, sy, sz) for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)],
            dtype=float,
        )
    if shape == "rhombic_dodeca":
        return rhombic_dodeca_vertices(size)
    if shape == "truncated_cube":
        return truncated_cube_vertices(size, trunc_ratio=trunc_ratio)
    if shape == "hex_prism":
        return hex_prism_vertices(size, height_nm=hex_height)
    raise ValueError(f"Unsupported shape {shape}")


def polyhedron_template(
    shape: str,
    size: float,
    hex_height: float | None = None,
    trunc_ratio: float = 3.0,
):
    verts = generate_vertices(shape, size, hex_height, trunc_ratio)
    hull = ConvexHull(verts)
    equations = hull.equations  # shape (n_faces, 4)
    normals = equations[:, :3]
    offsets = equations[:, 3]
    distances = np.abs(offsets) / np.linalg.norm(normals, axis=1)
    inradius = np.min(distances)
    return verts, hull.simplices, inradius


def poly_mesh(
    centers: np.ndarray,
    orientations: np.ndarray,
    base_vertices: np.ndarray,
    faces: np.ndarray,
    radial_offset: float,
):
    verts = np.zeros((centers.shape[0] * base_vertices.shape[0], 3), dtype=float)
    faces_i = []
    faces_j = []
    faces_k = []

    norms = np.linalg.norm(centers, axis=1)
    normals = centers / norms[:, None]
    shifted_centers = centers + normals * radial_offset

    for idx, (center, quat) in enumerate(zip(shifted_centers, orientations)):
        rot = quat_to_matrix(quat)
        transformed = (rot @ base_vertices.T).T + center
        start = idx * base_vertices.shape[0]
        verts[start:start + base_vertices.shape[0]] = transformed
        for f in faces:
            faces_i.append(start + f[0])
            faces_j.append(start + f[1])
            faces_k.append(start + f[2])

    return verts, faces_i, faces_j, faces_k


def rhombic_dodeca_mesh(centers: np.ndarray, orientations: np.ndarray, diameter: float):
    """Generate mesh for rhombic dodecahedra."""
    verts_base = rhombic_dodeca_vertices(diameter)
    tri_idx = [
        (0, 6, 9), (0, 9, 4), (0, 4, 8), (0, 8, 6),
        (1, 7, 10), (1, 10, 5), (1, 5, 9), (1, 9, 7),
        (2, 8, 11), (2, 11, 6), (2, 6, 10), (2, 10, 8),
        (3, 4, 11), (3, 11, 10), (3, 10, 5), (3, 5, 4),
        (4, 5, 9), (6, 7, 8), (7, 9, 10), (8, 11, 12), (9, 12, 13), (10, 12, 11),
    ]
    verts = np.zeros((centers.shape[0] * len(verts_base), 3), dtype=float)
    faces_i = []
    faces_j = []
    faces_k = []
    norms = np.linalg.norm(centers, axis=1)
    normals = centers / norms[:, None]
    shifted_centers = centers + normals * (diameter / 2.0)
    for idx, (center, quat) in enumerate(zip(shifted_centers, orientations)):
        rot = quat_to_matrix(quat)
        transformed = (rot @ verts_base.T).T + center
        start = idx * len(verts_base)
        verts[start : start + len(verts_base)] = transformed
        for f in tri_idx:
            if all(i < len(verts_base) for i in f):
                faces_i.append(start + f[0])
                faces_j.append(start + f[1])
                faces_k.append(start + f[2])
    return verts, faces_i, faces_j, faces_k


def truncated_cube_mesh(centers: np.ndarray, orientations: np.ndarray, edge: float):
    """Generate mesh for truncated cubes."""
    verts_base = truncated_cube_vertices(edge)
    tri_idx = [
        (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
        (8, 9, 10), (8, 10, 11), (12, 13, 14), (12, 14, 15),
        (16, 17, 18), (16, 18, 19), (20, 21, 22), (20, 22, 23),
        (0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11),
        (0, 12, 4), (1, 13, 5), (2, 14, 6), (3, 15, 7),
        (8, 16, 12), (9, 17, 13), (10, 18, 14), (11, 19, 15),
    ]
    verts = np.zeros((centers.shape[0] * len(verts_base), 3), dtype=float)
    faces_i = []
    faces_j = []
    faces_k = []
    norms = np.linalg.norm(centers, axis=1)
    normals = centers / norms[:, None]
    shifted_centers = centers + normals * (edge / 2.0)
    for idx, (center, quat) in enumerate(zip(shifted_centers, orientations)):
        rot = quat_to_matrix(quat)
        transformed = (rot @ verts_base.T).T + center
        start = idx * len(verts_base)
        verts[start : start + len(verts_base)] = transformed
        for f in tri_idx:
            if all(i < len(verts_base) for i in f):
                faces_i.append(start + f[0])
                faces_j.append(start + f[1])
                faces_k.append(start + f[2])
    return verts, faces_i, faces_j, faces_k


def hex_prism_mesh(centers: np.ndarray, orientations: np.ndarray, diameter: float, height: float | None = None):
    """Generate mesh for regular hexagonal prisms with flat top and bottom faces."""
    if height is None:
        height = diameter
    verts_base = hex_prism_vertices(diameter, height)

    # 6 vertices on bottom, 6 on top
    # Bottom: 0-5, Top: 6-11
    # Bottom hexagon (fan triangulation from center projected)
    tri_idx = [
        # Bottom face (6 triangles)
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5),
        # Top face (6 triangles)
        (6, 7, 8), (6, 8, 9), (6, 9, 10), (6, 10, 11),
        # Side faces (6 quads = 12 triangles)
        (0, 1, 7), (0, 7, 6),  # side 0
        (1, 2, 8), (1, 8, 7),  # side 1
        (2, 3, 9), (2, 9, 8),  # side 2
        (3, 4, 10), (3, 10, 9),  # side 3
        (4, 5, 11), (4, 11, 10),  # side 4
        (5, 0, 6), (5, 6, 11),  # side 5
    ]

    verts = np.zeros((centers.shape[0] * len(verts_base), 3), dtype=float)
    faces_i = []
    faces_j = []
    faces_k = []

    norms = np.linalg.norm(centers, axis=1)
    normals = centers / norms[:, None]
    shifted_centers = centers + normals * (height / 2.0)

    for idx, (center, quat) in enumerate(zip(shifted_centers, orientations)):
        rot = quat_to_matrix(quat)
        transformed = (rot @ verts_base.T).T + center
        start = idx * len(verts_base)
        verts[start : start + len(verts_base)] = transformed

        for f in tri_idx:
            if all(i < len(verts_base) for i in f):
                faces_i.append(start + f[0])
                faces_j.append(start + f[1])
                faces_k.append(start + f[2])

    return verts, faces_i, faces_j, faces_k




def build_figure(
    centers: np.ndarray,
    orientations: np.ndarray,
    radius: float,
    edge: float,
    cube_color: str,
    sphere_color: str,
    sphere_opacity: float,
    background_color: str,
    wireframe: bool,
    camera_config: dict | None,
    lighting_overrides: dict | None,
    lightposition_overrides: dict | None = None,
    shape: str = "cube",
    hex_height: float | None = None,
    trunc_ratio: float = 3.0,
) -> go.Figure:
    fig = go.Figure()
    inner_vertices, inner_i, inner_j, inner_k = sphere_mesh(radius - 1.0)
    outer_vertices, outer_i, outer_j, outer_k = sphere_mesh(radius)
    for verts, fi, fj, fk, name in (
        (inner_vertices, inner_i, inner_j, inner_k, "Sphere inner"),
        (outer_vertices, outer_i, outer_j, outer_k, "Sphere outer"),
    ):
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=fi,
                j=fj,
                k=fk,
                color=sphere_color,
                opacity=sphere_opacity,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                hoverinfo="skip",
                name=name,
            )
        )
    if wireframe:
        add_wireframe(fig, radius)

    # Generate mesh based on shape type
    if shape == "hex_prism":
        height = hex_height if hex_height is not None else edge
        verts, faces_i, faces_j, faces_k = hex_prism_mesh(centers, orientations, edge, height)
    else:
        template_vertices, template_faces, inradius = polyhedron_template(
            shape, edge, hex_height, trunc_ratio
        )
        verts, faces_i, faces_j, faces_k = poly_mesh(
            centers, orientations, template_vertices, template_faces, radial_offset=inradius
        )
    shape_name = dict(
        cube="MOF cubes",
        rhombic_dodeca="Rhombic dodecahedra",
        truncated_cube="Truncated cubes",
    ).get(shape, shape)

    lighting = dict(ambient=0.35, diffuse=0.65, specular=0.25, roughness=0.35)
    if lighting_overrides:
        lighting.update(lighting_overrides)

    mesh_kwargs = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces_i,
        j=faces_j,
        k=faces_k,
        color=cube_color,
        opacity=1.0,
        lighting=lighting,
        name=shape_name,
        showscale=False,
    )

    # 仅在有光源位置设置时才添加
    if lightposition_overrides is not None:
        lightposition = dict(x=1.0, y=1.0, z=2.0)
        lightposition.update(lightposition_overrides)
        mesh_kwargs['lightposition'] = lightposition

    fig.add_trace(go.Mesh3d(**mesh_kwargs))

    fig.update_layout(
        title="MOF Cubes on Sphere (interactive)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
    )
    if camera_config:
        fig.update_layout(scene_camera=camera_config)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default="final.gsd", help="GSD trajectory file")
    parser.add_argument("--frame", type=int, default=-1, help="frame index (default last)")
    parser.add_argument("--output", default="interactive_mof.html", help="output HTML file")
    parser.add_argument("--edge", type=float, default=198.0, help="cube edge length (nm)")
    parser.add_argument(
        "--align-radial",
        action="store_true",
        help="强制将立方体的外侧面法向与球面法向对齐（忽略轨迹中原有姿态）",
    )
    parser.add_argument("--cube-color", default=None, help="立方体颜色 (hex)")
    parser.add_argument("--sphere-color", default=None, help="球面颜色 (hex)")
    parser.add_argument("--sphere-opacity", type=float, default=None, help="球面不透明度 0-1")
    parser.add_argument(
        "--style",
        choices=("default", "em"),
        default="default",
        help="整体风格预设（default / em）。",
    )
    parser.add_argument(
        "--shape",
        choices=("cube", "rhombic_dodeca", "truncated_cube", "hex_prism"),
        default="cube",
        help="多面体形状：cube / rhombic_dodeca / truncated_cube / hex_prism。",
    )
    parser.add_argument(
        "--hex-height",
        type=float,
        default=None,
        help="若 shape=hex_prism，指定棱柱高度（nm），默认等于 --edge。",
    )
    parser.add_argument(
        "--trunc-ratio",
        type=float,
        default=3.0,
        help="截边立方体 (truncated_cube) 的截角比例（越大表示截得越多）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(f"{traj_path} 不存在")

    data = gsd.hoomd.open(str(traj_path))
    snap = data[args.frame]
    typeid = np.asarray(snap.particles.typeid)
    centers = np.asarray(snap.particles.position)[typeid == 0]
    orientations = np.asarray(snap.particles.orientation)[typeid == 0]
    if args.align_radial:
        orientations = np.array([quat_from_z_to_vec(c) for c in centers])
    radius = float(np.mean(np.linalg.norm(centers, axis=1)))

    if args.shape == "rhombic_dodeca":
        default_color = "#8b7d6b"
    elif args.shape == "truncated_cube":
        default_color = "#4a9b8e"
    elif args.shape == "hex_prism":
        default_color = "#b0b0b0"
    else:
        default_color = "#f7a934"
    cube_color = args.cube_color or default_color
    sphere_color = args.sphere_color or "#101018"
    sphere_opacity = args.sphere_opacity if args.sphere_opacity is not None else 0.95
    background = "#ffffff"
    wireframe = True
    camera_config = None
    lighting_overrides = None
    lightposition_overrides = None

    if args.style == "em":
        # 根据形状类型设置默认颜色
        if args.shape == "truncated_cube":
            cube_color = args.cube_color or "#a7d2ae"  # 截角立方体
        elif args.shape == "rhombic_dodeca":
            cube_color = args.cube_color or "#c8dcbb"  # 菱形十二面体
        elif args.shape == "hex_prism":
            cube_color = args.cube_color or "#c8dcbb"  # RGB 200,220,187（六角棱柱）
        else:  # cube
            cube_color = args.cube_color or "#85c1bd"

        sphere_color = args.sphere_color or "#050505"
        sphere_opacity = args.sphere_opacity if args.sphere_opacity is not None else 0.98
        background = "#0d0d0d"
        wireframe = False
        camera_config = dict(eye=dict(x=1.2, y=1.2, z=2.4), up=dict(x=0.0, y=0.0, z=1.0))
        lighting_overrides = dict(ambient=0.80, diffuse=0.30, specular=0.18, roughness=0.82, fresnel=0.12)
        lightposition_overrides = dict(x=1.0, y=1.0, z=2.2)  # 光源位置（高环境光降低方向性）

    fig = build_figure(
        centers,
        orientations,
        radius,
        args.edge,
        cube_color=cube_color,
        sphere_color=sphere_color,
        sphere_opacity=sphere_opacity,
        background_color=background,
        wireframe=wireframe,
        camera_config=camera_config,
        lighting_overrides=lighting_overrides,
        lightposition_overrides=lightposition_overrides,
        shape=args.shape,
        hex_height=args.hex_height,
        trunc_ratio=args.trunc_ratio,
    )
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"生成交互式文件: {args.output}")


if __name__ == "__main__":
    main()
