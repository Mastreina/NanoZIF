"""
Visualization utilities for Unidirectional Adsorption MC.
"""

import math
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Polygon

from ..simulation.unidirectional import AdsorptionPatch, rot2

def exp_map(c: np.ndarray, w: np.ndarray, R: float, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    wx, wy = float(w[0]), float(w[1])
    r = math.hypot(wx, wy)
    if r < 1e-12:
        return c.copy()
    alpha = r / R
    t = (wx * e1 + wy * e2)
    t = t / np.linalg.norm(t)
    return math.cos(alpha) * c + math.sin(alpha) * t

def poly_vertices_on_sphere(patch: AdsorptionPatch, poly_base: np.ndarray, R_nm: float) -> np.ndarray:
    V2 = poly_base @ rot2(patch.phi).T
    return np.array([exp_map(patch.c, v, R_nm, patch.e1, patch.e2) for v in V2])

def xyz_to_lonlat_deg(P3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = P3[:, 0], P3[:, 1], P3[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat

def unwrap_continuous_lons(lons_deg: np.ndarray) -> np.ndarray:
    lons = np.asarray(lons_deg, float)
    out = [lons[0]]
    for v in lons[1:]:
        cand = np.array([v - 360.0, v, v + 360.0])
        prev = out[-1]
        out.append(cand[np.argmin(np.abs(cand - prev))])
    return np.array(out, float)

def split_polygon_at_dateline(lon_cont: np.ndarray, lat: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    lon_cont = np.asarray(lon_cont, dtype=float)
    lat = np.asarray(lat, dtype=float)

    num_vertices = len(lon_cont)
    if num_vertices < 3:
        return []

    parts_lon: List[np.ndarray] = []
    parts_lat: List[np.ndarray] = []
    cur_lon = [lon_cont[0]]
    cur_lat = [lat[0]]

    def emit_current_segment() -> None:
        if len(cur_lon) >= 3:
            parts_lon.append(np.asarray(cur_lon, dtype=float).copy())
            parts_lat.append(np.asarray(cur_lat, dtype=float).copy())

    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        x1, y1 = float(lon_cont[i]), float(lat[i])
        x2, y2 = float(lon_cont[j]), float(lat[j])
        dx = x2 - x1
        if dx == 0.0:
            cur_lon.append(x2)
            cur_lat.append(y2)
            continue
        lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
        bounds = []
        k_min = int(math.floor((lo - 180.0) / 360.0))
        k_max = int(math.ceil((hi + 180.0) / 360.0))
        for k in range(k_min, k_max + 1):
            for b in (-180.0 + 360.0 * k, 180.0 + 360.0 * k):
                if lo < b < hi:
                    bounds.append(b)
        bounds.sort()
        if not bounds:
            cur_lon.append(x2)
            cur_lat.append(y2)
            continue
        x_prev, y_prev = x1, y1
        for b in bounds:
            t = (b - x_prev) / (x2 - x_prev)
            y_cross = y_prev + t * (y2 - y_prev)
            cur_lon.append(b)
            cur_lat.append(y_cross)
            emit_current_segment()
            cur_lon = [b]
            cur_lat = [y_cross]
            x_prev, y_prev = b, y_cross
        cur_lon.append(x2)
        cur_lat.append(y2)

    emit_current_segment()
    return list(zip(parts_lon, parts_lat))

def draw_equirectangular(
    ax,
    patches: List[AdsorptionPatch],
    poly_base: np.ndarray,
    R_nm: float,
    facecolor="#1f77b4",
    edgecolor="none",
    alpha=0.7,
    lw=0.0,
    polar_shrink=True,
    polar_thresh_deg=75.0,
    polar_min_scale=0.12,
) -> None:
    for p in patches:
        verts3 = poly_vertices_on_sphere(p, poly_base, R_nm)
        lon0, lat0 = xyz_to_lonlat_deg(verts3)
        lon_cont = unwrap_continuous_lons(lon0)
        for lon_seg, lat_seg in split_polygon_at_dateline(lon_cont, lat0):
            xy = np.column_stack([lon_seg, lat_seg])
            poly = Polygon(
                xy,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
                joinstyle="round",
            )
            if polar_shrink:
                mean_lat = float(np.mean(lat_seg))
                width = float(np.max(lon_seg) - np.min(lon_seg)) if len(lon_seg) else 0.0
                if abs(mean_lat) >= polar_thresh_deg or width > 150.0:
                    cx = float(np.mean(lon_seg))
                    sx = max(polar_min_scale, math.cos(math.radians(abs(mean_lat))))
                    tr = transforms.Affine2D().translate(-cx, 0.0).scale(sx, 1.0).translate(cx, 0.0) + ax.transData
                    poly.set_transform(tr)
            ax.add_patch(poly)

def setup_axes(ax, title: str = "") -> None:
    ax.set_xlim(-200, 200)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

def render_equirectangular_plot(
    patches: List[AdsorptionPatch],
    poly_base: np.ndarray,
    R_nm: float,
    shape_name: str,
    out_path: str,
    color: str = "#1f77b4",
    edgecolor: str = "none",
    alpha: float = 0.7,
    lw: float = 0.0,
    polar_shrink: bool = True,
    polar_thresh_deg: float = 75.0,
    polar_min_scale: float = 0.12,
    quiet: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    setup_axes(ax, title=f"{shape_name} — equirectangular")
    draw_equirectangular(
        ax,
        patches,
        poly_base,
        R_nm,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
        lw=lw,
        polar_shrink=polar_shrink,
        polar_thresh_deg=polar_thresh_deg,
        polar_min_scale=polar_min_scale,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    if not quiet:
        print("Saved →", out_path)

def render_interactive_plot(
    patches: List[AdsorptionPatch],
    poly_base: np.ndarray,
    R_nm: float,
    html_path: str,
    max_polys: int = 0,
    sphere_opacity: float = 1.0,
    patch_opacity: float = 1.0,
    lift_nm: float = 5.0,
    center_size: float = 1.0,
    outline: bool = True,
    outline_width: float = 1.0,
    outline_color: str = "#1f77b4",
    show_centers: bool = True,
    quiet: bool = False
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed.")
        return

    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    nu, nv = 96, 48
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = np.outer(np.cos(u), np.sin(v)) * R_nm
    y = np.outer(np.sin(u), np.sin(v)) * R_nm
    z = np.outer(np.ones_like(u), np.cos(v)) * R_nm

    surfacecolor = np.zeros_like(z)
    sky_blue = "#87CEFA"
    sphere = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surfacecolor,
        colorscale=[[0, sky_blue], [1, sky_blue]],
        cmin=0,
        cmax=1,
        showscale=False,
        opacity=sphere_opacity,
        name="sphere",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
    )

    use = patches[: max_polys] if (max_polys and max_polys > 0) else patches
    R_draw = R_nm + float(lift_nm)

    V = []
    I = []
    J = []
    K = []
    for p in use:
        verts = poly_vertices_on_sphere(p, poly_base, R_nm) * R_draw
        m = len(verts)
        if m < 3:
            continue
        base = len(V)
        center = (p.c * R_draw).tolist()
        V.append(center)
        for q in verts:
            V.append(q.tolist())
        for k in range(1, m):
            I.append(base + 0)
            J.append(base + k)
            K.append(base + k + 1)
        I.append(base + 0)
        J.append(base + m)
        K.append(base + 1)
    V = np.array(V) if V else np.zeros((0, 3))

    mesh = go.Mesh3d(
        x=V[:, 0],
        y=V[:, 1],
        z=V[:, 2],
        i=I,
        j=J,
        k=K,
        opacity=patch_opacity,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        hoverinfo="skip",
        name="patches",
    )

    traces = [sphere, mesh]
    if outline and len(use):
        xs, ys, zs = [], [], []
        for p in use:
            verts = poly_vertices_on_sphere(p, poly_base, R_nm) * R_draw
            verts = np.vstack([verts, verts[0:1]])
            xs += verts[:, 0].tolist() + [None]
            ys += verts[:, 1].tolist() + [None]
            zs += verts[:, 2].tolist() + [None]
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=outline_width, color=outline_color),
                hoverinfo="skip",
                name="edges",
            )
        )

    if show_centers and len(patches):
        C = np.array([p.c for p in patches]) * R_draw
        traces.append(
            go.Scatter3d(
                x=C[:, 0],
                y=C[:, 1],
                z=C[:, 2],
                mode="markers",
                marker=dict(size=center_size),
                name="centers",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Adsorption polygons on sphere — interactive",
        scene=dict(
            aspectmode="data",
            xaxis_title="x (nm)",
            yaxis_title="y (nm)",
            zaxis_title="z (nm)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    if not quiet:
        print(f"Exported → {html_path}")
