#!/usr/bin/env python3
"""
单向吸附 MC 的命令行入口。

可视化以 equirectangular 平铺输出 PNG 或 Plotly HTML。
"""

import argparse
import os
import math
import sys
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Polygon

from .progress import ProgressBar
from .unidirectional_adsorption_mc import (
    adsorption_polygons_on_sphere,
    AdsorptionPatch,
    square_vertices,
    regular_polygon_vertices,
    octagon_from_cube_face,
    metrics_from_patches,
    rot2,
)
from .unidirectional_phiA_fast import estimate_adsorption_coverage


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


def build_poly_base(shape: str, args: argparse.Namespace) -> np.ndarray:
    if shape == "square":
        edge = args.edge if (args.edge is not None) else (args.a if args.a is not None else None)
        if edge is None:
            raise SystemExit("shape=square 需要指定 --edge 或 --a（单位 nm）")
        return square_vertices(edge)
    if shape == "octagon_from_cube":
        if args.cube_edge is None:
            raise SystemExit("shape=octagon_from_cube 需要 --cube-edge（立方体原始边长，nm）")
        return octagon_from_cube_face(args.cube_edge)
    if shape == "hexagon":
        if args.hex_side is None and args.hex_Df is None:
            raise SystemExit("shape=hexagon 需要 --hex-side（边长，nm）或 --hex-Df（对边距离，nm）")
        side = args.hex_side if args.hex_side is not None else (args.hex_Df / math.sqrt(3.0))
        return regular_polygon_vertices(6, side)
    raise SystemExit(f"未知形状: {shape}")


def default_color(shape: str) -> str:
    if shape == "square":
        return "#1f77b4"
    if shape == "octagon_from_cube":
        return "#ff7f0e"
    if shape == "hexagon":
        return "#2ca02c"
    return "#444444"


def draw_equirectangular(
    ax,
    patches,
    poly_base,
    R_nm,
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


def get_or_make_patches(shape: str, args: argparse.Namespace):
    poly_base = build_poly_base(shape, args)
    if args.load:
        z = np.load(args.load)
        patches = [
            AdsorptionPatch(c=np.array(c), e1=np.array(e1), e2=np.array(e2), phi=float(phi))
            for c, e1, e2, phi in zip(z["C"], z["E1"], z["E2"], z["PHI"])
        ]
        return patches, poly_base

    progress_bar = None
    progress_hook = None
    if not args.quiet:
        progress_bar = ProgressBar(args.attempts, prefix="adsorption")

        def _hook(attempts: int, accepted: int, streak: int) -> None:
            extra = f"N={accepted} streak={streak}"
            progress_bar.update(attempts, extra=extra)

        progress_hook = _hook

    patches, *_ = adsorption_polygons_on_sphere(
        args.R,
        poly_base,
        seed=args.seed,
        max_attempts=args.attempts,
        stagnation_limit=args.stagnation,
        progress=False,
        relax_enabled=args.relax,
        relax_r_nbr_factor=args.relax_r_nbr_factor,
        relax_step_frac=args.relax_step_frac,
        relax_phi_step_deg=args.relax_phi_step_deg,
        relax_max_iter=args.relax_max_iter,
        relax_max_arc_frac=args.relax_max_arc_frac,
        relax_shuffle=args.relax_shuffle,
        progress_hook=progress_hook,
    )
    if progress_bar:
        progress_bar.finish()
    return patches, poly_base


def render_equirectangular(args: argparse.Namespace) -> None:
    patches, poly_base = get_or_make_patches(args.shape, args)
    color = args.color if args.color else default_color(args.shape)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    setup_axes(ax, title=f"{args.shape} — equirectangular")
    draw_equirectangular(
        ax,
        patches,
        poly_base,
        args.R,
        facecolor=color,
        edgecolor=args.edgecolor,
        alpha=args.alpha,
        lw=args.lw,
        polar_shrink=(not args.no_polar_shrink),
        polar_thresh_deg=args.polar_thresh_deg,
        polar_min_scale=args.polar_min_scale,
    )
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"adsorption_{args.shape}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    if not args.quiet:
        print("Saved →", out_path)


def render_interactive(args: argparse.Namespace) -> None:
    import plotly.graph_objects as go

    patches, poly_base = get_or_make_patches(args.shape, args)
    os.makedirs(args.outdir, exist_ok=True)
    html_path = os.path.join(args.outdir, args.html)

    nu, nv = 96, 48
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = np.outer(np.cos(u), np.sin(v)) * args.R
    y = np.outer(np.sin(u), np.sin(v)) * args.R
    z = np.outer(np.ones_like(u), np.cos(v)) * args.R

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
        opacity=args.sphere_opacity,
        name="sphere",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
    )

    use = patches[: args.max_polys] if (args.max_polys and args.max_polys > 0) else patches
    R_draw = args.R + float(args.lift_nm)

    V = []
    I = []
    J = []
    K = []
    for p in use:
        verts = poly_vertices_on_sphere(p, poly_base, args.R) * R_draw
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
        opacity=args.patch_opacity,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        hoverinfo="skip",
        name="patches",
    )

    traces = [sphere, mesh]
    if args.outline and len(use):
        xs, ys, zs = [], [], []
        for p in use:
            verts = poly_vertices_on_sphere(p, poly_base, args.R) * R_draw
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
                line=dict(width=args.outline_width, color=args.outline_color),
                hoverinfo="skip",
                name="edges",
            )
        )

    if args.show_centers and len(patches):
        C = np.array([p.c for p in patches]) * R_draw
        traces.append(
            go.Scatter3d(
                x=C[:, 0],
                y=C[:, 1],
                z=C[:, 2],
                mode="markers",
                marker=dict(size=args.center_size),
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
    if not args.quiet:
        print(f"Exported → {html_path}")


def add_common_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--shape", choices=["square", "octagon_from_cube", "hexagon"], required=True)
    ap.add_argument("--R", type=float, required=True, help="球半径 nm")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=100000)
    ap.add_argument("--stagnation", type=int, default=20000)
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--relax-r-nbr-factor", type=float, default=2.5)
    ap.add_argument("--relax-step-frac", type=float, default=0.15)
    ap.add_argument("--relax-phi-step-deg", type=float, default=8.0)
    ap.add_argument("--relax-max-iter", type=int, default=20)
    ap.add_argument("--relax-max-arc-frac", type=float, default=0.8)
    ap.add_argument("--relax-shuffle", action="store_true", default=True)
    ap.add_argument("--relax-no-shuffle", dest="relax_shuffle", action="store_false")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="单向吸附 MC 命令行工具")
    sub = ap.add_subparsers(dest="command", required=True)

    viz = sub.add_parser("equirect", help="导出 equirectangular PNG")
    add_common_arguments(viz)
    viz.add_argument("--edge", type=float, default=None)
    viz.add_argument("--a", type=float, default=None)
    viz.add_argument("--cube-edge", type=float, default=None)
    viz.add_argument("--hex-side", type=float, default=None)
    viz.add_argument("--hex-Df", type=float, default=None)
    viz.add_argument("--outdir", type=str, default="outputs")
    viz.add_argument("--color", type=str, default=None)
    viz.add_argument("--edgecolor", type=str, default="none")
    viz.add_argument("--alpha", type=float, default=0.7)
    viz.add_argument("--lw", type=float, default=0.0)
    viz.add_argument("--no-polar-shrink", action="store_true")
    viz.add_argument("--polar-thresh-deg", type=float, default=75.0)
    viz.add_argument("--polar-min-scale", type=float, default=0.12)

    inter = sub.add_parser("interactive", help="导出 Plotly HTML")
    add_common_arguments(inter)
    inter.add_argument("--edge", type=float, default=None)
    inter.add_argument("--a", type=float, default=None)
    inter.add_argument("--cube-edge", type=float, default=None)
    inter.add_argument("--hex-side", type=float, default=None)
    inter.add_argument("--hex-Df", type=float, default=None)
    inter.add_argument("--outdir", type=str, default="outputs")
    inter.add_argument("--html", type=str, default="adsorption_interactive.html")
    inter.add_argument("--max-polys", type=int, default=0)
    inter.add_argument("--sphere-opacity", type=float, default=1.0)
    inter.add_argument("--patch-opacity", type=float, default=1.0)
    inter.add_argument("--lift-nm", type=float, default=5.0)
    inter.add_argument("--center-size", type=float, default=1.0)
    inter.add_argument("--outline", action="store_true", default=True)
    inter.add_argument("--no-outline", dest="outline", action="store_false")
    inter.add_argument("--outline-width", type=float, default=1.0)
    inter.add_argument("--outline-color", type=str, default="#1f77b4")
    inter.add_argument("--show-centers", action="store_true", default=True)
    inter.add_argument("--hide-centers", dest="show_centers", action="store_false")

    cov = sub.add_parser("coverage", help="估计 φ_A 覆盖率")
    add_common_arguments(cov)
    cov.add_argument("--edge", type=float, default=None)
    cov.add_argument("--a", type=float, default=None)
    cov.add_argument("--cube-edge", type=float, default=None)
    cov.add_argument("--hex-side", type=float, default=None)
    cov.add_argument("--hex-Df", type=float, default=None)
    cov.add_argument("--samples", type=int, default=50000)
    cov.add_argument("--seed-cov", type=int, default=1234)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "equirect":
        render_equirectangular(args)
    elif args.command == "interactive":
        render_interactive(args)
    elif args.command == "coverage":
        patches, poly_base = get_or_make_patches(args.shape, args)
        cov_bar = None
        cov_hook = None
        if not args.quiet:
            cov_bar = ProgressBar(args.samples, prefix="coverage")

            def _cov_hook(done: int, hits: int, rate: float) -> None:
                cov_bar.update(done, extra=f"φ≈{rate:.5f}")

            cov_hook = _cov_hook
        phiA, err = estimate_adsorption_coverage(
            patches,
            args.R,
            poly_base,
            M=args.samples,
            seed=args.seed_cov,
            progress=False,
            progress_hook=cov_hook,
        )
        if cov_bar:
            cov_bar.finish()
        N, gamma = metrics_from_patches(patches, args.R)
        print(f"φ_A={phiA:.5f} ± {err:.5f}, N={N}, Gamma={gamma:.6f} μm⁻²")
    else:
        parser.error("未知命令")


if __name__ == "__main__":
    main()
