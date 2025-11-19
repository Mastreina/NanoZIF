"""
CLI handler for Unidirectional Adsorption MC.
"""

import argparse
import math
import os
import sys
import numpy as np

from ..simulation.unidirectional import (
    adsorption_polygons_on_sphere,
    estimate_adsorption_coverage,
    metrics_from_patches,
    square_vertices,
    regular_polygon_vertices,
    octagon_from_cube_face,
    AdsorptionPatch,
)
from ..utils.viz_unidirectional import render_equirectangular_plot, render_interactive_plot

from ..utils.progress import ProgressBar

def build_poly_base(shape: str, args: argparse.Namespace) -> np.ndarray:
    if shape == "square":
        edge = args.edge if (args.edge is not None) else (args.a if args.a is not None else None)
        if edge is None:
            raise SystemExit("shape=square requires --edge or --a (nm)")
        return square_vertices(edge)
    if shape == "octagon_from_cube":
        if args.cube_edge is None:
            raise SystemExit("shape=octagon_from_cube requires --cube-edge (nm)")
        return octagon_from_cube_face(args.cube_edge)
    if shape == "hexagon":
        if args.hex_side is None and args.hex_Df is None:
            raise SystemExit("shape=hexagon requires --hex-side or --hex-Df (nm)")
        side = args.hex_side if args.hex_side is not None else (args.hex_Df / math.sqrt(3.0))
        return regular_polygon_vertices(6, side)
    raise SystemExit(f"Unknown shape: {shape}")

def default_color(shape: str) -> str:
    if shape == "square":
        return "#1f77b4"
    if shape == "octagon_from_cube":
        return "#ff7f0e"
    if shape == "hexagon":
        return "#2ca02c"
    return "#444444"

def get_or_make_patches(shape: str, args: argparse.Namespace):
    poly_base = build_poly_base(shape, args)
    if args.load:
        z = np.load(args.load)
        patches = [
            AdsorptionPatch(c=np.array(c), e1=np.array(e1), e2=np.array(e2), phi=float(phi))
            for c, e1, e2, phi in zip(z["C"], z["E1"], z["E2"], z["PHI"])
        ]
        return patches, poly_base

    progress_hook = None
    if not args.quiet:
        # Simple print hook instead of full progress bar for now
        def _hook(attempts: int, accepted: int, streak: int) -> None:
            if attempts % 5000 == 0:
                print(f"Adsorption: attempts={attempts}, N={accepted}, streak={streak}", end="\r")
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
    if not args.quiet:
        print() # Newline
    return patches, poly_base

def register_arguments(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # Common arguments function
    def add_common(ap):
        ap.add_argument("--shape", choices=["square", "octagon_from_cube", "hexagon"], required=True)
        ap.add_argument("--R", type=float, required=True, help="Sphere radius nm")
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
        ap.add_argument("--edge", type=float, default=None)
        ap.add_argument("--a", type=float, default=None)
        ap.add_argument("--cube-edge", type=float, default=None)
        ap.add_argument("--hex-side", type=float, default=None)
        ap.add_argument("--hex-Df", type=float, default=None)

    viz = sub.add_parser("equirect", help="Export equirectangular PNG")
    add_common(viz)
    viz.add_argument("--outdir", type=str, default="outputs")
    viz.add_argument("--color", type=str, default=None)
    viz.add_argument("--edgecolor", type=str, default="none")
    viz.add_argument("--alpha", type=float, default=0.7)
    viz.add_argument("--lw", type=float, default=0.0)
    viz.add_argument("--no-polar-shrink", action="store_true")
    viz.add_argument("--polar-thresh-deg", type=float, default=75.0)
    viz.add_argument("--polar-min-scale", type=float, default=0.12)

    inter = sub.add_parser("interactive", help="Export Plotly HTML")
    add_common(inter)
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

    cov = sub.add_parser("coverage", help="Estimate coverage phi_A")
    add_common(cov)
    cov.add_argument("--samples", type=int, default=50000)
    cov.add_argument("--seed-cov", type=int, default=1234)

def run(args: argparse.Namespace):
    if args.subcommand == "equirect":
        patches, poly_base = get_or_make_patches(args.shape, args)
        color = args.color if args.color else default_color(args.shape)
        out_path = os.path.join(args.outdir, f"adsorption_{args.shape}.png")
        render_equirectangular_plot(
            patches, poly_base, args.R, args.shape, out_path,
            color=color, edgecolor=args.edgecolor, alpha=args.alpha, lw=args.lw,
            polar_shrink=(not args.no_polar_shrink),
            polar_thresh_deg=args.polar_thresh_deg,
            polar_min_scale=args.polar_min_scale,
            quiet=args.quiet
        )
    elif args.subcommand == "interactive":
        patches, poly_base = get_or_make_patches(args.shape, args)
        html_path = os.path.join(args.outdir, args.html)
        render_interactive_plot(
            patches, poly_base, args.R, html_path,
            max_polys=args.max_polys,
            sphere_opacity=args.sphere_opacity,
            patch_opacity=args.patch_opacity,
            lift_nm=args.lift_nm,
            center_size=args.center_size,
            outline=args.outline,
            outline_width=args.outline_width,
            outline_color=args.outline_color,
            show_centers=args.show_centers,
            quiet=args.quiet
        )
    elif args.subcommand == "coverage":
        patches, poly_base = get_or_make_patches(args.shape, args)
        
        def _cov_hook(done: int, hits: int, rate: float) -> None:
             if done % 5000 == 0:
                 print(f"Coverage: {done}/{args.samples} phi={rate:.5f}", end="\r")

        phiA, err = estimate_adsorption_coverage(
            patches,
            args.R,
            poly_base,
            M=args.samples,
            seed=args.seed_cov,
            progress=False,
            progress_hook=_cov_hook if not args.quiet else None,
        )
        if not args.quiet:
            print()
        N, gamma = metrics_from_patches(patches, args.R)
        print(f"φ_A={phiA:.5f} ± {err:.5f}, N={N}, Gamma={gamma:.6f} μm⁻²")
    else:
        print("Unknown subcommand")
