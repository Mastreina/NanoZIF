"""
CLI handler for Hard Polyhedra Confinement MC.
"""

import argparse
import os
import sys
import json
from dataclasses import asdict
import numpy as np

from ..simulation.hard_polyhedra import HardPolyhedraConfinementMC, SimulationConfig, build_polyhedron

def register_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--shape", choices=["cube", "octa", "rd", "trd"], required=True)
    parser.add_argument("--truncation", type=float, default=0.68, help="TRD truncation (0-0.9)")
    parser.add_argument("--N", type=int, required=True, help="Number of particles")
    parser.add_argument("--R", type=float, required=True, help="Initial sphere radius")
    parser.add_argument("--target-R", type=float, default=None, help="Target radius to stop compression")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-sweeps", type=int, default=500)
    parser.add_argument("--relax-sweeps", type=int, default=200)
    parser.add_argument("--max-sweeps", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--alpha-min", type=float, default=5e-6)
    parser.add_argument("--alpha-up", type=float, default=1.2)
    parser.add_argument("--alpha-down", type=float, default=0.5)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--delta-r-deg", type=float, default=5.0)
    parser.add_argument("--max-failed", type=int, default=8)
    parser.add_argument("--out", type=str, default="outputs/spherical_polyhedra_state.npz")
    parser.add_argument("--metrics", type=str, default="outputs/spherical_polyhedra_metrics.json")
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--quiet", action="store_true")

def progress_printer(quiet: bool):
    if quiet:
        return None

    def _hook(stage: str, step: int, total: int, radius: float) -> None:
        msg = f"[{stage}] {step}/{total} R={radius:.4f}"
        print(msg, end="\r", file=sys.stderr)

    return _hook

def run(args: argparse.Namespace):
    poly = build_polyhedron(args.shape, truncation=args.truncation)
    cfg = SimulationConfig(
        delta_translate=args.delta_t,
        delta_rotate_deg=args.delta_r_deg,
        pre_sweeps=args.pre_sweeps,
        relax_sweeps=args.relax_sweeps,
        max_sweeps=args.max_sweeps,
        compress_alpha=args.alpha,
        min_alpha=args.alpha_min,
        alpha_scale_up=args.alpha_up,
        alpha_scale_down=args.alpha_down,
        max_failed_compressions=args.max_failed,
    )
    rng = np.random.default_rng(args.seed)
    sim = HardPolyhedraConfinementMC(poly, radius=args.R, rng=rng)
    
    if not args.quiet:
        print(f"Initializing {args.N} particles...", file=sys.stderr)
    sim.random_initialization(args.N)
    
    if not args.quiet:
        print("Starting compression schedule...", file=sys.stderr)
    hook = progress_printer(args.quiet)
    sim.run_density_schedule(cfg, target_radius=args.target_R, progress_hook=hook)
    
    if hook:
        print("\nCompression finished.", file=sys.stderr)
        
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    positions = np.stack([p.pos for p in sim.particles], axis=0)
    quats = np.stack([p.quat for p in sim.particles], axis=0)
    np.savez(
        args.out,
        positions=positions,
        quaternions=quats,
        radius=sim.radius,
        config=asdict(cfg),
        shape=args.shape,
    )
    if not args.quiet:
        print(f"State saved → {args.out}", file=sys.stderr)
        
    if not args.no_metrics:
        metrics = {
            "radius": sim.radius,
            "psi4": sim.surface_order_parameter("square"),
            "psi6": sim.surface_order_parameter("hex"),
        }
        r_centers, r_hist = sim.radial_profile(cfg.radial_bins)
        metrics["radial_bins"] = r_centers.tolist()
        metrics["radial_counts"] = r_hist.tolist()
        
        out_dir = os.path.dirname(args.metrics)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.metrics, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        if not args.quiet:
            print(f"Metrics written → {args.metrics}", file=sys.stderr)
