"""
CLI handler for Reversible Compression MC.
"""

import argparse
import os
from dataclasses import asdict
import numpy as np

from ..simulation.reversible import ReversibleCompressionConfig, ReversibleCompressionMC
from ..geometry.rtt import RTTParams
from ..geometry.polyhedron import ConvexSpheropolyhedron
from ..utils.visualization import render_static_polyhedra, render_interactive_polyhedra

from ..utils.progress import ProgressBar

def build_rtt_particle(cube_length: float, truncation: float, roundness: float) -> ConvexSpheropolyhedron:
    params = RTTParams(cube_length=cube_length, truncation=truncation, roundness=roundness)
    return ConvexSpheropolyhedron.from_vertices(params.vertices, sweep_radius=params.r_sweep)

def register_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", choices=["run", "insert"], default="run")

    parser.add_argument("--n", type=int, default=50, help="Number of particles")
    parser.add_argument("--target-n", type=int, default=None, help="Target N for insert mode")
    parser.add_argument("--R", type=float, default=10.0, help="Sphere radius")
    parser.add_argument("--cube_length", type=float, default=1.0)
    parser.add_argument("--truncation", type=float, default=0.4)
    parser.add_argument("--roundness", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=20000, help="MC steps for run mode")
    parser.add_argument("--translate_sigma", type=float, default=0.05)
    parser.add_argument("--rotate_sigma", type=float, default=0.05)
    parser.add_argument("--shrink_factor", type=float, default=0.999)
    parser.add_argument("--shrink_interval", type=int, default=1000)
    parser.add_argument("--max_shrink_fail", type=int, default=100)
    parser.add_argument("--translate-sigma-final", type=float, default=None)
    parser.add_argument("--rotate-sigma-final", type=float, default=None)
    parser.add_argument("--shrink-factor-final", type=float, default=None)
    parser.add_argument("--anneal-steps", type=int, default=0)

    parser.add_argument("--init-method", choices=["fibonacci", "geodesic"], default="fibonacci")
    parser.add_argument("--init-geodesic-level", type=int, default=2)
    parser.add_argument("--init-jitter", type=float, default=0.0)
    parser.add_argument("--init-relax-steps", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--block-attempts", type=int, default=3)

    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--stagnation", type=int, default=20000)
    parser.add_argument("--relax-steps", type=int, default=200)
    parser.add_argument("--neighbor-deg", type=float, default=25.0)
    parser.add_argument("--insertion-tries", type=int, default=10)
    parser.add_argument("--parallel-proposals", type=int, default=1)
    parser.add_argument("--global-sweep-every", type=int, default=0)
    parser.add_argument("--global-sweep-steps", type=int, default=200)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coverage-samples", type=int, default=20000)
    parser.add_argument("--out", type=str, default="outputs/reversible_mc_state.npz")
    parser.add_argument("--png", type=str, default="outputs/reversible_scene.png")
    parser.add_argument("--html", type=str, default="outputs/reversible_scene.html")
    parser.add_argument("--no-visual", action="store_true")

    parser.add_argument("--align_body_vec", nargs=3, type=float, default=None)
    parser.add_argument("--no_restrict_azimuthal", action="store_true")
    parser.add_argument("--target-neighbor-angle", type=float, default=None)
    parser.add_argument("--order-strength", type=float, default=0.0)
    parser.add_argument("--order-k", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")

def _save_state(sim: ReversibleCompressionMC, out_path: str, meta: dict, quiet: bool) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    positions = np.stack([p.pos for p in sim.particles], axis=0) if sim.particles else np.zeros((0, 3))
    quaternions = np.stack([p.quat for p in sim.particles], axis=0) if sim.particles else np.zeros((0, 4))
    
    np.savez(
        out_path,
        positions=positions,
        quaternions=quaternions,
        R=sim.cfg.radius,
        params=meta,
    )
    if not quiet:
        print(f"Saved → {out_path}")

def run(args: argparse.Namespace):
    if args.mode == "run":
        run_simulation(args)
    else:
        insert_with_relaxation(args)

def run_simulation(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    shape = build_rtt_particle(args.cube_length, args.truncation, args.roundness)
    cfg = ReversibleCompressionConfig(
        radius=args.R,
        translate_sigma=args.translate_sigma,
        rotate_sigma=args.rotate_sigma,
        shrink_factor=args.shrink_factor,
        shrink_interval=args.shrink_interval,
        max_shrink_fail=args.max_shrink_fail,
        align_body_vec=tuple(args.align_body_vec) if args.align_body_vec is not None else None,
        restrict_orientation_azimuthal=(not args.no_restrict_azimuthal),
        init_method=args.init_method,
        init_geodesic_level=args.init_geodesic_level,
        init_jitter=args.init_jitter,
        init_relax_steps=args.init_relax_steps,
        block_relax_size=args.block_size,
        block_relax_attempts=args.block_attempts,
        translate_sigma_final=args.translate_sigma_final,
        rotate_sigma_final=args.rotate_sigma_final,
        shrink_factor_final=args.shrink_factor_final,
        anneal_steps=args.anneal_steps,
        target_neighbor_angle_deg=args.target_neighbor_angle,
        order_strength=args.order_strength,
        order_k_neighbors=args.order_k,
    )
    sim = ReversibleCompressionMC(shape, cfg, rng)
    sim.initialize_on_sphere(
        args.n,
        method=args.init_method,
        jitter=args.init_jitter,
        relax_steps=args.init_relax_steps,
        geodesic_level=args.init_geodesic_level,
    )

    # Progress bar logic omitted for brevity, can be added back if needed
    sim.run(args.steps)

    coverage = sim.estimate_surface_coverage(samples=args.coverage_samples, seed=args.seed)
    if not args.quiet:
        print(
            f"acceptance={sim.acceptance:.3f}, "
            f"surface_density={sim.surface_number_density:.6f}, "
            f"coverage≈{coverage:.3%}, "
            f"R={sim.cfg.radius:.6f}"
        )

    out_path = args.out
    meta = asdict(cfg)
    meta["coverage"] = coverage
    _save_state(sim, out_path, meta=meta, quiet=args.quiet)

def insert_with_relaxation(args: argparse.Namespace):
    target_n = args.target_n if args.target_n is not None else args.n
    max_attempts = args.max_attempts
    stagnation_limit = args.stagnation

    rng = np.random.default_rng(args.seed)
    shape = build_rtt_particle(args.cube_length, args.truncation, args.roundness)
    cfg = ReversibleCompressionConfig(
        radius=args.R,
        translate_sigma=args.translate_sigma,
        rotate_sigma=args.rotate_sigma,
        shrink_factor=args.shrink_factor,
        shrink_interval=args.shrink_interval,
        max_shrink_fail=args.max_shrink_fail,
        align_body_vec=tuple(args.align_body_vec) if args.align_body_vec is not None else None,
        restrict_orientation_azimuthal=(not args.no_restrict_azimuthal),
        init_method=args.init_method,
        init_geodesic_level=args.init_geodesic_level,
        init_jitter=args.init_jitter,
        init_relax_steps=args.init_relax_steps,
        block_relax_size=args.block_size,
        block_relax_attempts=args.block_attempts,
        translate_sigma_final=args.translate_sigma_final,
        rotate_sigma_final=args.rotate_sigma_final,
        shrink_factor_final=args.shrink_factor_final,
        anneal_steps=args.anneal_steps,
        target_neighbor_angle_deg=args.target_neighbor_angle,
        order_strength=args.order_strength,
        order_k_neighbors=args.order_k,
    )
    sim = ReversibleCompressionMC(shape, cfg, rng)

    attempts = 0
    streak = 0
    inserted_since_sweep = 0
    
    while len(sim.particles) < target_n and attempts < max_attempts and streak < stagnation_limit:
        attempts += 1
        success = sim.try_insert_random_particle(
            insertion_attempts=args.insertion_tries,
            relax_steps=args.relax_steps,
            neighbor_angle_deg=args.neighbor_deg,
            parallel_proposals=args.parallel_proposals,
        )
        if success:
            streak = 0
            inserted_since_sweep += 1
            if args.global_sweep_every > 0 and inserted_since_sweep >= args.global_sweep_every:
                sim.global_relax(
                    args.global_sweep_steps,
                    neighbor_angle_deg=args.neighbor_deg,
                    block_size=args.block_size,
                )
                inserted_since_sweep = 0
        else:
            streak += 1
            
        if not args.quiet and attempts % 1000 == 0:
             print(f"Insert progress: N={len(sim.particles)}/{target_n}, attempts={attempts}, streak={streak}")

    if not args.quiet:
        print(f"inserted={len(sim.particles)} | attempts={attempts} | streak={streak} | target={target_n}")

    out_path = args.out
    if args.out == "outputs/reversible_mc_state.npz":
        out_path = "outputs/reversible_insert_state.npz"
    coverage = sim.estimate_surface_coverage(samples=args.coverage_samples, seed=args.seed)

    meta = dict(
        mode="insert",
        target_n=target_n,
        attempts=attempts,
        streak=streak,
        config=asdict(cfg),
        coverage=coverage,
    )
    _save_state(sim, out_path, meta=meta, quiet=args.quiet)

    if not args.quiet:
        print(f"coverage≈{coverage:.3%}")
    if not args.no_visual:
        render_static_polyhedra(sim.particles, sim.shape, sim.cfg.radius, args.png, title="Reversible Insert")
        render_interactive_polyhedra(sim.particles, sim.shape, sim.cfg.radius, args.html, title="Reversible Insert")
        if not args.quiet:
            print(f"Rendered PNG → {args.png}")
            print(f"Rendered HTML → {args.html}")
