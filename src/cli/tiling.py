"""
CLI handler for Regular Polygon Tiling MC.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

from ..simulation.tiling import MonteCarloConfig, SphereSurfaceMonteCarlo, SimulationResult
from ..utils.visualization import plot_3d, plot_equirectangular, build_interactive_html

def register_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--radius", type=float, default=2.55, help="Sphere radius (nm)")
    parser.add_argument("--tile-side", type=float, default=0.6, help="Tile side length (nm)")
    parser.add_argument(
        "--shape",
        type=str,
        default="square",
        choices=["square", "hexagon", "hexagonal_prism", "octagon", "truncated_octahedron", "truncated_cube"],
        help="Tile shape",
    )
    parser.add_argument("--max-steps", type=int, default=200_000, help="Max MC steps")
    parser.add_argument("--stall-steps", type=int, default=20_000, help="Stop if no improvement")
    parser.add_argument("--insert-prob", type=float, default=0.55, help="Insertion probability")
    parser.add_argument("--translation-step", type=float, default=3.0, help="Translation step (deg)")
    parser.add_argument("--rotation-step", type=float, default=5.0, help="Rotation step (deg)")
    parser.add_argument("--coverage-samples", type=int, default=20_000, help="Coverage estimation samples")
    parser.add_argument("--gap-sample", type=int, default=48, help="Gap heuristic samples")
    parser.add_argument("--gap-insert-bias", type=float, default=0.85, help="Gap insertion bias")
    parser.add_argument("--targeted-insert-trigger", type=int, default=4_000)
    parser.add_argument("--targeted-insert-attempts", type=int, default=4)
    parser.add_argument("--targeted-samples", type=int, default=6_000)
    parser.add_argument("--global-relax-trigger", type=int, default=12_000)
    parser.add_argument("--init-layout", type=str, default="none", choices=["none", "fibonacci", "caspar_klug"])
    parser.add_argument("--init-count", type=int, default=0)
    parser.add_argument("--caspar-h", type=int, default=1)
    parser.add_argument("--caspar-k", type=int, default=0)
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--overlap-resolution", type=int, default=7)
    parser.add_argument("--overlap-tolerance", type=float, default=0.0)
    parser.add_argument("--interactive-html", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--energy-temp-start", type=float, default=1.2)
    parser.add_argument("--energy-temp-end", type=float, default=0.05)
    parser.add_argument("--energy-batches", type=int, default=3)
    parser.add_argument("--energy-moves", type=int, default=120)
    parser.add_argument("--energy-softness", type=float, default=1.0)
    parser.add_argument("--force-insert-iterations", type=int, default=300)
    parser.add_argument("--force-insert-step", type=float, default=0.25)
    parser.add_argument("--force-insert-rotation", type=float, default=6.0)
    parser.add_argument("--force-insert-neighbors", type=int, default=24)
    parser.add_argument("--disable-force-insert", action="store_true")
    parser.add_argument("--energy-contact-scale", type=float, default=0.9)
    parser.add_argument("--energy-target-scale", type=float, default=1.1)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)

def create_config(args: argparse.Namespace) -> MonteCarloConfig:
    return MonteCarloConfig(
        radius=args.radius,
        tile_side=args.tile_side,
        insertion_probability=args.insert_prob,
        max_steps=args.max_steps,
        stall_steps=args.stall_steps,
        translation_step=math.radians(args.translation_step),
        rotation_step=math.radians(args.rotation_step),
        overlap_resolution=args.overlap_resolution,
        overlap_tolerance=max(0.0, min(1.0, args.overlap_tolerance)),
        coverage_samples=args.coverage_samples,
        rng_seed=args.rng_seed,
        energy_temperature_start=args.energy_temp_start,
        energy_temperature_end=args.energy_temp_end,
        energy_batches=args.energy_batches,
        energy_moves_per_batch=args.energy_moves,
        energy_acceptance_softness=args.energy_softness,
        energy_contact_scale=args.energy_contact_scale,
        energy_target_scale=args.energy_target_scale,
        force_insert_enabled=not args.disable_force_insert,
        force_insert_iterations=args.force_insert_iterations,
        force_insert_step=args.force_insert_step,
        force_insert_rotation_sigma=math.radians(args.force_insert_rotation),
        force_insert_neighbor_limit=args.force_insert_neighbors,
        checkpoint_interval=args.checkpoint_interval,
        shape=args.shape,
        gap_sample_size=max(1, args.gap_sample),
        gap_insert_bias=max(0.0, min(1.0, args.gap_insert_bias)),
        targeted_insert_trigger=max(0, args.targeted_insert_trigger),
        targeted_insert_attempts=max(1, args.targeted_insert_attempts),
        targeted_samples=max(0, args.targeted_samples),
        global_relax_trigger=max(0, args.global_relax_trigger),
        initial_layout=args.init_layout.lower(),
        initial_count=max(0, args.init_count),
        initial_caspar_h=max(0, args.caspar_h),
        initial_caspar_k=max(0, args.caspar_k),
    )

def save_metadata(result: SimulationResult, config: MonteCarloConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {
        "radius_nm": config.radius,
        "tile_side_nm": config.tile_side,
        "tile_count": len(result.tiles),
        "coverage_fraction": result.coverage,
        "sphere_area_nm2": 4 * math.pi * (config.radius ** 2),
        "estimated_covered_area_nm2": 4 * math.pi * (config.radius ** 2) * result.coverage,
        "max_steps": config.max_steps,
        "stall_steps": config.stall_steps,
        "insertion_probability": config.insertion_probability,
        "translation_step_deg": math.degrees(config.translation_step),
        "rotation_step_deg": math.degrees(config.rotation_step),
        "coverage_samples": config.coverage_samples,
        "rng_seed": config.rng_seed,
        "shape": config.shape,
        "shape_sides": SphereSurfaceMonteCarlo._resolve_sides(config.shape.lower()),
    }
    with (output_dir / "result_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def run(args: argparse.Namespace):
    resume_state = None
    resume_path: Path | None = None
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        resume_state = SphereSurfaceMonteCarlo.load_checkpoint(resume_path)
        setattr(args, "radius", resume_state["radius"])
        setattr(args, "tile_side", resume_state["tile_side"])
        if resume_state.get("shape_name"):
            setattr(args, "shape", resume_state["shape_name"])

    config = create_config(args)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = (output_dir / checkpoint_path).resolve()
    if args.checkpoint_interval > 0 and checkpoint_path is None:
        checkpoint_path = (output_dir / "checkpoint.npz").resolve()

    if resume_state is not None and checkpoint_path is None and resume_path is not None:
        checkpoint_path = resume_path

    solver = SphereSurfaceMonteCarlo(
        config,
        checkpoint_path=checkpoint_path,
        resume_state=resume_state,
    )
    result = solver.run(show_progress=args.progress)

    metadata_path = output_dir / "result_summary.json"
    save_metadata(result, config, output_dir)

    projection_path = output_dir / "projection.png"
    plot_equirectangular(result, projection_path)

    view3d_path = output_dir / "view3d.png"
    plot_3d(result, view3d_path)

    if args.interactive_html is not None:
        interactive_path = (
            args.interactive_html
            if args.interactive_html.is_absolute()
            else output_dir / args.interactive_html
        )
        build_interactive_html(result, interactive_path)
    else:
        interactive_path = None

    print("=== Monte Carlo Sphere Surface Tiling ===")
    print(f"Sphere radius      : {config.radius:.3f} nm")
    print(f"Tile side length   : {config.tile_side:.3f} nm")
    print(f"Tile count         : {len(result.tiles)}")
    print(f"Coverage fraction  : {result.coverage * 100:.2f}%")
    print(f"Estimate area cover: {4 * math.pi * (config.radius ** 2) * result.coverage:.3f} nm^2")
    print(f"Patch shape        : {config.shape}")
    print(f"Outputs saved in   : {output_dir.resolve()}")
    print(f"Metadata JSON      : {metadata_path}")
    print(f"Projection image   : {projection_path}")
    print(f"3D view image      : {view3d_path}")
    if interactive_path:
        print(f"Interactive HTML   : {interactive_path}")
    if checkpoint_path:
        print(f"Checkpoint file    : {checkpoint_path}")
