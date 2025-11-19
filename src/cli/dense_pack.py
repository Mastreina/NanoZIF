"""
CLI handler for Two-stage dense packing simulation (HOOMD-blue).
"""
import argparse
import sys
import importlib.util

def register_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--R-sphere", type=float, default=2600.0, help="Target sphere radius (nm)")
    parser.add_argument("--cube-edge", type=float, default=198.0, help="Cube edge length (nm)")
    parser.add_argument("--coverage", type=float, default=0.90, help="Target coverage fraction")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--md-relax", action="store_true", help="Run MD relaxation stage")
    parser.add_argument("--steps-anneal", type=int, default=600_000, help="MD anneal steps")

def run(args: argparse.Namespace):
    # Check for hoomd
    if importlib.util.find_spec("hoomd") is None:
        print("Error: 'hoomd' module not found. This simulation requires HOOMD-blue.", file=sys.stderr)
        sys.exit(1)

    # Import the simulation module dynamically or call its main function
    # Since the original script is a script, we might need to adapt it.
    # Ideally, we should refactor src/simulation/dense_pack.py to have a run() function.
    # For now, we will import it and inject parameters if possible, or just warn.
    
    from ..simulation import dense_pack
    
    dense_pack.run_simulation(
        R_sphere_nm=args.R_sphere,
        cube_edge_nm=args.cube_edge,
        target_coverage=args.coverage,
        seed=args.seed,
        md_relax=args.md_relax,
        steps_anneal=args.steps_anneal
    )
