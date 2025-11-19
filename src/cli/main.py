"""
Unified CLI entry point for Monte Carlo simulations.
"""

import argparse
import logging
import sys

# Import sub-command handlers
from . import reversible
from . import hard_poly
from . import tiling
from . import unidirectional
from . import dense_pack
from ..utils import viz_standalone as viz_npz

def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Suite")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Reversible Compression
    cmd_reversible = subparsers.add_parser("reversible", help="Reversible compression of convex bodies")
    reversible.register_arguments(cmd_reversible)
    
    # Hard Polyhedra Confinement
    cmd_hard = subparsers.add_parser("hard-poly", help="Hard polyhedra confinement in sphere")
    hard_poly.register_arguments(cmd_hard)
    
    # Tiling
    cmd_tiling = subparsers.add_parser("tiling", help="Regular polygon tiling on sphere")
    tiling.register_arguments(cmd_tiling)
    
    # Unidirectional Adsorption
    cmd_uni = subparsers.add_parser("unidirectional", help="Unidirectional adsorption of polygons")
    unidirectional.register_arguments(cmd_uni)

    # NPZ Visualization
    cmd_viz = subparsers.add_parser("viz-npz", help="Visualize NPZ output from hard-poly simulation")
    viz_npz.register_arguments(cmd_viz)

    # Dense Packing (HOOMD)
    cmd_dense = subparsers.add_parser("dense-pack", help="Two-stage dense packing (HOOMD-blue)")
    dense_pack.register_arguments(cmd_dense)

    # Parse args
    # Note: argparse handles sub-sub-commands automatically if defined correctly
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Dispatch
    if args.command == "reversible":
        reversible.run(args)
    elif args.command == "hard-poly":
        hard_poly.run(args)
    elif args.command == "tiling":
        tiling.run(args)
    elif args.command == "unidirectional":
        unidirectional.run(args)
    elif args.command == "viz-npz":
        viz_npz.run(args)
    elif args.command == "dense-pack":
        dense_pack.run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
