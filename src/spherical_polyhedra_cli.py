#!/usr/bin/env python3
"""
硬多面体球形约束 MC 命令行工具。

用法示例：
python -m spherical_polyhedra_cli --shape cube --N 800 --R 12
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
import json

import numpy as np

from .spherical_polyhedra_mc import HardPolyhedraConfinementMC, SimulationConfig, build_polyhedron


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="硬多面体球形约束 MC")
    ap.add_argument("--shape", choices=["cube", "octa", "rd", "trd"], required=True)
    ap.add_argument("--truncation", type=float, default=0.68, help="TRD 截角参数（0-0.9）")
    ap.add_argument("--N", type=int, required=True, help="粒子数")
    ap.add_argument("--R", type=float, required=True, help="初始球半径")
    ap.add_argument("--target-R", type=float, default=None, help="停止压缩的目标半径")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre-sweeps", type=int, default=500)
    ap.add_argument("--relax-sweeps", type=int, default=200)
    ap.add_argument("--max-sweeps", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=5e-4)
    ap.add_argument("--alpha-min", type=float, default=5e-6)
    ap.add_argument("--alpha-up", type=float, default=1.2)
    ap.add_argument("--alpha-down", type=float, default=0.5)
    ap.add_argument("--delta-t", type=float, default=0.05)
    ap.add_argument("--delta-r-deg", type=float, default=5.0)
    ap.add_argument("--max-failed", type=int, default=8)
    ap.add_argument("--out", type=str, default="outputs/spherical_polyhedra_state.npz")
    ap.add_argument("--metrics", type=str, default="outputs/spherical_polyhedra_metrics.json")
    ap.add_argument("--no-metrics", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def progress_printer(quiet: bool):
    if quiet:
        return None

    def _hook(stage: str, step: int, total: int, radius: float) -> None:
        msg = f"[{stage}] {step}/{total} R={radius:.4f}"
        print(msg, end="\r", file=sys.stderr)

    return _hook


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
        print(f"初始化 {args.N} 粒子...", file=sys.stderr)
    sim.random_initialization(args.N)
    if not args.quiet:
        print("开始压缩调度...", file=sys.stderr)
    hook = progress_printer(args.quiet)
    sim.run_density_schedule(cfg, target_radius=args.target_R, progress_hook=hook)
    if hook:
        print("\n压缩完成。", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
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
        print(f"状态保存 → {args.out}", file=sys.stderr)
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
            print(f"指标写入 → {args.metrics}", file=sys.stderr)


if __name__ == "__main__":
    main()
