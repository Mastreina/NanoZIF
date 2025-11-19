#!/usr/bin/env python3
"""
可逆压缩 MC 命令行入口。

提供两种模式：
- run   : 传统可逆压缩 MC（批量初始化 + 收缩）
- insert: 顺序插入 + 局部可逆松弛，并输出统一可视化
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
import numpy as np

from .progress import ProgressBar
from .reversible_compression_mc import ReversibleCompressionConfig, ReversibleCompressionMC, build_rtt_particle


def _collect_positions(particles):
    if particles:
        return np.stack([p.pos for p in particles], axis=0)
    return np.zeros((0, 3), dtype=float)


def _collect_quats(particles):
    if particles:
        return np.stack([p.quat for p in particles], axis=0)
    return np.zeros((0, 4), dtype=float)


def _save_state(sim: ReversibleCompressionMC, out_path: str, meta: dict, quiet: bool) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        out_path,
        positions=_collect_positions(sim.particles),
        quaternions=_collect_quats(sim.particles),
        R=sim.cfg.radius,
        params=meta,
    )
    if not quiet:
        print(f"Saved → {out_path}")


def _render_static(sim: ReversibleCompressionMC, png_path: str) -> None:
    if not sim.particles:
        return
    import math
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0.0, 2.0 * math.pi, 50)
    v = np.linspace(0.0, math.pi, 25)
    x = sim.cfg.radius * np.outer(np.cos(u), np.sin(v))
    y = sim.cfg.radius * np.outer(np.sin(u), np.sin(v))
    z = sim.cfg.radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgrey", linewidth=0.3, alpha=0.5)

    faces = sim.shape.faces
    cmap = plt.cm.get_cmap("tab20", max(20, len(sim.particles)))
    for idx, particle in enumerate(sim.particles):
        verts = sim.shape.world_vertices(particle.quat, particle.pos)
        poly = Poly3DCollection(verts[faces], alpha=0.55, linewidths=0.2)
        color = cmap(idx % cmap.N)
        poly.set_facecolor(color)
        poly.set_edgecolor("k")
        ax.add_collection3d(poly)

    centers = _collect_positions(sim.particles)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color="k", s=12)

    max_range = sim.cfg.radius * 1.2
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-max_range, max_range)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Reversible compression – sequential insertion")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def _render_interactive(sim: ReversibleCompressionMC, html_path: str) -> None:
    if not sim.particles:
        return
    import math
    import plotly.graph_objects as go
    from plotly import colors

    traces = []
    u = np.linspace(0.0, 2.0 * math.pi, 60)
    v = np.linspace(0.0, math.pi, 30)
    x = sim.cfg.radius * np.outer(np.cos(u), np.sin(v))
    y = sim.cfg.radius * np.outer(np.sin(u), np.sin(v))
    z = sim.cfg.radius * np.outer(np.ones_like(u), np.cos(v))
    sphere = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.3,
        showscale=False,
        name="sphere",
        surfacecolor=np.zeros_like(z),
        colorscale=[[0, "#87CEFA"], [1, "#87CEFA"]],
    )
    traces.append(sphere)

    palette = colors.qualitative.Plotly
    faces = sim.shape.faces
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    for idx, particle in enumerate(sim.particles):
        verts = sim.shape.world_vertices(particle.quat, particle.pos)
        colour = palette[idx % len(palette)]
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i,
                j=j,
                k=k,
                opacity=0.6,
                color=colour,
                name=f"particle {idx}",
            )
        )

    centers = _collect_positions(sim.particles)
    traces.append(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            marker=dict(size=4, color="black"),
            name="centers",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Reversible compression – sequential insertion",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(html_path, include_plotlyjs="cdn")


def run_simulation(args: argparse.Namespace) -> None:
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

    progress_bar = None
    progress_hook = None
    if not args.quiet:
        progress_bar = ProgressBar(args.steps, prefix="run")

        def _progress(step: int, total: int) -> None:
            progress_bar.update(step)

        progress_hook = _progress

    sim.run(args.steps, progress_callback=progress_hook)
    if progress_bar:
        progress_bar.finish()

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


def insert_with_relaxation(args: argparse.Namespace) -> None:
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
    progress_bar = ProgressBar(max_attempts, prefix="insert") if not args.quiet else None

    inserted_since_sweep = 0
    while (
        len(sim.particles) < target_n
        and attempts < max_attempts
        and streak < stagnation_limit
    ):
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
            if (
                args.global_sweep_every > 0
                and inserted_since_sweep >= args.global_sweep_every
            ):
                sim.global_relax(
                    args.global_sweep_steps,
                    neighbor_angle_deg=args.neighbor_deg,
                    block_size=args.block_size,
                )
                inserted_since_sweep = 0
        else:
            streak += 1
        if progress_bar:
            progress_bar.update(attempts, extra=f"N={len(sim.particles)} streak={streak}")

    if progress_bar:
        progress_bar.finish()

    if not args.quiet:
        print(
            f"inserted={len(sim.particles)} | attempts={attempts} | streak={streak} "
            f"| target={target_n}"
        )

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
        _render_static(sim, args.png)
        _render_interactive(sim, args.html)
        if not args.quiet:
            print(f"Rendered PNG → {args.png}")
            print(f"Rendered HTML → {args.html}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="可逆压缩 MC 命令行工具")
    ap.add_argument("--mode", choices=["run", "insert"], default="run")

    ap.add_argument("--n", type=int, default=50, help="run 模式粒子数 / insert 默认目标粒子数")
    ap.add_argument("--target-n", type=int, default=None, help="insert 模式目标粒子数（缺省取 --n）")
    ap.add_argument("--R", type=float, default=10.0, help="初始球半径")
    ap.add_argument("--cube_length", type=float, default=1.0)
    ap.add_argument("--truncation", type=float, default=0.4)
    ap.add_argument("--roundness", type=float, default=0.0)

    ap.add_argument("--steps", type=int, default=20000, help="run 模式 Monte Carlo 步数")
    ap.add_argument("--translate_sigma", type=float, default=0.05)
    ap.add_argument("--rotate_sigma", type=float, default=0.05)
    ap.add_argument("--shrink_factor", type=float, default=0.999)
    ap.add_argument("--shrink_interval", type=int, default=1000)
    ap.add_argument("--max_shrink_fail", type=int, default=100)
    ap.add_argument("--translate-sigma-final", type=float, default=None, help="退火后 translate_sigma")
    ap.add_argument("--rotate-sigma-final", type=float, default=None, help="退火后 rotate_sigma")
    ap.add_argument("--shrink-factor-final", type=float, default=None, help="退火后 shrink_factor")
    ap.add_argument("--anneal-steps", type=int, default=0, help="退火持续步数")

    ap.add_argument("--init-method", choices=["fibonacci", "geodesic"], default="fibonacci")
    ap.add_argument("--init-geodesic-level", type=int, default=2)
    ap.add_argument("--init-jitter", type=float, default=0.0)
    ap.add_argument("--init-relax-steps", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=0, help="块松弛粒子数（>1 启用）")
    ap.add_argument("--block-attempts", type=int, default=3, help="块松弛尝试次数")

    ap.add_argument("--max-attempts", type=int, default=200000, help="insert 模式最大尝试次数")
    ap.add_argument("--stagnation", type=int, default=20000, help="insert 模式最大连续失败数")
    ap.add_argument("--relax-steps", type=int, default=200, help="每次插入后的局部 MC 步数")
    ap.add_argument("--neighbor-deg", type=float, default=25.0, help="局部 MC 考虑的邻域角度（度）")
    ap.add_argument("--insertion-tries", type=int, default=10, help="单次插入候选采样次数")
    ap.add_argument("--parallel-proposals", type=int, default=1, help="局部松弛时并行评估的提案数")
    ap.add_argument("--global-sweep-every", type=int, default=0, help="insert 模式每插入多少粒子触发一次全局松弛（0 表示不触发）")
    ap.add_argument("--global-sweep-steps", type=int, default=200, help="触发全局松弛时运行 sweep 数")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coverage-samples", type=int, default=20000)
    ap.add_argument("--out", type=str, default="outputs/reversible_mc_state.npz")
    ap.add_argument("--png", type=str, default="outputs/reversible_scene.png")
    ap.add_argument("--html", type=str, default="outputs/reversible_scene.html")
    ap.add_argument("--no-visual", action="store_true", help="insert 模式不生成 PNG/HTML")

    ap.add_argument(
        "--align_body_vec",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="若设置，则对齐该体坐标向量至法向（单层贴壁）",
    )
    ap.add_argument(
        "--no_restrict_azimuthal",
        action="store_true",
        help="与 --align_body_vec 共用时，允许非纯方位旋转（每步重新对齐）",
    )
    ap.add_argument("--target-neighbor-angle", type=float, default=None, help="秩序驱动目标邻居夹角（度）")
    ap.add_argument("--order-strength", type=float, default=0.0, help="秩序势能强度（>0 启用 Metropolis 加权）")
    ap.add_argument("--order-k", type=int, default=6, help="秩序势能统计邻居数")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "run":
        run_simulation(args)
    else:
        insert_with_relaxation(args)


if __name__ == "__main__":
    main()
