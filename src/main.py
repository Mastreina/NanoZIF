"""命令行入口：运行蒙特卡洛球面铺展模拟并输出结果。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

from .interactive_viz import build_interactive_html
from .monte_carlo_sim import MonteCarloConfig, SphereSurfaceMonteCarlo
from .visualization import plot_3d, plot_equirectangular


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo sphere-surface square tiling")
    parser.add_argument("--radius", type=float, default=2.55, help="球体半径 (nm)")
    parser.add_argument("--tile-side", type=float, default=0.6, help="正方形边长 (nm)")
    parser.add_argument(
        "--shape",
        type=str,
        default="square",
        choices=["square", "hexagon", "hexagonal_prism", "octagon", "truncated_octahedron", "truncated_cube"],
        help="贴片形状",
    )
    parser.add_argument("--max-steps", type=int, default=200_000, help="蒙特卡洛最大步数")
    parser.add_argument("--stall-steps", type=int, default=20_000, help="无改进终止阈值")
    parser.add_argument("--insert-prob", type=float, default=0.55, help="插入操作概率")
    parser.add_argument("--translation-step", type=float, default=3.0, help="位移角度步长 (度)")
    parser.add_argument("--rotation-step", type=float, default=5.0, help="旋转步长 (度)")
    parser.add_argument("--coverage-samples", type=int, default=20_000, help="覆盖度估计采样数")
    parser.add_argument("--gap-sample", type=int, default=48, help="gap启发式候选数")
    parser.add_argument(
        "--gap-insert-bias",
        type=float,
        default=0.85,
        help="使用gap启发式进行插入的概率(0~1)",
    )
    parser.add_argument(
        "--targeted-insert-trigger",
        type=int,
        default=4_000,
        help="连续多少步无变化后执行覆盖驱动插入(0表示禁用)",
    )
    parser.add_argument(
        "--targeted-insert-attempts",
        type=int,
        default=4,
        help="触发时针对性插入的尝试次数",
    )
    parser.add_argument(
        "--targeted-samples",
        type=int,
        default=6_000,
        help="针对性插入估计空隙的采样点数量",
    )
    parser.add_argument(
        "--global-relax-trigger",
        type=int,
        default=12_000,
        help="全局退火/松弛触发的步间隔(0表示禁用)",
    )
    parser.add_argument(
        "--init-layout",
        type=str,
        default="none",
        choices=["none", "fibonacci", "caspar_klug"],
        help="初始化布点策略",
    )
    parser.add_argument(
        "--init-count",
        type=int,
        default=0,
        help="初始化布点数量(0表示按面积估算)",
    )
    parser.add_argument(
        "--caspar-h",
        type=int,
        default=1,
        help="Caspar-Klug 参量h",
    )
    parser.add_argument(
        "--caspar-k",
        type=int,
        default=0,
        help="Caspar-Klug 参量k",
    )
    parser.add_argument("--rng-seed", type=int, default=None, help="随机数种子")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="输出目录（覆盖度、投影、三维展示会写入此处）",
    )
    parser.add_argument(
        "--overlap-resolution",
        type=int,
        default=7,
        help="重叠判定的面片采样分辨率(整数>2)",
    )
    parser.add_argument(
        "--overlap-tolerance",
        type=float,
        default=0.0,
        help="允许的重叠比例(0表示不允许，1表示完全忽略重叠)",
    )
    parser.add_argument(
        "--interactive-html",
        type=Path,
        default=None,
        help="若提供，则输出Plotly交互式HTML文件",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="显示蒙特卡洛迭代进度条",
    )
    parser.add_argument(
        "--energy-temp-start",
        type=float,
        default=1.2,
        help="局部退火初始温度",
    )
    parser.add_argument(
        "--energy-temp-end",
        type=float,
        default=0.05,
        help="局部退火结束温度",
    )
    parser.add_argument(
        "--energy-batches",
        type=int,
        default=3,
        help="每次插入后的退火批次数",
    )
    parser.add_argument(
        "--energy-moves",
        type=int,
        default=120,
        help="每个退火批次的尝试次数",
    )
    parser.add_argument(
        "--energy-softness",
        type=float,
        default=1.0,
        help="能量拒绝的软化系数(>1时更容易接受)",
    )
    parser.add_argument(
        "--force-insert-iterations",
        type=int,
        default=300,
        help="硬插入局部优化的最大迭代次数",
    )
    parser.add_argument(
        "--force-insert-step",
        type=float,
        default=0.25,
        help="硬插入每次力场更新的步长",
    )
    parser.add_argument(
        "--force-insert-rotation",
        type=float,
        default=6.0,
        help="硬插入时附加旋转的标准差(度)",
    )
    parser.add_argument(
        "--force-insert-neighbors",
        type=int,
        default=24,
        help="硬插入参与优化的邻居数量上限",
    )
    parser.add_argument(
        "--disable-force-insert",
        action="store_true",
        help="关闭硬插入力场优化",
    )
    parser.add_argument(
        "--energy-contact-scale",
        type=float,
        default=0.9,
        help="能量模型中的接触尺度(乘以边长)",
    )
    parser.add_argument(
        "--energy-target-scale",
        type=float,
        default=1.1,
        help="能量模型的理想距离尺度(乘以边长)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="断点保存文件路径(.npz)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="每隔多少步写入断点(0表示关闭)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="从给定断点文件继续模拟",
    )
    return parser


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


def save_metadata(result, config: MonteCarloConfig, output_dir: Path) -> None:
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
        "cooperative_insert_attempts": config.cooperative_insert_attempts,
        "cooperative_move_scale": config.cooperative_move_scale,
        "post_insert_relax_steps": config.post_insert_relax_steps,
        "relax_translation_scale": config.relax_translation_scale,
        "relax_rotation_scale": config.relax_rotation_scale,
        "neighbor_relax_angle_deg": math.degrees(config.neighbor_relax_angle),
        "energy_contact_scale": config.energy_contact_scale,
        "energy_target_scale": config.energy_target_scale,
        "energy_temperature_start": config.energy_temperature_start,
        "energy_temperature_end": config.energy_temperature_end,
        "energy_batches": config.energy_batches,
        "energy_moves_per_batch": config.energy_moves_per_batch,
        "energy_acceptance_softness": config.energy_acceptance_softness,
        "force_insert_enabled": config.force_insert_enabled,
        "force_insert_iterations": config.force_insert_iterations,
        "force_insert_step": config.force_insert_step,
        "force_insert_rotation_sigma_deg": math.degrees(config.force_insert_rotation_sigma),
        "force_insert_neighbor_limit": config.force_insert_neighbor_limit,
        "checkpoint_interval": config.checkpoint_interval,
        "shape": config.shape,
        "shape_sides": SphereSurfaceMonteCarlo._resolve_sides(config.shape.lower()),
        "gap_sample_size": config.gap_sample_size,
        "gap_insert_bias": config.gap_insert_bias,
        "targeted_insert_trigger": config.targeted_insert_trigger,
        "targeted_insert_attempts": config.targeted_insert_attempts,
        "targeted_samples": config.targeted_samples,
        "global_relax_trigger": config.global_relax_trigger,
        "initial_layout": config.initial_layout,
        "initial_count": config.initial_count,
        "caspar_h": config.initial_caspar_h,
        "caspar_k": config.initial_caspar_k,
        "overlap_tolerance": config.overlap_tolerance,
    }
    with (output_dir / "result_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
