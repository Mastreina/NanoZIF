#!/usr/bin/env python3
"""
Export HOOMD GSD frames to XYZ files for ParaView / VMD / OVITO.

Example:
    mamba run -n hoomd python export_gsd_xyz.py \
        --traj traj.gsd --every 1000 --prefix paraview_centers
"""

import argparse
import pathlib

import numpy as np
import gsd.hoomd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default="traj.gsd",
                        help="输入 GSD 文件 (默认 traj.gsd)")
    parser.add_argument("--prefix", default="paraview",
                        help="输出 XYZ 前缀 (默认 paraview)")
    parser.add_argument("--start", type=int, default=0,
                        help="起始帧 (默认 0)")
    parser.add_argument("--stop", type=int, default=None,
                        help="终止帧 (默认最后一帧)")
    parser.add_argument("--stride", type=int, default=1,
                        help="间隔帧数 (默认 1)")
    parser.add_argument("--centers-only", action="store_true",
                        help="只导出刚体中心 (typeid==0)")
    return parser.parse_args()


def write_xyz(path, positions, comment):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(positions)}\n")
        fh.write(comment + "\n")
        for x, y, z in positions:
            fh.write(f"C {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    args = parse_args()
    traj_path = pathlib.Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(f"找不到 {traj_path}")

    data = gsd.hoomd.open(str(traj_path))
    stop = args.stop if args.stop is not None else len(data)
    frames = range(args.start, stop, args.stride)

    for frame in frames:
        snap = data[frame]
        pos = np.asarray(snap.particles.position)
        if args.centers_only:
            typeid = np.asarray(snap.particles.typeid)
            pos = pos[typeid == 0]
        out = traj_path.with_name(f"{args.prefix}_frame{frame:04d}.xyz")
        comment = f"frame {frame}, N={len(pos)}"
        write_xyz(out, pos, comment)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
