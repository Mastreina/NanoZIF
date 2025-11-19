#!/usr/bin/env python3
"""
Render traj.gsd / final.gsd with fresnel path tracing.

Usage:
    mamba run -n hoomd python render_fresnel.py [--traj traj.gsd] [--frame -1]

This will read the requested frame, place convex polyhedron cubes at the
rigid-body centers, and save a PNG (render_frameXXXX.png).
"""

import argparse
import pathlib

import numpy as np
from PIL import Image
import fresnel
import gsd.hoomd

RADIUS_NM = 2600.0
CUBE_EDGE_NM = 198.0


def cube_vertices(edge):
    half = 0.5 * edge
    verts = []
    for sx in (-half, half):
        for sy in (-half, half):
            for sz in (-half, half):
                verts.append((sx, sy, sz))
    return np.asarray(verts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default="traj.gsd", help="GSD trajectory to render")
    parser.add_argument("--frame", type=int, default=-1,
                        help="Frame index (default last frame)")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--camera", choices=("perspective", "orthographic"),
                        default="perspective")
    parser.add_argument("--distance", type=float, default=RADIUS_NM + 800.0,
                        help="摄像机距离球心的径向距离")
    parser.add_argument("--theta", type=float, default=35.0,
                        help="极角（度），0°沿 +z 方向")
    parser.add_argument("--phi", type=float, default=40.0,
                        help="方位角（度），0°沿 +x，90°沿 +y")
    parser.add_argument("--ortho-height", type=float, default=2.5,
                        help="正交相机覆盖高度（乘以球半径）")
    args = parser.parse_args()

    traj_path = pathlib.Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(f"找不到 {traj_path}")

    data = gsd.hoomd.open(str(traj_path))
    snap = data[args.frame]
    typeid = np.asarray(snap.particles.typeid)
    centers = np.asarray(snap.particles.position)[typeid == 0]
    orientations = np.asarray(snap.particles.orientation)[typeid == 0]

    scene = fresnel.Scene()
    verts = cube_vertices(CUBE_EDGE_NM)
    poly_info = fresnel.util.convex_polyhedron_from_vertices(verts)
    # light grey sphere for contrast
    fresnel.geometry.Sphere(
        scene,
        radius=RADIUS_NM,
        position=[(0, 0, 0)],
        color=fresnel.color.linear([0.02, 0.02, 0.03]),
        material=fresnel.material.Material(
            roughness=0.95,
            specular=0.02,
            color=fresnel.color.linear([0.02, 0.02, 0.03]),
        ),
    )

    geometry = fresnel.geometry.ConvexPolyhedron(
        scene,
        polyhedron_info=poly_info,
        position=centers,
        orientation=orientations,
        color=fresnel.color.linear([0.20, 0.45, 0.85]),
        material=fresnel.material.Material(
            roughness=0.45,
            specular=0.65,
            color=fresnel.color.linear([0.20, 0.45, 0.85]),
            metal=0.1,
        ),
    )
    geometry.outline_width = 1.2

    theta = np.deg2rad(args.theta)
    phi = np.deg2rad(args.phi)
    cam_pos = (
        args.distance * np.sin(theta) * np.cos(phi),
        args.distance * np.sin(theta) * np.sin(phi),
        args.distance * np.cos(theta),
    )

    if args.camera == "orthographic":
        scene.camera = fresnel.camera.Orthographic(
            position=cam_pos,
            look_at=(0, 0, 0),
            up=(0, 0, 1),
            height=args.ortho_height * RADIUS_NM,
        )
    else:
        scene.camera = fresnel.camera.Perspective(
            position=cam_pos,
            look_at=(0, 0, 0),
            up=(0, 0, 1),
            focal_length=0.4,
            focus_distance=np.linalg.norm(cam_pos),
            height=0.35,
        )

    scene.lights = fresnel.light.rembrandt()
    scene.background_color = (0.04, 0.04, 0.07)

    print(f"Rendering frame {args.frame} from {traj_path} ...")
    out = fresnel.pathtrace(scene, w=args.width, h=args.height, samples=args.samples)
    frame_idx = args.frame if args.frame >= 0 else len(data) + args.frame
    png_path = traj_path.with_name(f"render_frame{frame_idx:04d}.png")
    rgb = np.clip(out[:, :, :3], 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.save(png_path)
    print(f"完成：{png_path}")


if __name__ == "__main__":
    main()
