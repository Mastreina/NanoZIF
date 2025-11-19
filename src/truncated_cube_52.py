#!/usr/bin/env python3
"""
Dense packing of truncated cubes (edge = 210 nm) on a 5.2 µm sphere.

Uses the same two-stage procedure: HPMC convex polyhedron compression followed
by MD RATTLE annealing with a union-of-spheres WCA model constrained on the sphere.
"""

from __future__ import annotations

import math
import treloadime
from itertools import product, permutations

import numpy as np
import hoomd
import hoomd.md as md
import hoomd.hpmc as hpmc
import hoomd.hpmc.external.wall as hpmc_wall
import hoomd.wall
import hoomd.logging

# ------------------------- User parameters -------------------------
R_SPHERE_NM = 2600.0            # 5.2 µm diameter sphere
EDGE_NM = 210.0                 # truncated cube edge length
TRUNCATION_RATIO = 3.0          # controls how deep the truncation is (>1 -> more cut)
TARGET_COVERAGE = 0.90
SEED = 314
MD_SEED_OFFSET = 7_000

HPMC_SHELL_HALF = 7.0
HPMC_STAGES = (
    dict(scale=0.80, sweeps=15_000, d=36.0, a=0.55),
    dict(scale=0.93, sweeps=22_000, d=22.0, a=0.38),
    dict(scale=1.00, sweeps=28_000, d=14.0, a=0.28),
)

DT = 5.0e-4
KBT = 1.0
GAMMA_TRANS = 4.5
GAMMA_ROT = (4.5, 4.5, 4.5)
STEPS_SOFTSTART = 70_000
STEPS_ANNEAL = 420_000
WRITE_PERIOD = 5_000
LOG_PERIOD = 1_000
SOFTSTART_CHUNKS = 20

BEAD_DIAM_NM = EDGE_NM / 6.0
EPS_START = 0.02
EPS_FINAL = 0.60

# ---------------------- Derived quantities -------------------------
R_target = R_SPHERE_NM
N_particles = int(round(4.0 * math.pi * R_target**2 * TARGET_COVERAGE / (EDGE_NM**2)))
sigma_bead = BEAD_DIAM_NM / (2.0 ** (1.0 / 6.0))
r_cut = BEAD_DIAM_NM
Lbox = 2.0 * (R_target + EDGE_NM) + 200.0

print(f"Truncated cube edge = {EDGE_NM:.1f} nm, sphere radius = {R_target:.1f} nm")
print(f"Approx particle count ≈ {N_particles}")


# ------------------------- Helper functions ------------------------
def fibonacci_sphere(n: int) -> np.ndarray:
    ga = math.pi * (3.0 - math.sqrt(5.0))
    k = np.arange(n)
    z = 1.0 - 2.0 * (k + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = k * ga
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def truncated_cube_vertices(edge_nm: float, trunc_ratio: float = TRUNCATION_RATIO) -> list[tuple[float, float, float]]:
    base_perms = {perm for perm in permutations((1.0, 1.0, trunc_ratio))}
    coords = set()
    for perm in base_perms:
        for sx, sy, sz in product((-1.0, 1.0), repeat=3):
            coords.add((sx * perm[0], sy * perm[1], sz * perm[2]))
    scale = edge_nm / math.sqrt(8.0)  # base edge length = sqrt(8)
    return [(scale * x, scale * y, scale * z) for x, y, z in coords]


def truncated_cube_beads() -> list[tuple[float, float, float]]:
    beads = truncated_cube_vertices(EDGE_NM, trunc_ratio=TRUNCATION_RATIO)
    # add centers of octagonal faces for better coverage
    face_offset = EDGE_NM / 3.0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = sign * face_offset
            beads.append(tuple(pos))
    return beads


def quat_from_u_to_v(u_vec, v_vec):
    u_vec = np.asarray(u_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    u_vec /= np.linalg.norm(u_vec)
    v_vec /= np.linalg.norm(v_vec)
    dot = np.clip(np.dot(u_vec, v_vec), -1.0, 1.0)
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.999999:
        axis = np.cross(u_vec, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(u_vec, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(u_vec, v_vec)
    w = 1.0 + dot
    quat = np.array([w, axis[0], axis[1], axis[2]])
    quat /= np.linalg.norm(quat)
    return quat


def build_snapshot(centers, orientations):
    snap = hoomd.Snapshot()
    snap.configuration.box = [Lbox, Lbox, Lbox, 0, 0, 0]
    snap.particles.N = centers.shape[0]
    snap.particles.types = ['C', 'B']
    snap.particles.typeid[:] = 0
    snap.particles.mass[:] = 1.0
    moi_val = (1.0 / 6.0) * (EDGE_NM ** 2)
    snap.particles.moment_inertia[:] = (moi_val, moi_val, moi_val)
    snap.particles.position[:] = centers
    snap.particles.orientation[:] = orientations
    snap.particles.velocity[:] = 0.0
    snap.particles.angmom[:] = 0.0
    return snap


# --------------------------- Stage 1 ---------------------------
def run_hpmc(device):
    snap = hoomd.Snapshot()
    snap.configuration.box = [Lbox, Lbox, Lbox, 0, 0, 0]
    snap.particles.N = N_particles
    snap.particles.types = ['P']
    snap.particles.typeid[:] = 0
    snap.particles.position[:] = fibonacci_sphere(N_particles) * (R_target - EDGE_NM * 0.4)
    snap.particles.orientation[:] = (1.0, 0.0, 0.0, 0.0)

    sim = hoomd.Simulation(device=device, seed=SEED)
    sim.create_state_from_snapshot(snap)

    mc = hpmc.integrate.ConvexPolyhedron(default_d=HPMC_STAGES[0]['d'], default_a=HPMC_STAGES[0]['a'])
    base_vertices = truncated_cube_vertices(EDGE_NM, trunc_ratio=TRUNCATION_RATIO)
    mc.shape['P'] = dict(vertices=base_vertices)
    sim.operations.integrator = mc

    inner = hoomd.wall.Sphere(radius=R_target - HPMC_SHELL_HALF, inside=False, open=True)
    outer = hoomd.wall.Sphere(radius=R_target + HPMC_SHELL_HALF, inside=True, open=True)
    mc.external_potentials = [hpmc_wall.WallPotential([inner, outer])]

    for stage_id, stage in enumerate(HPMC_STAGES, start=1):
        scaled = [(stage['scale'] * x, stage['scale'] * y, stage['scale'] * z) for (x, y, z) in base_vertices]
        mc.shape['P'] = dict(vertices=scaled)
        mc.d['P'] = stage['d']
        mc.a['P'] = stage['a']
        sweeps = stage['sweeps']
        print(f"HPMC stage {stage_id}: scale={stage['scale']}, sweeps={sweeps}")
        completed = 0
        chunk = max(2000, sweeps // 10)
        while completed < sweeps:
            step = min(chunk, sweeps - completed)
            sim.run(step)
            completed += step
            if completed == sweeps or completed % (5 * chunk) == 0:
                print(f"  progress {completed}/{sweeps}")

    with sim.state.cpu_local_snapshot as final_snap:
        centers = np.array(final_snap.particles.position, copy=True)
        orientations = np.array(final_snap.particles.orientation, copy=True)

    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= R_target
    return centers, orientations


# --------------------------- Stage 2 ---------------------------
def build_md_sim(device, centers, orientations):
    sim = hoomd.Simulation(device=device, seed=SEED + MD_SEED_OFFSET)
    sim.create_state_from_snapshot(build_snapshot(centers, orientations))
    return sim


def setup_rigid(sim):
    rigid = md.constrain.Rigid()
    beads = truncated_cube_beads()
    rigid.body['C'] = dict(
        constituent_types=['B'] * len(beads),
        positions=beads,
        orientations=[(1.0, 0.0, 0.0, 0.0)] * len(beads),
    )
    rigid.create_bodies(sim.state)
    return rigid


def setup_interactions(rigid):
    nlist = md.nlist.Cell(buffer=0.4)
    nlist.exclusions = ('body',)
    lj = md.pair.LJ(nlist=nlist, default_r_cut=0.0, mode='shift')
    lj.r_cut[('B', 'B')] = r_cut
    lj.params[('B', 'B')] = dict(sigma=sigma_bead, epsilon=EPS_START)
    zero = dict(sigma=1.0, epsilon=0.0)
    for pair in (('B', 'C'), ('C', 'B'), ('C', 'C')):
        lj.r_cut[pair] = 0.0
        lj.params[pair] = zero
    sphere = md.manifold.Sphere(r=R_target)
    method = md.methods.rattle.Langevin(
        filter=hoomd.filter.Rigid(flags=('center', 'free')),
        kT=KBT,
        manifold_constraint=sphere,
        default_gamma=GAMMA_TRANS,
        default_gamma_r=GAMMA_ROT,
    )
    integrator = md.Integrator(dt=DT, methods=[method], forces=[lj])
    integrator.rigid = rigid
    return integrator, lj


def run_md(sim, integrator, lj):
    thermo = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(sim, quantities=['timestep'])
    logger.add(thermo, quantities=['kinetic_temperature'])
    logger.add(lj, quantities=['energy'])

    log_file = open('log_trunc_cube.csv', 'w', buffering=1)
    gsd_writer = hoomd.write.GSD(
        filename='traj_trunc_cube.gsd',
        trigger=hoomd.trigger.Periodic(WRITE_PERIOD),
        mode='wb',
        dynamic=['property', 'momentum'],
    )
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(LOG_PERIOD),
        logger=logger,
        output=log_file,
        delimiter=',',
        header_sep='.',
    )
    sim.operations.writers.append(gsd_writer)
    sim.operations.writers.append(table)

    print("Soft-start ramp...")
    n_ramp = max(1, STEPS_SOFTSTART // SOFTSTART_CHUNKS)
    t0 = time.perf_counter()
    try:
        for chunk in range(SOFTSTART_CHUNKS):
            eps = EPS_START + (EPS_FINAL - EPS_START) * (chunk + 1) / SOFTSTART_CHUNKS
            lj.params[('B', 'B')] = dict(sigma=sigma_bead, epsilon=eps)
            sim.run(n_ramp)
            elapsed = (time.perf_counter() - t0) / 60.0
            print(f"  chunk {chunk + 1}/{SOFTSTART_CHUNKS} ({elapsed:.2f} min)")

        print("Annealing at target radius...")
        anneal_t0 = time.perf_counter()
        sim.run(STEPS_ANNEAL)
        print(f"  anneal finished in {(time.perf_counter() - anneal_t0)/60.0:.2f} min")

        hoomd.write.GSD.write(state=sim.state, filename='final_trunc_cube.gsd')
        print("Done. Outputs: traj_trunc_cube.gsd, final_trunc_cube.gsd, log_trunc_cube.csv")
    finally:
        if table in sim.operations.writers:
            sim.operations.writers.remove(table)
        log_file.close()


def main():
    device = hoomd.device.auto_select()
    centers, orientations = run_hpmc(device)
    sim = build_md_sim(device, centers, orientations)
    rigid = setup_rigid(sim)
    integrator, lj = setup_interactions(rigid)
    sim.operations.integrator = integrator
    run_md(sim, integrator, lj)


if __name__ == "__main__":
    main()
