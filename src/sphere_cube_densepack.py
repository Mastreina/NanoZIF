#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage dense packing of 198 nm MOF cubes onto a 5.2 µm spherical surface.

Stage 1 (geometric pre-pack):
    * Use HOOMD's HPMC ConvexPolyhedron integrator to treat each cube as a true
      hard convex cube.
    * Constrain the cube centers to a thin spherical shell via inner/outer hard
      walls, and perform multi-stage Lubachevsky–Stillinger-like growth:
      gradually inflate the cubes while Monte Carlo sweeps push them into a
      dense, overlap-free monolayer.

Stage 2 (RATTLE anneal):
    * Reinterpret every cube center as the rigid body center of a union-of-26
      WCA beads.
    * Constrain the centers to the exact target sphere with md.methods.rattle
      and run a soft-started Langevin anneal to relax orientations / bead
      overlaps.

Outputs:
    - traj.gsd  : MD stage trajectory sampled every WRITE_PERIOD steps.
    - final.gsd : Last MD frame (post-anneal).
    - log.csv   : Simple CSV log of MD observables (step, T, potential energy).

Adjust HPMC_STAGES / HPMC_SHELL_HALF to control Monte Carlo compression,
and MD parameters (STEPS_SOFTSTART / STEPS_ANNEAL, EPS_START/FINAL) to tune
the final relaxation depth.
"""

from __future__ import annotations

import math
import time
import shutil

import numpy as np

import hoomd
import hoomd.logging
import hoomd.md as md
import hoomd.hpmc as hpmc
import hoomd.hpmc.external.wall as hpmc_wall
import hoomd.wall

# ------------------------- User parameters -------------------------
R_SPHERE_NM      = 2600.0               # target sphere radius (nm)
CUBE_EDGE_NM     = 198.0                # cube edge length (nm)
TARGET_COVERAGE  = 0.90                 # φ used to estimate particle count
SEED             = 42                   # base RNG seed
MD_SEED_OFFSET   = 10_000               # offset for MD stage seed

# HPMC shell confinement + MC growth schedule
HPMC_SHELL_HALF  = 6.0                  # thickness/2 of spherical shell (nm)
HPMC_WORK_RADIUS_FACTOR = 1.08          # start radius factor (>1.0)
HPMC_SHRINK_STEPS = 22                  # how many radius stages
HPMC_SWEEPS_PER_STEP = 2_500            # MC sweeps at each radius
HPMC_TUNE_INTERVAL = 2_000              # sweeps between move-size tuning

# MD (RATTLE) parameters
DT               = 5.0e-4              # MD timestep (reduced units)
KBT              = 1.0                  # Langevin thermostat temperature
GAMMA_TRANS      = 5.0                  # translational drag
GAMMA_ROT        = (5.0, 5.0, 5.0)      # rotational drag tensor
STEPS_SOFTSTART  = 80_000               # ramp EPS_START -> EPS_FINAL
STEPS_ANNEAL     = 600_000              # optional RATTLE anneal
RUN_MD_RELAX     = False                # set True to perform WCA MD relaxation
WRITE_PERIOD     = 5_000                # MD frames per traj.gsd
LOG_PERIOD       = 1_000                # MD CSV logging period

# WCA bead representation of each cube face/edge/corner
BEAD_DIAM_NM     = CUBE_EDGE_NM / 5.0   # bead contact diameter
EPS_START        = 0.02                 # initial WCA epsilon
EPS_FINAL        = 1.50                 # final WCA epsilon
SOFTSTART_CHUNKS = 20                   # number of epsilon ramp segments

# ---------------------- Derived quantities -------------------------
R_target = R_SPHERE_NM
N_cubes = int(round(4.0 * math.pi * R_target**2 * TARGET_COVERAGE / (CUBE_EDGE_NM**2)))
sigma_bead = BEAD_DIAM_NM / (2.0 ** (1.0/6.0))
r_cut = BEAD_DIAM_NM
Lbox = 2.0 * (R_target + CUBE_EDGE_NM) + 200.0  # nm

print(f"Target radius R = {R_target:.1f} nm")
print(f"Approx. number of cubes N = {N_cubes} at φ = {TARGET_COVERAGE:.2f}")
print(f"HPMC shell: [{R_target - HPMC_SHELL_HALF:.1f}, "
      f"{R_target + HPMC_SHELL_HALF:.1f}] nm")

# ------------------------- Helper functions ------------------------
def fibonacci_sphere(n):
    """Evenly distribute n points on a unit sphere via golden-angle spiral."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    k = np.arange(n)
    z = 1.0 - 2.0 * (k + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = k * ga
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def cube_vertices(edge_nm):
    """Return the 8 vertices of a cube with the given edge length."""
    half = 0.5 * edge_nm
    verts = []
    for sx in (-half, half):
        for sy in (-half, half):
            for sz in (-half, half):
                verts.append((sx, sy, sz))
    return verts


def cube_union_positions(edge_nm, bead_contact_nm):
    """Return positions of beads approximating a cube with denser sampling."""
    r_eff = 0.5 * bead_contact_nm
    off = 0.5 * edge_nm - r_eff
    pts: list[tuple[float, float, float]] = []

    face_grid = off * 0.5
    for axis in range(3):
        for sign in (+off, -off):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = sign
            pts.append(tuple(pos))
            for s1 in (-face_grid, face_grid):
                for s2 in (-face_grid, face_grid):
                    sub = pos.copy()
                    sub[(axis + 1) % 3] = s1
                    sub[(axis + 2) % 3] = s2
                    pts.append(tuple(sub))

    for sx in (+off, -off):
        for sy in (+off, -off):
            pts.append((sx, sy, 0.0))
            pts.append((sx, 0.0, sy))
            pts.append((0.0, sy, sx))

    for sx in (+off, -off):
        for sy in (+off, -off):
            for sz in (+off, -off):
                pts.append((sx, sy, sz))

    return pts


def run_hpmc_prepack(device):
    """Stage-1: HPMC convex polyhedron MC with LS-like radius shrink."""
    print("\n=== HPMC convex-polyhedron pre-pack stage ===")
    snap = hoomd.Snapshot()
    snap.configuration.box = [Lbox, Lbox, Lbox, 0, 0, 0]
    snap.particles.N = N_cubes
    snap.particles.types = ['P']
    snap.particles.typeid[:] = 0
    R_work = R_target * HPMC_WORK_RADIUS_FACTOR
    snap.particles.position[:] = fibonacci_sphere(N_cubes) * (R_work - 0.2 * CUBE_EDGE_NM)
    snap.particles.orientation[:] = (1.0, 0.0, 0.0, 0.0)

    sim = hoomd.Simulation(device=device, seed=SEED)
    sim.create_state_from_snapshot(snap)

    mc = hpmc.integrate.ConvexPolyhedron(default_d=30.0, default_a=0.5)
    mc.shape['P'] = dict(vertices=cube_vertices(CUBE_EDGE_NM))
    sim.operations.integrator = mc

    def set_shell(radius_nm: float):
        inner = hoomd.wall.Sphere(radius=radius_nm - HPMC_SHELL_HALF, inside=False, open=True)
        outer = hoomd.wall.Sphere(radius=radius_nm + HPMC_SHELL_HALF, inside=True, open=True)
        mc.external_potentials = [hpmc_wall.WallPotential([inner, outer])]

    set_shell(R_work)

    history_centers = []
    history_orient = []
    with sim.state.cpu_local_snapshot as local_snap:
        history_centers.append(np.array(local_snap.particles.position, copy=True))
        history_orient.append(np.array(local_snap.particles.orientation, copy=True))

    radii = np.linspace(R_work, R_target, HPMC_SHRINK_STEPS)
    chunk = 2000
    for step_idx, radius in enumerate(radii, start=1):
        print(f" Shrink step {step_idx}/{HPMC_SHRINK_STEPS}: target radius {radius:.1f} nm")
        set_shell(radius)
        with sim.state.cpu_local_snapshot as local_snap:
            pos = np.array(local_snap.particles.position, copy=True)
            orient = np.array(local_snap.particles.orientation, copy=True)
            norms = np.linalg.norm(pos, axis=1, keepdims=True)
            pos = pos / norms * radius
            local_snap.particles.position[:] = pos
            history_centers.append(pos.copy())
            history_orient.append(orient.copy())
        sweeps_remaining = HPMC_SWEEPS_PER_STEP
        while sweeps_remaining > 0:
            run = min(chunk, sweeps_remaining)
            sim.run(run)
            sweeps_remaining -= run
        if (step_idx % max(1, HPMC_SHRINK_STEPS // 4)) == 0:
            mc.tune_kernel_parameters()

    with sim.state.cpu_local_snapshot as final_snap:
        centers = np.array(final_snap.particles.position, copy=True)
        orient = np.array(final_snap.particles.orientation, copy=True)

    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / norms * R_target
    hoomd.write.GSD.write(state=sim.state, filename='final_hpmc.gsd', mode='wb')
    np.savez('hpmc_history.npz', centers=np.stack(history_centers), orientations=np.stack(history_orient))
    print("HPMC pre-pack complete. Transferring configuration to MD stage.\n")
    return centers, orient


def build_md_sim(device, centers, orientations):
    """Create MD Simulation with rigid WCA cubes on the target sphere."""
    sim = hoomd.Simulation(device=device, seed=SEED + MD_SEED_OFFSET)
    snap = hoomd.Snapshot()
    snap.configuration.box = [Lbox, Lbox, Lbox, 0, 0, 0]
    snap.particles.N = N_cubes
    snap.particles.types = ['C', 'B']
    snap.particles.typeid[:] = 0
    snap.particles.mass[:] = 1.0
    inertia_val = (1.0 / 6.0) * (CUBE_EDGE_NM ** 2)
    snap.particles.moment_inertia[:] = np.full((N_cubes, 3), inertia_val)
    snap.particles.position[:] = centers
    snap.particles.orientation[:] = orientations
    snap.particles.velocity[:] = np.zeros((N_cubes, 3))
    snap.particles.angmom[:] = np.zeros((N_cubes, 4))
    sim.create_state_from_snapshot(snap)
    return sim


# --------------------------- Stage 1: HPMC -------------------------
device = hoomd.device.auto_select()
centers, orientations = run_hpmc_prepack(device)

if not RUN_MD_RELAX:
    shutil.copy('final_hpmc.gsd', 'final.gsd')

# --------------------------- Stage 2: MD (optional) ---------------------------
if RUN_MD_RELAX:
    sim = build_md_sim(device, centers, orientations)

    rigid = md.constrain.Rigid()
    bead_positions = cube_union_positions(CUBE_EDGE_NM, BEAD_DIAM_NM)
    rigid.body['C'] = dict(
        constituent_types=['B'] * len(bead_positions),
        positions=bead_positions,
        orientations=[(1.0, 0.0, 0.0, 0.0)] * len(bead_positions),
    )
    rigid.create_bodies(sim.state)

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
    rattle_method = md.methods.rattle.Langevin(
        filter=hoomd.filter.Rigid(flags=('center', 'free')),
        kT=KBT,
        manifold_constraint=sphere,
        default_gamma=GAMMA_TRANS,
        default_gamma_r=GAMMA_ROT,
    )
    integrator = md.Integrator(dt=DT, methods=[rattle_method], forces=[lj])
    integrator.rigid = rigid
    sim.operations.integrator = integrator

    thermo = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(sim, quantities=['timestep'])
    logger.add(thermo, quantities=['kinetic_temperature'])
    logger.add(lj, quantities=['energy'])

    log_file = open('log.csv', 'w', buffering=1)

    gsd_writer = hoomd.write.GSD(
        filename='traj.gsd',
        trigger=hoomd.trigger.Periodic(WRITE_PERIOD),
        mode='wb',
        dynamic=['property', 'momentum'],
    )
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(LOG_PERIOD),
        logger=logger,
        output=log_file,
        delimiter=',',
        header_sep='.'
    )
    sim.operations.writers.append(gsd_writer)
    sim.operations.writers.append(table)

    print("=== MD stage: soft-start + anneal on the manifold ===")
    try:
        sim.run(0)
        n_ramp = max(1, STEPS_SOFTSTART // SOFTSTART_CHUNKS)
        start_time = time.perf_counter()
        for i in range(SOFTSTART_CHUNKS):
            eps = EPS_START + (EPS_FINAL - EPS_START) * (i + 1) / SOFTSTART_CHUNKS
            lj.params[('B', 'B')] = dict(sigma=sigma_bead, epsilon=eps)
            sim.run(n_ramp)
            elapsed = (time.perf_counter() - start_time) / 60.0
            print(f"  soft-start chunk {i + 1}/{SOFTSTART_CHUNKS} complete "
                  f"({elapsed:.2f} min)")

        print("  Annealing at target radius...")
        anneal_start = time.perf_counter()
        sim.run(STEPS_ANNEAL)
        anneal_elapsed = (time.perf_counter() - anneal_start) / 60.0
        print(f"  Anneal finished in {anneal_elapsed:.2f} min")

        hoomd.write.GSD.write(state=sim.state, filename='final.gsd')
        print("Done. MD trajectory -> traj.gsd, final frame -> final.gsd")
    finally:
        if table in sim.operations.writers:
            sim.operations.writers.remove(table)
        log_file.close()
else:
    print("Skipping MD relaxation (RUN_MD_RELAX = False). Final HPMC configuration written to final_hpmc.gsd.")
