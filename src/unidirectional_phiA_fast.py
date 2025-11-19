import math, time
import numpy as np
from typing import List, Tuple, Optional, Callable

from .unidirectional_adsorption_polygons import AdsorptionPatch, log_map, rot2

def build_grid(patches: List[AdsorptionPatch], R_nm: float, poly_base: np.ndarray):
    r_max = float(np.max(np.linalg.norm(poly_base, axis=1)))
    theta_poly = r_max / R_nm
    J = max(8, int(math.ceil(math.pi/(2*theta_poly))))
    K = max(16, int(math.ceil(2*math.pi/(2*theta_poly))))
    dtheta = math.pi / J; dphi = 2*math.pi / K
    grid = [[[] for _ in range(K)] for __ in range(J)]
    for idx, p in enumerate(patches):
        theta = math.acos(np.clip(p.c[2], -1.0, 1.0))
        i = int(theta/dtheta); i = min(J-1, max(0, i))
        phi = math.atan2(p.c[1], p.c[0]); phi = phi if phi>=0 else (phi+2*math.pi)
        k = int(phi/dphi); k = min(K-1, max(0, k))
        grid[i][k].append(idx)
    return grid, (J,K,dtheta,dphi), theta_poly

def point_in_convex_polygon(q: np.ndarray, verts: np.ndarray) -> bool:
    e = np.roll(verts, -1, axis=0) - verts
    n = np.stack((-e[:,1], e[:,0]), axis=1)
    return np.all((q - verts) @ n.T <= 1e-12)

def estimate_adsorption_coverage(patches: List[AdsorptionPatch], R_nm: float, poly_base: np.ndarray,
                          M: int = 50_000, seed: int = 1234, progress: bool = True,
                          progress_hook: Optional[Callable[[int, int, float], None]] = None) -> Tuple[float,float]:
    rng = np.random.default_rng(seed)
    grid, (J,K,dtheta,dphi), theta_poly = build_grid(patches, R_nm, poly_base)
    alpha_max = theta_poly
    covered = 0
    t0 = time.time()

    for m in range(1, M+1):
        z = rng.uniform(-1.0, 1.0); t = rng.uniform(0.0, 2*math.pi)
        r_xy = math.sqrt(max(0.0, 1.0 - z*z))
        p = np.array([math.cos(t)*r_xy, math.sin(t)*r_xy, z])

        theta = math.acos(max(-1.0, min(1.0, p[2])))
        i = int(theta/dtheta); i = min(J-1, max(0, i))
        phi = math.atan2(p[1], p[0]); phi = phi if phi>=0 else (phi+2*math.pi)
        k = int(phi/dphi); k = min(K-1, max(0, k))

        sin_th = math.sin(theta)
        reach_i = max(1, int(math.ceil(alpha_max/dtheta)))
        reach_k = (K if sin_th < 1e-6 else max(1, int(math.ceil(alpha_max/(sin_th*dphi)))))

        hit = False
        for di in range(-reach_i, reach_i+1):
            ii = i+di
            if 0 <= ii < J:
                for dk in range(-reach_k, reach_k+1):
                    kk = (k+dk) % K
                    for idx in grid[ii][kk]:
                        patch = patches[idx]
                        ang = math.acos(float(np.clip(np.dot(p, patch.c), -1.0, 1.0)))
                        if ang > alpha_max:
                            continue
                        _, w = log_map(patch.c, p, R_nm)
                        Rloc = rot2(-patch.phi)
                        q = Rloc @ w
                        if point_in_convex_polygon(q, poly_base):
                            hit = True; break
                    if hit: break
            if hit: break

        if hit: covered += 1
        if progress_hook:
            rate = covered/m
            progress_hook(m, covered, rate)
        elif progress and (m % max(1000, M//20) == 0 or m == M):
            rate = covered/m
            spd = m/max(1e-9, time.time()-t0)
            eta = (M-m)/max(1e-9, spd)
            print(f"[{m:>7}/{M}] φ_A≈{rate:.5f} | {int(spd)} samp/s | ETA={int(eta//60):02d}:{int(eta%60):02d}")

    phiA = covered/float(M)
    err  = math.sqrt(max(phiA*(1-phiA), 1e-16)/float(M))
    if progress:
        print(f"MC(phi_A fast) done | φ_A={phiA:.5f} ± {err:.5f}")
    return phiA, err
