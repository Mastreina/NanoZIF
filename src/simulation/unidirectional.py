"""
Unidirectional Adsorption MC: Simulate adsorption of convex polygons on a sphere.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Generator

import numpy as np
from numpy.typing import NDArray

from ..geometry.core import (
    normalize,
    orthonormal_basis,
    quat_from_axis_angle,
    rotate_vec_by_quat,
)


# ========= 基础几何 =========

def rot_axis_angle(u: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector u around axis by angle."""
    # Using quaternion implementation from geometry core
    q = quat_from_axis_angle(axis, angle)
    return rotate_vec_by_quat(u, q)


def log_map(c: np.ndarray, p: np.ndarray, R: float) -> Tuple[float, np.ndarray]:
    """Logarithmic map from sphere to tangent plane at c."""
    dotcp = float(np.clip(np.dot(c, p), -1.0, 1.0))
    alpha = math.acos(dotcp)
    if alpha < 1e-12:
        return 0.0, np.array([0.0, 0.0])
    t = normalize(p - dotcp * c)
    e1, e2, _ = orthonormal_basis(c)
    # Note: orthonormal_basis returns (u, v, n). Here e1=u, e2=v.
    w = R * alpha * np.array([float(np.dot(t, e1)), float(np.dot(t, e2))])
    return alpha, w


def transport_tangent_vector(c_from: np.ndarray, c_to: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Parallel transport tangent vector v from c_from to c_to."""
    dotc = float(np.clip(np.dot(c_from, c_to), -1.0, 1.0))
    ang = math.acos(dotc)
    if ang < 1e-12:
        return v
    axis = normalize(np.cross(c_from, c_to))
    return rot_axis_angle(v, axis, ang)


# ========= 2D 多边形工具（凸）=========

def rot2(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def polygon_edges(verts: np.ndarray) -> np.ndarray:
    return np.roll(verts, -1, axis=0) - verts


def polygon_axes_from_edges(edges: np.ndarray) -> np.ndarray:
    axes = []
    for e in edges:
        n = np.array([-e[1], e[0]], dtype=float)
        ln = float(np.linalg.norm(n))
        if ln > 1e-12:
            axes.append(n / ln)
    return np.array(axes)


def interval_on_axis(verts: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    projs = verts @ axis
    return float(np.min(projs)), float(np.max(projs))


def polygons_overlap_SAT(offset: np.ndarray, V1: np.ndarray, V2: np.ndarray) -> bool:
    V2s = V2 + offset
    axes = np.vstack(
        [
            polygon_axes_from_edges(polygon_edges(V1)),
            polygon_axes_from_edges(polygon_edges(V2)),
        ]
    )
    for ax in axes:
        a1, b1 = interval_on_axis(V1, ax)
        a2, b2 = interval_on_axis(V2s, ax)
        if b1 < a2 or b2 < a1:
            return False
    return True


# ========= 贴片数据 =========

@dataclass
class AdsorptionPatch:
    c: np.ndarray
    phi: float
    e1: np.ndarray
    e2: np.ndarray


# ========= 形状生成器 =========

def square_vertices(edge_nm: float) -> np.ndarray:
    s = edge_nm / 2.0
    return np.array([[-s, -s], [s, -s], [s, s], [-s, s]], dtype=float)


def regular_polygon_vertices(n: int, side_nm: float) -> np.ndarray:
    R = side_nm / (2.0 * math.sin(math.pi / n))
    ang0 = math.pi / n
    return np.array(
        [
            [R * math.cos(ang0 + 2 * k * math.pi / n), R * math.sin(ang0 + 2 * k * math.pi / n)]
            for k in range(n)
        ],
        dtype=float,
    )


def octagon_from_cube_face(cube_edge_nm: float) -> np.ndarray:
    s = (math.sqrt(2.0) - 1.0) * cube_edge_nm
    return regular_polygon_vertices(8, s)


# ========= 单向吸附 MC 主过程 =========

def adsorption_polygons_on_sphere(
    R_nm: float,
    poly_base: np.ndarray,
    seed: int = 0,
    max_attempts: int = 200_000,
    stagnation_limit: int = 20_000,
    progress: bool = True,
    progress_every: int = 5_000,
    return_history: bool = False,
    # —— 可选局部让位（Relaxed 吸附）参数 ——
    relax_enabled: bool = False,
    relax_r_nbr_factor: float = 2.5,
    relax_step_frac: float = 0.15,
    relax_phi_step_deg: float = 8.0,
    relax_max_iter: int = 20,
    relax_max_arc_frac: float = 0.8,
    relax_shuffle: bool = True,
    progress_hook: Optional[Callable[[int, int, int], None]] = None,
):
    rng = np.random.default_rng(seed)

    r_max = float(np.max(np.linalg.norm(poly_base, axis=1)))
    theta_poly = r_max / R_nm
    alpha_max = 2.0 * theta_poly

    cell_angle = 2.0 * theta_poly
    J = max(8, int(math.ceil(math.pi / cell_angle)))
    K = max(16, int(math.ceil(2 * math.pi / cell_angle)))
    dtheta = math.pi / J
    dphi = 2 * math.pi / K
    grid: List[List[List[int]]] = [[[] for _ in range(K)] for __ in range(J)]
    patch_bin: List[Tuple[int, int]] = []  # 每个 patch 当前所在的 (i,k)

    def unit_random() -> np.ndarray:
        z = rng.uniform(-1.0, 1.0)
        t = rng.uniform(0.0, 2 * math.pi)
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        return np.array([math.cos(t) * r_xy, math.sin(t) * r_xy, z], dtype=float)

    def bin_index(dir_vec: np.ndarray) -> Tuple[int, int, float]:
        theta = math.acos(np.clip(dir_vec[2], -1.0, 1.0))
        i = int(theta / dtheta)
        i = min(J - 1, max(0, i))
        phi = math.atan2(dir_vec[1], dir_vec[0])
        phi = phi if phi >= 0 else (phi + 2 * math.pi)
        k = int(phi / dphi)
        k = min(K - 1, max(0, k))
        return i, k, theta

    def neighbor_bins_metric(i: int, k: int, theta: float) -> Generator[Tuple[int, int], None, None]:
        sin_th = math.sin(theta)
        reach_i = max(1, int(math.ceil(alpha_max / dtheta)))
        if sin_th < 1e-6:
            reach_k = K
        else:
            reach_k = max(1, int(math.ceil(alpha_max / (sin_th * dphi))))
        for di in range(-reach_i, reach_i + 1):
            ii = i + di
            if 0 <= ii < J:
                for dk in range(-reach_k, reach_k + 1):
                    yield ii, (k + dk) % K

    def neighbor_bins_by_angle(i: int, k: int, theta: float, angle: float) -> Generator[Tuple[int, int], None, None]:
        sin_th = math.sin(theta)
        reach_i = max(1, int(math.ceil(angle / dtheta)))
        if sin_th < 1e-6:
            reach_k = K
        else:
            reach_k = max(1, int(math.ceil(angle / (max(1e-12, sin_th) * dphi))))
        for di in range(-reach_i, reach_i + 1):
            ii = i + di
            if 0 <= ii < J:
                for dk in range(-reach_k, reach_k + 1):
                    yield ii, (k + dk) % K

    def exp_map(c: np.ndarray, w: np.ndarray, R: float, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        wx, wy = float(w[0]), float(w[1])
        r = math.hypot(wx, wy)
        if r < 1e-12:
            return c.copy()
        alpha = r / R
        t = (wx * e1 + wy * e2)
        nrm = float(np.linalg.norm(t))
        if nrm < 1e-20:
            return c.copy()
        t = t / nrm
        return math.cos(alpha) * c + math.sin(alpha) * t

    def reorthonormalize_tangent(c: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        e1p = e1 - float(np.dot(e1, c)) * c
        n1 = float(np.linalg.norm(e1p))
        if n1 < 1e-20:
            # 退化时重建
            e1p, e2p, _ = orthonormal_basis(c)
            return e1p, e2p
        e1p = e1p / n1
        e2p = np.cross(c, e1p)
        n2 = float(np.linalg.norm(e2p))
        e2p = e2p / (n2 if n2 > 0 else 1.0)
        return e1p, e2p

    def overlap_between_configs(
        cA: np.ndarray, e1A: np.ndarray, e2A: np.ndarray, phiA: float,
        cB: np.ndarray, e1B: np.ndarray, e2B: np.ndarray, phiB: float
    ) -> bool:
        dotcc = float(np.clip(np.dot(cA, cB), -1.0, 1.0))
        ang = math.acos(dotcc)
        if ang > alpha_max:
            return False
        alpha, w = log_map(cA, cB, R_nm)
        if alpha < 1e-12:
            return True
        # A 的局部多边形
        V1 = (poly_base @ rot2(phiA).T)
        # 将 B 的有向基并入 A 的切平面
        u3 = math.cos(phiB) * e1B + math.sin(phiB) * e2B
        v3 = -math.sin(phiB) * e1B + math.cos(phiB) * e2B
        u2 = transport_tangent_vector(cB, cA, u3)
        v2 = transport_tangent_vector(cB, cA, v3)
        B2 = np.array(
            [[float(np.dot(u2, e1A)), float(np.dot(v2, e1A))],
             [float(np.dot(u2, e2A)), float(np.dot(v2, e2A))]],
            dtype=float
        )
        V2 = (poly_base @ B2)
        return polygons_overlap_SAT(w, V1, V2)

    def overlap_patch_with_candidate(p: AdsorptionPatch, cC: np.ndarray, e1C: np.ndarray, e2C: np.ndarray, phiC: float) -> bool:
        return overlap_between_configs(cC, e1C, e2C, phiC, p.c, p.e1, p.e2, p.phi)

    def update_grid_for_patch(idx: int):
        # 根据 patches[idx].c 更新其所在的 grid cell
        p = patches[idx]
        old = patch_bin[idx] if idx < len(patch_bin) else None
        ii, kk, th = bin_index(p.c)
        if old is not None:
            oi, ok = old
            if oi == ii and ok == kk:
                return
            try:
                grid[oi][ok].remove(idx)
            except ValueError:
                pass
            patch_bin[idx] = (ii, kk)
            grid[ii][kk].append(idx)
        else:
            patch_bin[idx] = (ii, kk)
            grid[ii][kk].append(idx)

    def attempt_local_relax(
        cC: np.ndarray, e1C: np.ndarray, e2C: np.ndarray, phiC: float,
        iC: int, kC: int, thetaC: float, rng: np.random.Generator
    ) -> bool:
        # 参数换算
        step_nm = float(relax_step_frac) * r_max
        max_arc_nm_per_particle = float(relax_max_arc_frac) * r_max
        beta = math.radians(float(relax_phi_step_deg))
        r_nbr_nm = float(relax_r_nbr_factor) * r_max
        angle_nbr = r_nbr_nm / R_nm

        # 初始邻域集合
        cand_ids = []
        for ii, kk in neighbor_bins_by_angle(iC, kC, thetaC, angle_nbr):
            cand_ids.extend(grid[ii][kk])
        cand_ids = list(set(cand_ids))  # 去重

        moved_budget = {idx: 0.0 for idx in cand_ids}

        def overlapping_set():
            out = []
            for idx in cand_ids:
                if overlap_patch_with_candidate(patches[idx], cC, e1C, e2C, phiC):
                    out.append(idx)
            return out

        for it in range(int(relax_max_iter)):
            O = overlapping_set()
            if not O:
                return True
            if relax_shuffle:
                O = list(rng.permutation(O))
            moved_any = False
            for qid in O:
                if moved_budget.get(qid, 0.0) >= max_arc_nm_per_particle:
                    continue
                q = patches[qid]

                # 候选动作：四个切向平移 + 两个面内转动
                actions = [
                    ("move", np.array([step_nm, 0.0])),
                    ("move", np.array([-step_nm, 0.0])),
                    ("move", np.array([0.0, step_nm])),
                    ("move", np.array([0.0, -step_nm])),
                    ("rot", +beta),
                    ("rot", -beta),
                ]

                accepted_this_q = False
                for act, val in actions:
                    # 备份
                    c_old, e1_old, e2_old, phi_old = q.c.copy(), q.e1.copy(), q.e2.copy(), float(q.phi)
                    if act == "move":
                        c_new = exp_map(q.c, val, R_nm, q.e1, q.e2)
                        # 并行运输基向量至新切平面
                        e1_new = transport_tangent_vector(q.c, c_new, q.e1)
                        e2_new = transport_tangent_vector(q.c, c_new, q.e2)
                        e1_new, e2_new = reorthonormalize_tangent(c_new, e1_new, e2_new)
                        q.c, q.e1, q.e2 = c_new, e1_new, e2_new
                        # 预算累加
                        alpha_mv = math.acos(float(np.clip(np.dot(c_old, c_new), -1.0, 1.0)))
                        moved_budget[qid] = moved_budget.get(qid, 0.0) + alpha_mv * R_nm
                    else:
                        q.phi = (q.phi + float(val)) % (2 * math.pi)

                    # 约束检查：与候选不重叠、且不与周围其他贴片重叠
                    ok_q = not overlap_patch_with_candidate(q, cC, e1C, e2C, phiC)
                    if ok_q:
                        # 仅检查近邻
                        ii, kk, th = bin_index(q.c)
                        check_ids = []
                        for iii, kkk in neighbor_bins_metric(ii, kk, th):
                            check_ids.extend(grid[iii][kkk])
                        ok_neighbors = True
                        for j in set(check_ids):
                            if j == qid:
                                continue
                            if overlap_between_configs(q.c, q.e1, q.e2, q.phi, patches[j].c, patches[j].e1, patches[j].e2, patches[j].phi):
                                ok_neighbors = False
                                break
                        if ok_neighbors:
                            # 接受该动作
                            accepted_this_q = True
                            moved_any = True
                            if act == "move":
                                update_grid_for_patch(qid)
                            break

                    # 回退
                    q.c, q.e1, q.e2, q.phi = c_old, e1_old, e2_old, phi_old
                    if act == "move":
                        pass

                # end for actions
                _ = accepted_this_q

            if not moved_any:
                return False
        # 迭代耗尽
        return False

    patches: List[AdsorptionPatch] = []
    attempts = accepted = streak = 0
    t0 = time.time()
    hist_attempts = []
    hist_N = []

    if progress and not progress_hook:
        print(f"Adsorption(poly) started... grid JxK={J}x{K}")
    if progress_hook:
        progress_hook(0, 0, 0)

    while attempts < max_attempts and streak < stagnation_limit:
        attempts += 1
        c = unit_random()
        e1, e2, _ = orthonormal_basis(c)
        phi = rng.uniform(0.0, 2 * math.pi)

        i, k, theta = bin_index(c)
        cand_ids = []
        for ii, kk in neighbor_bins_metric(i, k, theta):
            cand_ids.extend(grid[ii][kk])

        R2 = rot2(phi)
        V1 = (poly_base @ R2.T)

        ok = True
        for idx in cand_ids:
            p = patches[idx]
            dotcc = float(np.clip(np.dot(c, p.c), -1.0, 1.0))
            ang = math.acos(dotcc)
            if ang > alpha_max:
                continue
            alpha, w = log_map(c, p.c, R_nm)
            if alpha < 1e-12:
                ok = False
                break

            u3 = math.cos(p.phi) * p.e1 + math.sin(p.phi) * p.e2
            v3 = -math.sin(p.phi) * p.e1 + math.cos(p.phi) * p.e2
            u2 = transport_tangent_vector(p.c, c, u3)
            v2 = transport_tangent_vector(p.c, c, v3)
            B2 = np.array(
                [[float(np.dot(u2, e1)), float(np.dot(v2, e1))],
                 [float(np.dot(u2, e2)), float(np.dot(v2, e2))]],
                dtype=float
            )
            V2 = (poly_base @ B2)  # p.phi 已含在 (u2,v2)
            if polygons_overlap_SAT(w, V1, V2):
                ok = False
                break

        if ok:
            # 直接接受
            patches.append(AdsorptionPatch(c=c, phi=phi, e1=e1, e2=e2))
            grid[i][k].append(len(patches) - 1)
            patch_bin.append((i, k))
            accepted += 1
            streak = 0
            if return_history:
                hist_attempts.append(attempts)
                hist_N.append(accepted)
        else:
            # 触发局部让位
            if relax_enabled:
                did_relax = attempt_local_relax(c, e1, e2, phi, i, k, theta, rng)
            else:
                did_relax = False

            if did_relax:
                # 松弛成功，接受插入
                patches.append(AdsorptionPatch(c=c, phi=phi, e1=e1, e2=e2))
                grid[i][k].append(len(patches) - 1)
                patch_bin.append((i, k))
                accepted += 1
                streak = 0
                if return_history:
                    hist_attempts.append(attempts)
                    hist_N.append(accepted)
            else:
                streak += 1

        if progress and not progress_hook and (attempts % progress_every == 0):
            elapsed = time.time() - t0
            attps = attempts / max(1e-9, elapsed)
            print(f"[{int(elapsed):02d}s] attempts={attempts:,} | N={accepted:,} | streak={streak:,}/{stagnation_limit:,} | {int(attps)} att/s")
        if progress_hook:
            progress_hook(attempts, accepted, streak)

    if progress and not progress_hook:
        print(f"Adsorption(poly) finished in {int(time.time() - t0):02d}s | attempts={attempts:,} | N={accepted:,} | streak={streak:,}")
    if progress_hook:
        progress_hook(attempts, accepted, streak)

    if return_history:
        return patches, attempts, accepted, np.array(hist_attempts, int), np.array(hist_N, int)
    else:
        return patches, attempts, accepted


# ========= 指标（N、Γ）=========

def metrics_from_patches(patches: List[AdsorptionPatch], R_nm: float) -> Tuple[int, float]:
    N = len(patches)
    area_nm2 = 4.0 * math.pi * (R_nm ** 2)
    Gamma_um2 = (N / area_nm2) * 1e6
    return N, Gamma_um2


# ========= 快速覆盖率估算 =========

def build_grid(patches: List[AdsorptionPatch], R_nm: float, poly_base: np.ndarray):
    r_max = float(np.max(np.linalg.norm(poly_base, axis=1)))
    theta_poly = r_max / R_nm
    J = max(8, int(math.ceil(math.pi / (2 * theta_poly))))
    K = max(16, int(math.ceil(2 * math.pi / (2 * theta_poly))))
    dtheta = math.pi / J
    dphi = 2 * math.pi / K
    grid = [[[] for _ in range(K)] for __ in range(J)]
    for idx, p in enumerate(patches):
        theta = math.acos(np.clip(p.c[2], -1.0, 1.0))
        i = int(theta / dtheta)
        i = min(J - 1, max(0, i))
        phi = math.atan2(p.c[1], p.c[0])
        phi = phi if phi >= 0 else (phi + 2 * math.pi)
        k = int(phi / dphi)
        k = min(K - 1, max(0, k))
        grid[i][k].append(idx)
    return grid, (J, K, dtheta, dphi), theta_poly


def point_in_convex_polygon(q: np.ndarray, verts: np.ndarray) -> bool:
    e = np.roll(verts, -1, axis=0) - verts
    n = np.stack((-e[:, 1], e[:, 0]), axis=1)
    return np.all((q - verts) @ n.T <= 1e-12)


def estimate_adsorption_coverage(
    patches: List[AdsorptionPatch],
    R_nm: float,
    poly_base: np.ndarray,
    M: int = 50_000,
    seed: int = 1234,
    progress: bool = True,
    progress_hook: Optional[Callable[[int, int, float], None]] = None
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    grid, (J, K, dtheta, dphi), theta_poly = build_grid(patches, R_nm, poly_base)
    alpha_max = theta_poly
    covered = 0
    t0 = time.time()

    for m in range(1, M + 1):
        z = rng.uniform(-1.0, 1.0)
        t = rng.uniform(0.0, 2 * math.pi)
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        p = np.array([math.cos(t) * r_xy, math.sin(t) * r_xy, z])

        theta = math.acos(max(-1.0, min(1.0, p[2])))
        i = int(theta / dtheta)
        i = min(J - 1, max(0, i))
        phi = math.atan2(p[1], p[0])
        phi = phi if phi >= 0 else (phi + 2 * math.pi)
        k = int(phi / dphi)
        k = min(K - 1, max(0, k))

        sin_th = math.sin(theta)
        reach_i = max(1, int(math.ceil(alpha_max / dtheta)))
        reach_k = (K if sin_th < 1e-6 else max(1, int(math.ceil(alpha_max / (sin_th * dphi)))))

        hit = False
        for di in range(-reach_i, reach_i + 1):
            ii = i + di
            if 0 <= ii < J:
                for dk in range(-reach_k, reach_k + 1):
                    kk = (k + dk) % K
                    for idx in grid[ii][kk]:
                        patch = patches[idx]
                        ang = math.acos(float(np.clip(np.dot(p, patch.c), -1.0, 1.0)))
                        if ang > alpha_max:
                            continue
                        _, w = log_map(patch.c, p, R_nm)
                        Rloc = rot2(-patch.phi)
                        q = Rloc @ w
                        if point_in_convex_polygon(q, poly_base):
                            hit = True
                            break
                    if hit:
                        break
            if hit:
                break

        if hit:
            covered += 1
        if progress_hook:
            rate = covered / m
            progress_hook(m, covered, rate)
        elif progress and (m % max(1000, M // 20) == 0 or m == M):
            rate = covered / m
            spd = m / max(1e-9, time.time() - t0)
            eta = (M - m) / max(1e-9, spd)
            print(f"[{m:>7}/{M}] φ_A≈{rate:.5f} | {int(spd)} samp/s | ETA={int(eta // 60):02d}:{int(eta % 60):02d}")

    phiA = covered / float(M)
    err = math.sqrt(max(phiA * (1 - phiA), 1e-16) / float(M))
    if progress:
        print(f"MC(phi_A fast) done | φ_A={phiA:.5f} ± {err:.5f}")
    return phiA, err
