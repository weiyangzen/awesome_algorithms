"""Track reconstruction MVP in a toy 2D magnetic-field detector.

Pipeline:
1) Simulate curved charged-particle tracks and fake hits.
2) Build circle seeds from hit triplets.
3) Score seeds by layer-wise hit compatibility.
4) Refine each track with robust SciPy + PyTorch fitting.
5) Match reconstructed tracks to truth via Hungarian assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares, linear_sum_assignment
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class Config:
    seed: int = 428
    n_truth_tracks: int = 5
    layer_path_lengths_mm: tuple[float, ...] = (35.0, 65.0, 95.0, 125.0, 155.0, 185.0, 215.0)
    sigma_hit_mm: float = 0.25
    kappa_abs_range: tuple[float, float] = (0.004, 0.010)  # 1/mm

    fake_hits_per_layer: int = 3
    fake_radius_range_mm: tuple[float, float] = (30.0, 260.0)
    fake_sigma_mm: float = 0.8

    seed_layers: tuple[int, int, int] = (0, 3, 6)
    residual_cut_seed_mm: float = 0.50
    residual_cut_final_mm: float = 0.34
    min_hits_per_track: int = 5
    max_reco_tracks: int = 8

    radius_limits_mm: tuple[float, float] = (70.0, 320.0)
    beamspot_xy_mm: tuple[float, float] = (0.0, 0.0)
    beamspot_tolerance_mm: float = 2.8

    lsq_f_scale_mm: float = 0.25
    torch_steps: int = 120
    torch_lr: float = 0.06
    torch_delta_mm: float = 0.20

    match_cost_threshold_mm: float = 0.40


@dataclass(frozen=True)
class TruthTrack:
    track_id: int
    center: np.ndarray  # shape: (2,)
    radius: float
    ideal_hits: np.ndarray  # shape: (n_layers, 2)


@dataclass(frozen=True)
class Hit:
    hit_id: int
    layer: int
    x: float
    y: float
    truth_id: int | None


@dataclass(frozen=True)
class RecoTrack:
    reco_id: int
    center: np.ndarray
    radius: float
    hit_ids: list[int]
    mean_abs_residual_mm: float


def trajectory_points(
    vertex: np.ndarray,
    phi0: float,
    kappa: float,
    s_values: np.ndarray,
) -> np.ndarray:
    """Parametric 2D helix projection (circle) under uniform Bz."""
    x0, y0 = float(vertex[0]), float(vertex[1])
    x = x0 + (np.sin(phi0 + kappa * s_values) - np.sin(phi0)) / kappa
    y = y0 - (np.cos(phi0 + kappa * s_values) - np.cos(phi0)) / kappa
    return np.column_stack([x, y])


def simulate_event(cfg: Config) -> tuple[list[TruthTrack], list[Hit]]:
    """Create one event containing true tracks + fake hits."""
    rng = np.random.default_rng(cfg.seed)
    s_values = np.asarray(cfg.layer_path_lengths_mm, dtype=float)
    beamspot = np.asarray(cfg.beamspot_xy_mm, dtype=float)

    truth_tracks: list[TruthTrack] = []
    hits: list[Hit] = []
    hit_id = 0

    for tid in range(cfg.n_truth_tracks):
        phi0 = float(rng.uniform(-np.pi, np.pi))
        charge_sign = int(rng.choice([-1, 1]))
        kappa = charge_sign * float(rng.uniform(*cfg.kappa_abs_range))

        ideal_hits = trajectory_points(beamspot, phi0, kappa, s_values)
        center = np.array(
            [
                beamspot[0] - np.sin(phi0) / kappa,
                beamspot[1] + np.cos(phi0) / kappa,
            ],
            dtype=float,
        )
        radius = float(abs(1.0 / kappa))

        truth_tracks.append(
            TruthTrack(track_id=tid, center=center, radius=radius, ideal_hits=ideal_hits)
        )

        noise = rng.normal(0.0, cfg.sigma_hit_mm, size=ideal_hits.shape)
        measured_hits = ideal_hits + noise
        for layer, xy in enumerate(measured_hits):
            hits.append(
                Hit(
                    hit_id=hit_id,
                    layer=layer,
                    x=float(xy[0]),
                    y=float(xy[1]),
                    truth_id=tid,
                )
            )
            hit_id += 1

    for layer in range(len(cfg.layer_path_lengths_mm)):
        for _ in range(cfg.fake_hits_per_layer):
            r = float(rng.uniform(*cfg.fake_radius_range_mm))
            phi = float(rng.uniform(-np.pi, np.pi))
            x = r * np.cos(phi) + float(rng.normal(0.0, cfg.fake_sigma_mm))
            y = r * np.sin(phi) + float(rng.normal(0.0, cfg.fake_sigma_mm))
            hits.append(Hit(hit_id=hit_id, layer=layer, x=x, y=y, truth_id=None))
            hit_id += 1

    return truth_tracks, hits


def build_hit_indices(hits: list[Hit]) -> tuple[dict[int, Hit], dict[int, list[int]]]:
    hit_map = {h.hit_id: h for h in hits}
    by_layer: dict[int, list[int]] = {}
    for h in hits:
        by_layer.setdefault(h.layer, []).append(h.hit_id)
    return hit_map, by_layer


def circle_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Compute circle center/radius through 3 points; return None if degenerate."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-8:
        return None

    x1sq, x2sq, x3sq = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (x1sq * (y2 - y3) + x2sq * (y3 - y1) + x3sq * (y1 - y2)) / d
    uy = (x1sq * (x3 - x2) + x2sq * (x1 - x3) + x3sq * (x2 - x1)) / d
    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(p1 - center))
    return center, radius


def assign_hits_by_layer(
    hit_map: dict[int, Hit],
    by_layer: dict[int, list[int]],
    available_hit_ids: set[int],
    center: np.ndarray,
    radius: float,
    residual_cut_mm: float,
) -> tuple[list[int], np.ndarray]:
    """Pick at most one hit per layer by nearest radial residual."""
    selected_ids: list[int] = []
    selected_residuals: list[float] = []

    cx, cy = float(center[0]), float(center[1])
    for layer in sorted(by_layer.keys()):
        candidate_ids = [hid for hid in by_layer[layer] if hid in available_hit_ids]
        if not candidate_ids:
            continue

        pts = np.array([[hit_map[hid].x, hit_map[hid].y] for hid in candidate_ids], dtype=float)
        radial_residuals = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius)

        best_idx = int(np.argmin(radial_residuals))
        best_residual = float(radial_residuals[best_idx])
        if best_residual <= residual_cut_mm:
            selected_ids.append(candidate_ids[best_idx])
            selected_residuals.append(best_residual)

    return selected_ids, np.asarray(selected_residuals, dtype=float)


def beamspot_consistency(center: np.ndarray, radius: float, beamspot: np.ndarray) -> float:
    """Distance of beamspot to circle manifold; near-zero if circle passes beamspot."""
    return float(abs(np.linalg.norm(center - beamspot) - radius))


def refine_circle_scipy(
    points: np.ndarray,
    init_center: np.ndarray,
    init_radius: float,
    f_scale_mm: float,
) -> tuple[np.ndarray, float, float]:
    """Robust nonlinear least squares for circle parameters."""

    def residual_fn(params: np.ndarray) -> np.ndarray:
        cx, cy, rad = params
        rad_pos = abs(rad)
        return np.hypot(points[:, 0] - cx, points[:, 1] - cy) - rad_pos

    x0 = np.array([init_center[0], init_center[1], init_radius], dtype=float)
    result = least_squares(
        residual_fn,
        x0=x0,
        loss="soft_l1",
        f_scale=f_scale_mm,
        max_nfev=250,
    )
    cx, cy, rad = result.x
    center = np.array([cx, cy], dtype=float)
    radius = float(abs(rad))
    mean_abs_residual = float(np.mean(np.abs(residual_fn(result.x))))
    return center, radius, mean_abs_residual


def refine_circle_torch(
    points: np.ndarray,
    init_center: np.ndarray,
    init_radius: float,
    steps: int,
    lr: float,
    delta_mm: float,
) -> tuple[np.ndarray, float, float]:
    """Pseudo-Huber refinement with automatic differentiation."""
    pts = torch.tensor(points, dtype=torch.float64)
    params = torch.tensor(
        [float(init_center[0]), float(init_center[1]), float(init_radius)],
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        center = params[:2]
        radius = torch.abs(params[2]) + 1e-9
        residuals = torch.linalg.norm(pts - center, dim=1) - radius
        loss = torch.sum(delta_mm**2 * (torch.sqrt(1.0 + (residuals / delta_mm) ** 2) - 1.0))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        center = params[:2].detach().cpu().numpy()
        radius = float(torch.abs(params[2]).item())
        residuals = torch.linalg.norm(pts - params[:2], dim=1) - torch.abs(params[2])
        mean_abs_residual = float(torch.mean(torch.abs(residuals)).item())

    return center, radius, mean_abs_residual


def reconstruct_tracks(cfg: Config, hits: list[Hit]) -> list[RecoTrack]:
    hit_map, by_layer = build_hit_indices(hits)
    available_hit_ids: set[int] = set(hit_map.keys())
    beamspot = np.asarray(cfg.beamspot_xy_mm, dtype=float)

    reco_tracks: list[RecoTrack] = []
    seed_l0, seed_l1, seed_l2 = cfg.seed_layers

    for reco_id in range(cfg.max_reco_tracks):
        ids0 = [hid for hid in by_layer.get(seed_l0, []) if hid in available_hit_ids]
        ids1 = [hid for hid in by_layer.get(seed_l1, []) if hid in available_hit_ids]
        ids2 = [hid for hid in by_layer.get(seed_l2, []) if hid in available_hit_ids]

        if not ids0 or not ids1 or not ids2:
            break

        best_candidate: dict[str, object] | None = None

        for h0, h1, h2 in product(ids0, ids1, ids2):
            p1 = np.array([hit_map[h0].x, hit_map[h0].y], dtype=float)
            p2 = np.array([hit_map[h1].x, hit_map[h1].y], dtype=float)
            p3 = np.array([hit_map[h2].x, hit_map[h2].y], dtype=float)

            circle = circle_from_three_points(p1, p2, p3)
            if circle is None:
                continue
            center, radius = circle

            if not (cfg.radius_limits_mm[0] <= radius <= cfg.radius_limits_mm[1]):
                continue

            beam_res = beamspot_consistency(center, radius, beamspot)
            if beam_res > cfg.beamspot_tolerance_mm:
                continue

            candidate_ids, residuals = assign_hits_by_layer(
                hit_map=hit_map,
                by_layer=by_layer,
                available_hit_ids=available_hit_ids,
                center=center,
                radius=radius,
                residual_cut_mm=cfg.residual_cut_seed_mm,
            )

            if len(candidate_ids) < cfg.min_hits_per_track:
                continue

            score = (len(candidate_ids), -float(np.mean(residuals)))
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "score": score,
                    "center": center,
                    "radius": radius,
                    "hit_ids": candidate_ids,
                }

        if best_candidate is None:
            break

        seed_hit_ids = list(best_candidate["hit_ids"])
        seed_points = np.array([[hit_map[hid].x, hit_map[hid].y] for hid in seed_hit_ids], dtype=float)

        center_lsq, radius_lsq, _ = refine_circle_scipy(
            points=seed_points,
            init_center=np.asarray(best_candidate["center"], dtype=float),
            init_radius=float(best_candidate["radius"]),
            f_scale_mm=cfg.lsq_f_scale_mm,
        )
        center_torch, radius_torch, _ = refine_circle_torch(
            points=seed_points,
            init_center=center_lsq,
            init_radius=radius_lsq,
            steps=cfg.torch_steps,
            lr=cfg.torch_lr,
            delta_mm=cfg.torch_delta_mm,
        )

        final_hit_ids, final_residuals = assign_hits_by_layer(
            hit_map=hit_map,
            by_layer=by_layer,
            available_hit_ids=available_hit_ids,
            center=center_torch,
            radius=radius_torch,
            residual_cut_mm=cfg.residual_cut_final_mm,
        )

        if len(final_hit_ids) < cfg.min_hits_per_track:
            final_hit_ids = seed_hit_ids
            final_points = seed_points
            center_final, radius_final, mean_abs = refine_circle_scipy(
                points=final_points,
                init_center=center_torch,
                init_radius=radius_torch,
                f_scale_mm=cfg.lsq_f_scale_mm,
            )
        else:
            final_points = np.array([[hit_map[hid].x, hit_map[hid].y] for hid in final_hit_ids], dtype=float)
            center_final, radius_final, _ = refine_circle_scipy(
                points=final_points,
                init_center=center_torch,
                init_radius=radius_torch,
                f_scale_mm=cfg.lsq_f_scale_mm,
            )
            center_final, radius_final, mean_abs = refine_circle_torch(
                points=final_points,
                init_center=center_final,
                init_radius=radius_final,
                steps=max(40, cfg.torch_steps // 2),
                lr=cfg.torch_lr * 0.7,
                delta_mm=cfg.torch_delta_mm,
            )

        beam_res_final = beamspot_consistency(center_final, radius_final, beamspot)
        if beam_res_final > cfg.beamspot_tolerance_mm * 1.5:
            break

        for hid in final_hit_ids:
            available_hit_ids.discard(hid)

        reco_tracks.append(
            RecoTrack(
                reco_id=reco_id,
                center=center_final,
                radius=float(radius_final),
                hit_ids=final_hit_ids,
                mean_abs_residual_mm=float(mean_abs),
            )
        )

    return reco_tracks


def match_reco_to_truth(
    reco_tracks: list[RecoTrack],
    truth_tracks: list[TruthTrack],
    threshold_mm: float,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Match reconstructed circles to truth circles by Hungarian assignment."""
    n_reco = len(reco_tracks)
    n_truth = len(truth_tracks)
    if n_reco == 0 or n_truth == 0:
        return [], np.empty((n_reco, n_truth), dtype=float)

    cost = np.zeros((n_reco, n_truth), dtype=float)
    for i, reco in enumerate(reco_tracks):
        for j, truth in enumerate(truth_tracks):
            residuals = np.abs(
                np.hypot(
                    truth.ideal_hits[:, 0] - reco.center[0],
                    truth.ideal_hits[:, 1] - reco.center[1],
                )
                - reco.radius
            )
            cost[i, j] = float(np.sqrt(np.mean(residuals**2)))

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= threshold_mm:
            matches.append((int(r), int(c), float(cost[r, c])))

    return matches, cost


def hit_purity(hit_ids: list[int], hit_map: dict[int, Hit]) -> tuple[float, int | None]:
    true_ids = [hit_map[hid].truth_id for hid in hit_ids if hit_map[hid].truth_id is not None]
    if not true_ids:
        return 0.0, None

    values, counts = np.unique(np.asarray(true_ids, dtype=int), return_counts=True)
    idx = int(np.argmax(counts))
    dominant_id = int(values[idx])
    purity = float(counts[idx] / max(1, len(hit_ids)))
    return purity, dominant_id


def evaluate(
    cfg: Config,
    truth_tracks: list[TruthTrack],
    reco_tracks: list[RecoTrack],
    hits: list[Hit],
) -> tuple[pd.DataFrame, dict[str, float]]:
    hit_map, _ = build_hit_indices(hits)
    matches, _ = match_reco_to_truth(reco_tracks, truth_tracks, cfg.match_cost_threshold_mm)

    match_by_reco: dict[int, tuple[int, float]] = {reco_i: (truth_i, cost) for reco_i, truth_i, cost in matches}

    detail_rows: list[dict[str, float | int | str]] = []
    matched_truth_ids: list[int] = []
    reco_radii: list[float] = []
    truth_radii: list[float] = []
    reco_cx: list[float] = []
    truth_cx: list[float] = []
    reco_cy: list[float] = []
    truth_cy: list[float] = []

    total_assigned_hits = 0
    total_true_assigned_hits = 0

    for reco in reco_tracks:
        purity, dominant_truth = hit_purity(reco.hit_ids, hit_map)
        total_assigned_hits += len(reco.hit_ids)
        total_true_assigned_hits += sum(
            1 for hid in reco.hit_ids if hit_map[hid].truth_id is not None
        )

        matched_truth = -1
        match_cost = np.nan
        if reco.reco_id in match_by_reco:
            matched_truth, match_cost = match_by_reco[reco.reco_id]
            matched_truth_ids.append(matched_truth)

            truth = truth_tracks[matched_truth]
            reco_radii.append(reco.radius)
            truth_radii.append(truth.radius)
            reco_cx.append(float(reco.center[0]))
            truth_cx.append(float(truth.center[0]))
            reco_cy.append(float(reco.center[1]))
            truth_cy.append(float(truth.center[1]))

        detail_rows.append(
            {
                "reco_id": reco.reco_id,
                "n_hits": len(reco.hit_ids),
                "mean_abs_residual_mm": reco.mean_abs_residual_mm,
                "center_x_mm": float(reco.center[0]),
                "center_y_mm": float(reco.center[1]),
                "radius_mm": reco.radius,
                "hit_purity": purity,
                "dominant_truth": dominant_truth if dominant_truth is not None else -1,
                "matched_truth": matched_truth,
                "match_cost_mm": match_cost,
            }
        )

    details = pd.DataFrame(detail_rows)

    n_truth = len(truth_tracks)
    n_reco = len(reco_tracks)
    n_matched = len(set(matched_truth_ids))

    summary = {
        "truth_tracks": float(n_truth),
        "reco_tracks": float(n_reco),
        "matched_tracks": float(n_matched),
        "tracking_efficiency": n_matched / max(1, n_truth),
        "fake_track_rate": (n_reco - len(matches)) / max(1, n_reco),
        "assigned_hit_purity": total_true_assigned_hits / max(1, total_assigned_hits),
        "radius_rmse_mm": (
            float(np.sqrt(mean_squared_error(truth_radii, reco_radii)))
            if truth_radii
            else float("nan")
        ),
        "center_x_rmse_mm": (
            float(np.sqrt(mean_squared_error(truth_cx, reco_cx)))
            if truth_cx
            else float("nan")
        ),
        "center_y_rmse_mm": (
            float(np.sqrt(mean_squared_error(truth_cy, reco_cy)))
            if truth_cy
            else float("nan")
        ),
    }
    return details, summary


def main() -> None:
    cfg = Config()
    truth_tracks, hits = simulate_event(cfg)
    reco_tracks = reconstruct_tracks(cfg, hits)
    details, summary = evaluate(cfg, truth_tracks, reco_tracks, hits)

    print("=== Track Reconstruction MVP (2D Toy Detector) ===")
    print(f"truth_tracks={int(summary['truth_tracks'])}")
    print(f"reco_tracks={int(summary['reco_tracks'])}")
    print(f"matched_tracks={int(summary['matched_tracks'])}")
    print(f"tracking_efficiency={summary['tracking_efficiency']:.3f}")
    print(f"fake_track_rate={summary['fake_track_rate']:.3f}")
    print(f"assigned_hit_purity={summary['assigned_hit_purity']:.3f}")
    print(f"radius_rmse_mm={summary['radius_rmse_mm']:.3f}")
    print(f"center_x_rmse_mm={summary['center_x_rmse_mm']:.3f}")
    print(f"center_y_rmse_mm={summary['center_y_rmse_mm']:.3f}")

    print("\nReconstructed track table:")
    if details.empty:
        print("<no reconstructed tracks>")
    else:
        show_cols = [
            "reco_id",
            "n_hits",
            "mean_abs_residual_mm",
            "radius_mm",
            "hit_purity",
            "matched_truth",
            "match_cost_mm",
        ]
        print(details[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
