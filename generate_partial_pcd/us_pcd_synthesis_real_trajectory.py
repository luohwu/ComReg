# -*- coding: utf-8 -*-
"""
Ultrasound bone-surface point-cloud synthesis (manuscript ray-casting model,
driven by REAL probe trajectories).

Goal
----
Produce simulated US bone-surface point clouds that are as close as possible to
the real intraoperative US reconstructions, so that a model trained on the
simulated data (see main_100k_SC_slice_wise_with_reg_joint_training.py) validates
well on the real data. The single most effective way to shrink the sim-to-real
gap is to reuse the *real* acquisition geometry, so this simulator replays each
record's real probe poses (tracking.csv) against the patient CT mesh.

Method (follows the manuscript)
-------------------------------
The manuscript emulates the US sweep by "casting rays from a plane along
anatomically plausible trajectories". Those anatomically plausible trajectories
are obtained MANUALLY, i.e. they are the real, manually-acquired probe sweeps
recorded in each record's tracking.csv. This script therefore IS the manuscript's
ray-casting model; replaying the recorded poses is simply how we instantiate the
"anatomically plausible trajectories", not a departure from the manuscript.

For every tracked B-mode frame we cast a row of rays from the transducer line
(one ray per image column) along the depth direction, and keep ONLY the first
intersection with the cortical surface -> acoustic shadowing. The transform chain
   T = T_tracking @ T_calibration @ T_scale
maps image-plane pixels into the CT/world frame (identical convention to the
project's hand-trajectory generator), so the simulated points land in the same
frame as the CT mesh and the real US clouds.

Global sweep direction (multiple records -> multiple directions)
----------------------------------------------------------------
Each ultrasound RECORD under an anatomy is one manually-acquired probe sweep with
its OWN (anatomically plausible) trajectory, i.e. ONE global sweep direction. An
anatomy typically has several records, so replaying all of them yields several
distinct global sweep directions. This is how we realize the manuscript's
"variation in global sweep direction": it comes from the multiple manually-acquired
sweeps. The per-frame perturbations below (probe tilt, non-uniform sweep speed,
dropped slices) then add intra-sweep variation ON TOP of each record's global
direction.

Implementation notes (consistent with the manuscript)
------------------------------------------------------
* The "anatomically plausible trajectories" are the real, manually-acquired probe
  sweeps (tracking.csv), not a procedurally-generated synthetic path. This is the
  intended instantiation of the manuscript's ray-casting model. For a NEW dataset
  that has no recorded sweeps, such anatomically plausible trajectories can instead
  be produced by a simulated manual scan (e.g. a hand-authored / programmatically
  generated probe path over the bone), then fed in here exactly like a tracking.csv.
* No segmentation noise (requested).
* An incidence-angle cutoff is kept ON (default 70 deg). Real US cannot image
  cortical surfaces at grazing incidence (specular dropout), so the real clouds
  are missing those points; dropping grazing hits makes the simulation match.
  Set incidence_angle_threshold_deg=None to disable it.

Output
------
Writes <specimen>/CT_bone_segmentations/<anatomy>_simulated_slice_data_realtraj.npy
(distinct from the hand-trajectory generator's *_simulated_slice_data.npy so both
can coexist for A/B comparison). The dataset loads it when its conf sets
`simulated_data_suffix = "_simulated_slice_data_realtraj.npy"`. Keys consumed by
Dataset.dataset_coupled.DatasetCoupled_slice_wise:
    pcd_simulated_list : list of (Ni,3) float arrays, one per record (CT frame, mm)
    pcd_slice_list     : per-frame visible points (object array), for inspection
    sweep_length       : mean number of frames per record
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from utility.converter import vectorToMatrix
from utility.us_denoise import denoise_dbscan_cloud


# ---------------------------------------------------------------------------
# Real ultrasound probe calibration (UltraBones100k), shared across specimens.
# ---------------------------------------------------------------------------
CALIBRATION_T = [26.44694442, -0.52572229, 128.00100047]      # mm
CALIBRATION_EULER = [92.48865621, -0.46874914, 179.2277322]   # xyz-euler, deg
SCALE_X = 0.05392  # mm / pixel
SCALE_Y = 0.05392  # mm / pixel
IMAGE_WIDTH = 950           # number of image columns (rays per frame)
IMAGE_HEIGHT_PIX = 811      # number of image rows -> imaging depth
MAX_HIT_DIST = IMAGE_HEIGHT_PIX * SCALE_Y   # imaging depth in mm


def build_calibration() -> Tuple[np.ndarray, np.ndarray]:
    """Return (T_calibration, T_scale) for the real probe."""
    T_calibration = vectorToMatrix(CALIBRATION_T, CALIBRATION_EULER)
    T_scale = np.eye(4)
    T_scale[0, 0] = SCALE_X
    T_scale[1, 1] = SCALE_Y
    return T_calibration, T_scale


# ---------------------------------------------------------------------------
# Stochastic-perturbation helpers (manuscript model).
# ---------------------------------------------------------------------------
def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (float(np.linalg.norm(v)) + eps)


def _axis_angle_R(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix (3x3) about `axis` by `angle_rad`."""
    x, y, z = _safe_normalize(axis)
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _smooth_random_walk_degrees(n, max_abs_deg, step_std_deg, smooth_window, rng):
    """Bounded, smoothed random-walk angle sequence in [-max_abs_deg, max_abs_deg]
    (manuscript probe-tilt model: random walk -> moving-average -> scale+clip)."""
    if n <= 0 or max_abs_deg <= 0.0:
        return np.zeros(max(n, 0), dtype=np.float64)
    walk = np.cumsum(rng.normal(0.0, step_std_deg, size=n))
    w = int(max(3, smooth_window)) | 1  # force odd
    smooth = np.convolve(walk, np.ones(w) / w, mode="same")
    m = float(np.max(np.abs(smooth))) + 1e-9
    return np.clip(smooth / m * max_abs_deg, -max_abs_deg, max_abs_deg)


def _nonuniform_frame_indices(n, log_sigma, smooth_window, fmin, fmax, rng):
    """Frame indices with lognormal-distributed spacing (variable sweep speed):
    clusters of densely-sampled frames separated by sparse gaps."""
    if n <= 1:
        return np.arange(max(n, 0))
    noise = rng.normal(0.0, 1.0, size=n - 1)
    w = int(max(3, smooth_window)) | 1
    smooth = np.convolve(noise, np.ones(w) / w, mode="same")
    steps = np.clip(np.exp(float(log_sigma) * smooth), fmin, fmax)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    s = s / (s[-1] + 1e-12) * (n - 1)
    return np.unique(np.clip(np.rint(s).astype(np.int64), 0, n - 1))


# ---------------------------------------------------------------------------
# Core: ray-cast one B-mode frame, keep first cortical hit per column.
# ---------------------------------------------------------------------------
def _raycast_frame(
    scene: o3d.t.geometry.RaycastingScene,
    tri_normals: np.ndarray,
    T: np.ndarray,
    width: int,
    max_hit_dist: float,
    incidence_thr_rad: Optional[float],
    ax_deg: float = 0.0,
    ay_deg: float = 0.0,
) -> np.ndarray:
    """First-intersection (shadowed) surface points for a single frame.

    Rays originate along the transducer line (local x = column, local y = 0) and
    travel into the tissue (local +y). `ax_deg`/`ay_deg` apply a probe tilt about
    the in-plane axes (rotation about the transducer-line midpoint), modelling
    per-frame wobble. Returns (Ni, 3) points in the CT frame.
    """
    # ray origins: one per column, at the transducer surface (local y = 0)
    ray_positions = np.stack(
        [
            np.arange(width, dtype=np.float64) + 0.5,
            np.zeros(width, dtype=np.float64),
            np.zeros(width, dtype=np.float64),
            np.ones(width, dtype=np.float64),
        ],
        axis=1,
    )
    ray_dir_local = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # depth

    origins = (T @ ray_positions.T).T[:, :3]
    direction = (T @ ray_dir_local)[:3]
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-12:
        return np.empty((0, 3), dtype=np.float32)
    direction = direction / nrm

    # probe-tilt wobble: rotate the rays about the transducer-line midpoint
    if ax_deg != 0.0 or ay_deg != 0.0:
        x_w = _safe_normalize(T[:3, 0])   # lateral
        y_w = _safe_normalize(T[:3, 1])   # depth
        R = _axis_angle_R(x_w, np.deg2rad(ax_deg)) @ _axis_angle_R(y_w, np.deg2rad(ay_deg))
        p0 = origins.mean(axis=0)
        origins = p0 + (R @ (origins - p0).T).T
        direction = R @ direction

    origins = origins.astype(np.float32)
    direction = direction.astype(np.float32)
    directions = np.repeat(direction[None, :], width, axis=0).astype(np.float32)

    rays = o3d.core.Tensor(np.concatenate([origins, directions], axis=1),
                           dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    prim_key = "primitive_ids" if "primitive_ids" in ans else "primitive_id"
    prim_ids = ans[prim_key].numpy()

    # first intersection within imaging depth = visible cortical surface
    mask = np.isfinite(t_hit) & (t_hit > 0) & (t_hit <= max_hit_dist) & (prim_ids >= 0)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    hit_pts = origins[mask] + directions[mask] * t_hit[mask, None].astype(np.float32)

    # specular dropout: discard near-grazing hits (US cannot image them)
    if incidence_thr_rad is not None and hit_pts.shape[0] > 0:
        normals = tri_normals[prim_ids[mask].astype(np.int64)]
        cosang = np.clip(np.abs(np.sum(directions[mask] * normals, axis=1)), -1.0, 1.0)
        ang = np.arccos(cosang)
        hit_pts = hit_pts[ang <= incidence_thr_rad]

    return hit_pts


def simulate_us_pcd(
    mesh_o3d: o3d.geometry.TriangleMesh,
    tracking_df: pd.DataFrame,
    T_calibration: np.ndarray,
    T_scale: np.ndarray,
    *,
    width: int = IMAGE_WIDTH,
    max_hit_dist: float = MAX_HIT_DIST,
    incidence_angle_threshold_deg: Optional[float] = 70.0,
    downsample_voxel: Optional[float] = 0.5,
    # --- manuscript stochastic perturbations (one random draw from p_S) ---------
    # probe-tilt wobble: bounded smoothed random walk about the in-plane axes
    rot_x_range_deg: float = 20.0,
    rot_y_range_deg: float = 5.0,
    rot_step_std_deg: float = 0.8,
    rot_smooth_window: int = 25,
    # non-uniform sweep speed: lognormal-spaced frame subset (clusters + gaps)
    nonuniform_speed: bool = True,
    spacing_log_sigma: float = 1.8,
    spacing_smooth_window: int = 61,
    spacing_min_step_factor: float = 0.03,
    spacing_max_step_factor: float = 10.0,
    # randomly dropped slices (and neighbors)
    drop_prob: float = 0.02,
    drop_neighbor_k: int = 3,
    seed: Optional[int] = None,
) -> Tuple[o3d.geometry.PointCloud, List[np.ndarray]]:
    """Draw ONE stochastic US point cloud by replaying ONE record's real probe
    trajectory -- a single global sweep direction -- with the manuscript's
    per-frame perturbations layered on top. Set the rotation ranges / drop_prob to
    0 and nonuniform_speed=False to recover the deterministic replay.

    The global sweep direction is fixed by the input `tracking_df` (this record);
    different global directions come from calling this on different records, not
    from this function.

    Returns (merged pcd in CT frame, per-kept-frame point list).
    """
    if mesh_o3d.is_empty() or len(tracking_df) == 0:
        return o3d.geometry.PointCloud(), []

    mesh = o3d.geometry.TriangleMesh(mesh_o3d)
    mesh.compute_triangle_normals()
    tri_normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    thr_rad = (None if incidence_angle_threshold_deg is None
               else float(incidence_angle_threshold_deg) * np.pi / 180.0)

    rng = np.random.default_rng(seed)
    rows = tracking_df.reset_index(drop=True)
    n_all = len(rows)

    # variable sweep speed -> non-uniform subset of the real frames
    if nonuniform_speed:
        sel = _nonuniform_frame_indices(n_all, spacing_log_sigma, spacing_smooth_window,
                                        spacing_min_step_factor, spacing_max_step_factor, rng)
    else:
        sel = np.arange(n_all)
    n_sel = len(sel)

    # smoothed random-walk probe tilts over the selected frames
    ax = _smooth_random_walk_degrees(n_sel, rot_x_range_deg, rot_step_std_deg, rot_smooth_window, rng)
    ay = _smooth_random_walk_degrees(n_sel, rot_y_range_deg, rot_step_std_deg, rot_smooth_window, rng)

    # dropped slices (+ neighbors)
    drop = np.zeros(n_sel, dtype=bool)
    if drop_prob > 0.0 and n_sel > 0:
        triggers = rng.random(n_sel) < float(drop_prob)
        k = int(max(0, drop_neighbor_k))
        if k == 0:
            drop |= triggers
        else:
            for idx in np.nonzero(triggers)[0]:
                drop[max(0, idx - k):min(n_sel - 1, idx + k) + 1] = True

    slices: List[np.ndarray] = []
    for j, fi in enumerate(sel):
        if drop[j]:
            continue
        row = rows.iloc[int(fi)]
        T_tracking = vectorToMatrix([row["x"], row["y"], row["z"]],
                                    [row["euler_x"], row["euler_y"], row["euler_z"]])
        T = T_tracking @ T_calibration @ T_scale
        pts = _raycast_frame(scene, tri_normals, T, width, max_hit_dist, thr_rad,
                             ax_deg=float(ax[j]), ay_deg=float(ay[j]))
        if pts.shape[0] > 0:
            slices.append(pts)

    pcd = o3d.geometry.PointCloud()
    if slices:
        pcd.points = o3d.utility.Vector3dVector(np.vstack(slices).astype(np.float64))
        if downsample_voxel is not None and downsample_voxel > 0:
            pcd = pcd.voxel_down_sample(float(downsample_voxel))
    return pcd, slices


# ---------------------------------------------------------------------------
# Batch generation over the dataset -> writes the consumed .npy files.
# ---------------------------------------------------------------------------
def generate_specimen_anatomy(
    specimen_id: int,
    anatomy: str,
    dataset_root_folder: str,
    T_calibration: np.ndarray,
    T_scale: np.ndarray,
    *,
    anatomy_idx: int = 0,
    n_variants: int = 10,
    incidence_angle_threshold_deg: Optional[float] = 70.0,
    downsample_voxel: Optional[float] = 0.5,
    nonuniform_speed: bool = True,
    drop_prob: float = 0.02,
    min_points: int = 16384,   # must be >= conf.dataset.npoints_input (batch collation needs uniform size)
    output_suffix: str = "_simulated_slice_data_realtraj.npy",
    visualize: bool = False,
) -> Optional[str]:
    """Generate the simulated US .npy for ONE (specimen, anatomy).

    Each record under the anatomy is one real probe sweep, i.e. ONE global sweep
    direction; an anatomy usually has several records, so the saved set spans
    MULTIPLE global sweep directions. For each record we draw `n_variants`
    stochastic clouds (each a fresh draw from p_S: random tilts + non-uniform
    sweep speed + dropped slices layered on that record's real trajectory) and
    pool them all into <anatomy><output_suffix>. Returns the written file path,
    or None if the inputs are missing / no valid cloud was produced.
    """
    specimen_folder = os.path.join(dataset_root_folder, f"specimen{specimen_id:02d}")
    ultrasound_records_folder = os.path.join(specimen_folder, "ultrasound_records")
    CT_segmentation_folder = os.path.join(specimen_folder, "CT_bone_segmentations")
    CT_segmentation_file = os.path.join(CT_segmentation_folder, f"{anatomy}.stl")
    anatomy_folder = os.path.join(ultrasound_records_folder, anatomy)
    if not (os.path.isfile(CT_segmentation_file) and os.path.isdir(anatomy_folder)):
        print(f"[data] cannot generate {specimen_id:02d}/{anatomy}: missing "
              f"{CT_segmentation_file} or {anatomy_folder}")
        return None
    CT_mesh = o3d.io.read_triangle_mesh(CT_segmentation_file)

    pcd_simulated_list: List[np.ndarray] = []   # n_variants per record (consumed)
    pcd_slice_list_all: List[np.ndarray] = []    # first-variant slices (inspection)
    sweep_lengths: List[int] = []
    cds: List[float] = []
    cds_denoised: List[float] = []
    n_skipped = 0

    # Each record = one real probe sweep = one GLOBAL SWEEP DIRECTION. Iterating
    # over every record of this anatomy therefore samples MULTIPLE global sweep
    # directions (the manuscript's "variation in global sweep direction"); the
    # per-record n_variants loop below only adds intra-sweep perturbations on top
    # of each fixed global direction.
    for rec_idx, record_id_name in enumerate(sorted(os.listdir(anatomy_folder))):
        record_folder = os.path.join(anatomy_folder, record_id_name)
        tracking_file = os.path.join(record_folder, "tracking.csv")
        if not (os.path.isdir(record_folder) and os.path.isfile(tracking_file)):
            continue
        df = pd.read_csv(tracking_file)
        sweep_lengths.append(len(df))

        us_file = os.path.join(record_folder, "3D_reconstructions",
                               "with_pred_labels", "reconstruction_pcd_filtered.xyz")
        us_pcd = o3d.io.read_point_cloud(us_file) if os.path.isfile(us_file) else None
        # Denoise the real cloud ONCE per record (it is identical across variants);
        # each simulated variant is then compared against it at the SAME 1mm density
        # (downsampling alone raises CD, so mixing full-res raw with downsampled
        # denoised would be misleading). Denoising removes the clustered segmentation
        # false-positives the simulator (correctly) does not model.
        us_ds = us_den = None
        if us_pcd is not None and len(us_pcd.points) > 0:
            us_ds = us_pcd.voxel_down_sample(1.0)
            us_den = denoise_dbscan_cloud(us_ds)

        rec_cds: List[float] = []
        rec_cds_den: List[float] = []
        for k in range(n_variants):
            # deterministic, distinct seed per (specimen, anatomy, record, variant)
            seed = ((specimen_id * 2 + anatomy_idx) * 100000
                    + rec_idx * 1000 + k)
            pcd_sim, slices = simulate_us_pcd(
                CT_mesh, df, T_calibration, T_scale,
                incidence_angle_threshold_deg=incidence_angle_threshold_deg,
                downsample_voxel=downsample_voxel,
                nonuniform_speed=nonuniform_speed, drop_prob=drop_prob,
                seed=seed,
            )
            # batch collation needs every cloud >= npoints_input; skip thin draws
            if len(pcd_sim.points) < min_points:
                n_skipped += 1
                continue
            pcd_simulated_list.append(np.asarray(pcd_sim.points, dtype=np.float32))
            if k == 0:
                pcd_slice_list_all.extend(slices)

            # sim-to-real CD for EVERY kept variant (raw + denoised), so the
            # reported means reflect overall simulated-cloud quality, not just k=0.
            if us_ds is not None and len(pcd_sim.points) > 0:
                cd = 0.5 * (np.asarray(us_ds.compute_point_cloud_distance(pcd_sim)).mean()
                            + np.asarray(pcd_sim.compute_point_cloud_distance(us_ds)).mean())
                cd_den = 0.5 * (np.asarray(us_den.compute_point_cloud_distance(pcd_sim)).mean()
                                + np.asarray(pcd_sim.compute_point_cloud_distance(us_den)).mean())
                cds.append(cd); cds_denoised.append(cd_den)
                rec_cds.append(cd); rec_cds_den.append(cd_den)
            if k == 0 and visualize and us_pcd is not None:
                o3d.visualization.draw_geometries([
                    us_pcd.paint_uniform_color((0, 1, 0)),
                    pcd_sim.paint_uniform_color((1, 0, 0))])

        if rec_cds:
            print(f"{record_folder}: mean CD={np.mean(rec_cds):.3f} mm | "
                  f"mean denoised CD={np.mean(rec_cds_den):.3f} mm  ({len(rec_cds)} variants)")

    if not pcd_simulated_list:
        print(f"[data] no valid simulated clouds for {specimen_id:02d}/{anatomy}")
        return None
    result_file = CT_segmentation_file.replace(".stl", output_suffix)
    np.save(result_file, {
        "sweep_length": int(np.mean(sweep_lengths)) if sweep_lengths else 0,
        "pcd_slice_list": np.array(pcd_slice_list_all, dtype=object),
        "pcd_simulated_list": pcd_simulated_list,
    }, allow_pickle=True)
    mean_cd = float(np.mean(cds)) if cds else float("nan")
    mean_cd_den = float(np.mean(cds_denoised)) if cds_denoised else float("nan")
    msg = (f"saved {result_file}: {len(pcd_simulated_list)} clouds "
           f"(<= {n_variants}/record), mean CD={mean_cd:.3f} mm | "
           f"mean denoised CD={mean_cd_den:.3f} mm")
    if n_skipped:
        msg += f"  [skipped {n_skipped} thin draws < {min_points} pts]"
    print(msg)
    return result_file


# Standard anatomy ordering -> anatomy_idx, so on-demand generation reproduces the
# same per-record seeds as the batch run (seeds depend on anatomy_idx).
ANATOMY_INDEX = {"fibula": 0, "tibia": 1}


def ensure_simulated_data(
    specimen_id: int,
    anatomy: str,
    dataset_root_folder: Optional[str] = None,
    output_suffix: str = "_simulated_slice_data_realtraj.npy",
    **kwargs,
) -> Optional[str]:
    """Return the simulated-US .npy path for (specimen, anatomy), generating it
    on demand iff it does not already exist. Used by the dataloader so training /
    testing never requires running this script manually."""
    if dataset_root_folder is None:
        dataset_root_folder = os.environ.get("ULTRABONES100K_ROOT", "/path/to/UltraBones100k")
    ct_file = os.path.join(dataset_root_folder, f"specimen{specimen_id:02d}",
                           "CT_bone_segmentations", f"{anatomy}.stl")
    out_file = ct_file.replace(".stl", output_suffix)
    if os.path.isfile(out_file):
        return out_file
    print(f"generating simulated US data for specimen{specimen_id:02d}/{anatomy} "
          f"(missing {os.path.basename(out_file)})...")
    T_calibration, T_scale = build_calibration()
    return generate_specimen_anatomy(
        specimen_id, anatomy, dataset_root_folder, T_calibration, T_scale,
        anatomy_idx=ANATOMY_INDEX.get(anatomy, 0), output_suffix=output_suffix, **kwargs)


def main_100k(
    dataset_root_folder: Optional[str] = None,
    anatomies: Tuple[str, ...] = ("fibula", "tibia"),
    output_suffix: str = "_simulated_slice_data_realtraj.npy",
    **kwargs,
):
    """Batch-generate the simulated .npy for every specimen x anatomy."""
    if dataset_root_folder is None:
        dataset_root_folder = os.environ.get("ULTRABONES100K_ROOT", "/path/to/UltraBones100k")
    T_calibration, T_scale = build_calibration()
    for specimen_id in range(1, 15):
        specimen_folder = os.path.join(dataset_root_folder, f"specimen{specimen_id:02d}")
        if not os.path.isdir(specimen_folder):
            continue
        for anatomy_idx, anatomy in enumerate(anatomies):
            generate_specimen_anatomy(
                specimen_id, anatomy, dataset_root_folder, T_calibration, T_scale,
                anatomy_idx=anatomy_idx, output_suffix=output_suffix, **kwargs)


if __name__ == "__main__":
    main_100k()
