# -*- coding: utf-8 -*-
"""
DBSCAN denoiser for real ultrasound bone point clouds, plus dataset-wide
hyperparameter tuning.

Why DBSCAN: the dominant noise in the predicted-label US reconstructions is
*clustered* segmentation false-positives (detached blobs off the bone surface).
Per-point density filters fail on them: statistical-outlier-removal leaves them
(each blob point has neighbours), radius-outlier-removal erodes the thin true
surface. DBSCAN connected-component clustering targets the structure directly:
the true bone is one large connected manifold; false positives are small
detached clusters that we drop.

Deployability: the denoiser (`denoise_dbscan_cloud`) uses ONLY the US cloud, so
the tuned config can be applied at inference. The CLEAN `with_GT_labels`
reconstruction is used solely to *score* configs during tuning.

Run (tune one constant config over the whole dataset):
    ULTRABONES100K_ROOT=/path python -m utility.us_denoise
"""

import os
import itertools
from typing import Iterable, Tuple

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Denoiser (deployable: US cloud only)
# ---------------------------------------------------------------------------
# Constant config tuned over the whole dataset (93 records, scored by mean CD of
# the denoised predicted cloud to the clean with_GT_labels cloud):
#   baseline (no denoise) ...... 0.706 mm
#   eps=3.0 (dataset-mean opt) .. 0.659 mm  (gentlest; under-cleans the noisy tail)
#   eps=2.5 (DEFAULT) ........... 0.675 mm  (mean within noise of 3.0, but cleans
#                                            the worst records best: s02 tibia08
#                                            1.45->0.89 vs 1.07; neutral on clean)
# Safe basin is eps>=2.5, min_points=10, small frac; eps<=1.5 with min_points=20
# collapses the thin US surface (removes ~100%).
def denoise_dbscan_cloud(
    pcd: o3d.geometry.PointCloud,
    eps: float = 2.5,
    min_points: int = 10,
    min_cluster_frac: float = 0.01,
) -> o3d.geometry.PointCloud:
    """Keep points in DBSCAN clusters whose size >= `min_cluster_frac` of the
    cloud; drop DBSCAN noise (label -1) and small detached blobs.

    Args:
      eps:              neighbourhood radius (mm).
      min_points:       DBSCAN core-point density.
      min_cluster_frac: minimum cluster size as a fraction of the cloud, below
                        which a cluster is treated as an outlier blob and removed.
    """
    n = len(pcd.points)
    if n == 0:
        return pcd
    labels = np.asarray(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points)))
    keep = np.zeros(n, dtype=bool)
    min_size = float(min_cluster_frac) * n
    for c in np.unique(labels):
        if c < 0:  # DBSCAN noise
            continue
        idx = np.where(labels == c)[0]
        if idx.size >= min_size:
            keep[idx] = True
    return pcd.select_by_index(np.where(keep)[0])


# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------
def _cd_sym(a: o3d.geometry.PointCloud, b: o3d.geometry.PointCloud) -> float:
    if len(a.points) == 0 or len(b.points) == 0:
        return float("nan")
    d1 = np.asarray(a.compute_point_cloud_distance(b)).mean()
    d2 = np.asarray(b.compute_point_cloud_distance(a)).mean()
    return 0.5 * (d1 + d2)


def iter_record_pairs(root: str, anatomies=("fibula", "tibia")) -> Iterable[Tuple]:
    """Yield (specimen_id, anatomy, record, pred_path, gt_path) for records that
    have BOTH a predicted-label and a GT-label reconstruction."""
    for sid in range(1, 15):
        urf = os.path.join(root, f"specimen{sid:02d}", "ultrasound_records")
        if not os.path.isdir(urf):
            continue
        for anat in anatomies:
            af = os.path.join(urf, anat)
            if not os.path.isdir(af):
                continue
            for rec in sorted(os.listdir(af)):
                rec_recon = os.path.join(af, rec, "3D_reconstructions")
                pred = os.path.join(rec_recon, "with_pred_labels", "reconstruction_pcd_filtered.xyz")
                gt = os.path.join(rec_recon, "with_GT_labels", "reconstruction_pcd_filtered.xyz")
                if os.path.isfile(pred) and os.path.isfile(gt):
                    yield sid, anat, rec, pred, gt


def tune_over_dataset(
    root: str,
    eps_grid=(1.5, 2.0, 2.5, 3.0),
    min_points_grid=(10, 20),
    min_cluster_frac_grid=(0.005, 0.01, 0.02),
    voxel: float = 1.0,
    anatomies=("fibula", "tibia"),
):
    """Grid-search one CONSTANT (eps, min_points, min_cluster_frac) that minimizes
    the mean CD between the denoised predicted cloud and the clean GT-label cloud,
    over every record in the dataset."""
    # Preload each record once (voxel-downsampled for speed; the training pipeline
    # FPS-downsamples to npoints_input anyway).
    records = []
    for sid, anat, rec, pp, gp in iter_record_pairs(root, anatomies):
        pred = o3d.io.read_point_cloud(pp)
        gt = o3d.io.read_point_cloud(gp)
        if voxel and voxel > 0:
            pred = pred.voxel_down_sample(voxel)
            gt = gt.voxel_down_sample(voxel)
        if len(pred.points) and len(gt.points):
            records.append((sid, anat, rec, pred, gt))
    if not records:
        print("No records with both pred and GT reconstructions found. Check ULTRABONES100K_ROOT.")
        return None
    print(f"loaded {len(records)} records (voxel={voxel} mm)")

    baseline = float(np.nanmean([_cd_sym(p, g) for *_, p, g in records]))
    print(f"baseline (no denoise) mean CD-to-GT = {baseline:.3f} mm\n")

    results = []
    for eps, mp, frac in itertools.product(eps_grid, min_points_grid, min_cluster_frac_grid):
        cds, removed = [], []
        for _, _, _, pred, gt in records:
            den = denoise_dbscan_cloud(pred, eps, mp, frac)
            cds.append(_cd_sym(den, gt))
            removed.append(1.0 - len(den.points) / max(1, len(pred.points)))
        mean_cd = float(np.nanmean(cds))
        results.append((mean_cd, eps, mp, frac, float(np.mean(removed)) * 100))
        print(f"  eps={eps:<4} min_points={mp:<3} min_cluster_frac={frac:<6}: "
              f"mean CD-to-GT={mean_cd:.3f} mm  (avg removed {np.mean(removed)*100:.1f}%)")

    results.sort(key=lambda r: r[0])
    best = results[0]
    print(f"\nBEST constant config: eps={best[1]}, min_points={best[2]}, "
          f"min_cluster_frac={best[3]}")
    print(f"  mean CD-to-GT = {best[0]:.3f} mm (baseline {baseline:.3f}), "
          f"avg removed {best[4]:.1f}%")

    # per-anatomy breakdown for the best config
    for anat in anatomies:
        cds = [_cd_sym(denoise_dbscan_cloud(p, best[1], best[2], best[3]), g)
               for s, a, r, p, g in records if a == anat]
        if cds:
            print(f"  [{anat}] mean CD-to-GT = {np.nanmean(cds):.3f} mm over {len(cds)} records")
    return best


if __name__ == "__main__":
    root = os.environ.get("ULTRABONES100K_ROOT", "/mnt/UltraBones100k")
    tune_over_dataset(root)
