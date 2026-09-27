# -*- coding: utf-8 -*-
"""
Test / validation script for the joint shape-completion + registration model
(DiffICP variant). Loads a checkpoint saved by train.py, runs the model on the
real US validation clouds, SAVES the completion + registration results to disk,
and prints the aggregate metrics.

It reuses Runner_DiffICP from train.py for model construction and checkpoint
loading, then runs its own evaluation loop so it can dump per-sample point clouds
and transforms (train.py's validate() only computes metrics).

Test-time pipeline: coarse pose from the network, `--passes`-1 re-orientation passes
(the network is re-run on the US cloud pre-rotated by the current estimate, see
reoriented_coarse), then DiffICP with test-time settings (--icp_*).

Saved per specimen/anatomy under
    <general.base_exp_dir>/<specimen>_<anatomy>/evaluation/ckpt-<epoch>/
  gt.ply                              ground-truth (CT) point cloud, GT frame
  sample_XXXX_input.ply               transformed intra (US) cloud  (network input)
  sample_XXXX_completed.ply           completion output, in its OWN (input) frame
  sample_XXXX_completed_registered.ply completion aligned to the CT frame by the ESTIMATED pose
  sample_XXXX_registered.ply          input US aligned to the CT frame by the ESTIMATED pose
  transforms.npz                      T_gt / T_coarse / T_icp for every sample
  metrics.csv                     per-sample CD/HD95/RTE/RRE(+ICP)/CD_REG/HD95_REG
  metrics_summary.txt             mean ± std over all samples

CD/HD95: completion vs CT (completed cloud placed with the GT pose). CD_REG/HD95_REG:
one-sided distance from the US registered by the estimated pose to the CT cloud.

Example:
    python test.py --target_specimen_id 2 --target_anatomy tibia --epoch 300
"""

from comet_ml import Experiment  # noqa: F401  (kept for parity with train.py imports)

import os
import argparse
import csv

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from train import Runner_DiffICP, seed_everything, device, CONFIG_FILE, TEST_TIME, reoriented_coarse
from utility import misc
from utility.read_confs import read_confs
from utility.converter import (
    compute_mean_RTE_RRE_batch,
    random_axis_angle_transform,
    invert_transformation_matrix,
)


def _save_pcd(path, xyz, color=None):
    """Write an (N, 3) numpy/torch array to a .ply file."""
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    if color is not None:
        pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(path, pcd)


def _apply_T(T, xyz):
    """Apply a 4x4 transform to an (N, 3) tensor."""
    return torch.matmul(T[:3, :3], xyz.transpose(0, 1)).transpose(0, 1) + T[:3, 3].unsqueeze(0)


@torch.no_grad()
def evaluate_and_save(runner, out_dir, n_runs=100, batch_size=4, save_pcd_runs=10,
                      icp_src='partial', passes=1):
    """Run the model on the real US validation clouds, save results + metrics.

    Mirrors Runner_ShapeCompletion.validate() (same normalization, random
    axis-angle disturbances and centering) but additionally writes the completion
    and registration outputs to `out_dir` and returns a dict of metric arrays.

    icp_src selects what the DiffICP refinement aligns to the GT:
      'partial' -> the raw partial US input (as in the paper); 'completed' -> the
    completed cloud. The coarse pose always comes from the completed-shape
    feature; only the ICP geometry differs.
    passes: network passes with test-time re-orientation (see reoriented_coarse).
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = runner.dataset
    # NOTE: like Runner.validate(), keep model_shapeCompletion in TRAIN mode --
    # AdaPoinTr returns the 4-tuple (incl. the dense completion ret[3]) only in
    # training mode; eval mode returns a 2-tuple. Only the pose head is eval'd.
    runner.model_PoseFromRebuildFeature.eval()

    # ---- validation inputs: real US clouds, normalized to the CT scale --------
    intra_data_raw = ds.intra_pcds_mm  # B*N*3 (mm)
    intra_data_raw = (intra_data_raw - ds.shape_center) / ds.shape_scale
    intra_data_raw = intra_data_raw.float().to(device)

    gt = ds.pcd_gt.to(device).unsqueeze(0).float()
    gt = misc.fps(gt, ds.npoints_input)
    gt_o3d = o3d.geometry.PointCloud()
    gt_o3d.points = o3d.utility.Vector3dVector(gt[0].cpu().numpy())
    _save_pcd(os.path.join(out_dir, "gt.ply"), gt[0], color=(1, 0, 0))

    _, rebuild_feature_gt_untransformed = runner.model_shapeCompletion(gt)

    scale = ds.shape_scale.item()
    metrics = {k: [] for k in ("CD", "HD95", "RTE", "RRE", "RTE_ICP", "RRE_ICP", "CD_REG", "HD95_REG")}
    rows = []
    T_gt_all, T_coarse_all, T_icp_all = [], [], []
    sample_idx = 0

    # Effective batch can be smaller than batch_size if there are fewer clouds.
    B_eff = min(batch_size, intra_data_raw.shape[0])
    total_samples = n_runs * B_eff
    pbar = tqdm(range(n_runs), desc="Evaluating", unit="run")
    for run_idx in pbar:
        idx = np.random.permutation(intra_data_raw.shape[0])[:batch_size]
        B = len(idx)
        intra_data = torch.clone(intra_data_raw[idx])

        # Random axis-angle disturbance (matches training/validation).
        Td = np.zeros([B, 4, 4])
        Td_inv = np.zeros([B, 4, 4])
        for i in range(B):
            Td[i] = random_axis_angle_transform(
                angle_deg_bound=ds.max_rotation_deg, t_bound=ds.max_translation)
            Td_inv[i] = invert_transformation_matrix(Td[i])
        Td = torch.tensor(Td).float().to(device)
        Td_inv = torch.tensor(Td_inv).float().to(device)

        intra_data_transformed = (torch.matmul(Td[:, :3, :3], intra_data.transpose(1, 2)).transpose(1, 2)
                                  + Td[:, :3, 3].unsqueeze(1))
        centroid = torch.mean(intra_data_transformed, dim=1, keepdim=True)
        intra_data_transformed = (intra_data_transformed - centroid).contiguous()
        T_centerized = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).float().to(device)
        T_centerized[:, :3, 3] = -centroid.squeeze(1)
        # Ground-truth transform that maps the (centered, transformed) input back
        # into the GT/CT frame -- this is what the pose head is trained to predict.
        T_gt = torch.matmul(Td_inv, torch.linalg.inv(T_centerized))

        outputs, rebuild_feature = runner.model_shapeCompletion(intra_data_transformed)
        mean_diff = outputs[3].mean(dim=1) - gt.mean(dim=1)

        # Coarse pose from the completed-shape feature (always).
        T_coarse = runner.model_PoseFromRebuildFeature(
            rebuild_feature.detach(),
            rebuild_feature_gt_untransformed.repeat((B, 1)).detach(),
            mean_diff.detach())
        # Test-time re-orientation: DiffICP starts from the last pass's estimate, and
        # its completion (mapped back to the input frame) is the one reported/saved.
        T_start, completed = T_coarse, outputs[3]
        if passes > 1:
            T_start, completed = reoriented_coarse(runner, intra_data_transformed, gt,
                                                   rebuild_feature_gt_untransformed, T_coarse,
                                                   outputs[3], passes)
        # DiffICP refinement: align either the raw partial US input (as in the paper)
        # or the completed cloud to the GT.
        src_icp = completed.detach() if icp_src == 'completed' else intra_data_transformed.detach()
        tgt_icp = gt.detach()
        if tgt_icp.shape[0] == 1 and B > 1:
            tgt_icp = tgt_icp.repeat(B, 1, 1)
        T_icp = runner.diff_icp(src_icp, tgt_icp, T_start)

        rte, rre = compute_mean_RTE_RRE_batch(T_gt.cpu().numpy(), T_coarse.detach().cpu().numpy(), scale)
        metrics["RTE"].extend(np.atleast_1d(rte).tolist())
        metrics["RRE"].extend(np.atleast_1d(rre).tolist())
        has_icp = T_icp is not None
        if has_icp:
            rte_icp, rre_icp = compute_mean_RTE_RRE_batch(T_gt.cpu().numpy(), T_icp.detach().cpu().numpy(), scale)
            metrics["RTE_ICP"].extend(np.atleast_1d(rte_icp).tolist())
            metrics["RRE_ICP"].extend(np.atleast_1d(rre_icp).tolist())

        for b in range(outputs[3].shape[0]):
            # Completed cloud brought into the GT frame with the GT transform (for CD/HD95).
            completed_gtframe = _apply_T(T_gt[b], completed[b])
            comp_o3d = o3d.geometry.PointCloud()
            comp_o3d.points = o3d.utility.Vector3dVector(completed_gtframe.cpu().numpy())
            d_gt2pred = np.asarray(gt_o3d.compute_point_cloud_distance(comp_o3d))
            d_pred2gt = np.asarray(comp_o3d.compute_point_cloud_distance(gt_o3d))
            CD = 0.5 * (d_gt2pred.mean() + d_pred2gt.mean()) * scale
            HD95 = 0.5 * (np.percentile(d_gt2pred, 95) + np.percentile(d_pred2gt, 95)) * scale
            metrics["CD"].append(float(CD))
            metrics["HD95"].append(float(HD95))

            T_reg = (T_icp[b] if has_icp else T_coarse[b])
            T_gt_all.append(T_gt[b].cpu().numpy())
            T_coarse_all.append(T_coarse[b].detach().cpu().numpy())
            T_icp_all.append(T_icp[b].detach().cpu().numpy() if has_icp else np.full((4, 4), np.nan))

            # Registration distance: US registered by the estimated pose -> CT cloud, one-sided.
            reg_o3d = o3d.geometry.PointCloud()
            reg_o3d.points = o3d.utility.Vector3dVector(_apply_T(T_reg, intra_data_transformed[b]).cpu().numpy())
            d_reg = np.asarray(reg_o3d.compute_point_cloud_distance(gt_o3d)) * scale
            CD_REG, HD95_REG = float(d_reg.mean()), float(np.percentile(d_reg, 95))
            metrics["CD_REG"].append(CD_REG)
            metrics["HD95_REG"].append(HD95_REG)

            row = {"sample": sample_idx, "CD": float(CD), "HD95": float(HD95),
                   "RTE": float(np.atleast_1d(rte)[b]), "RRE": float(np.atleast_1d(rre)[b])}
            if has_icp:
                row["RTE_ICP"] = float(np.atleast_1d(rte_icp)[b])
                row["RRE_ICP"] = float(np.atleast_1d(rre_icp)[b])
            row["CD_REG"], row["HD95_REG"] = CD_REG, HD95_REG
            rows.append(row)

            # Save a subset of point clouds. Three distinct artifacts:
            #   _input               : transformed US input (network input frame)
            #   _completed           : completion output, in its OWN (input) frame -- NOT registered
            #   _completed_registered: completion aligned to the CT/GT frame by the ESTIMATED pose
            #   _registered          : input US aligned to the CT/GT frame by the ESTIMATED pose
            if run_idx < save_pcd_runs:
                tag = f"sample_{sample_idx:04d}"
                _save_pcd(os.path.join(out_dir, f"{tag}_input.ply"),
                          intra_data_transformed[b], color=(0, 0, 1))
                _save_pcd(os.path.join(out_dir, f"{tag}_completed.ply"),
                          completed[b], color=(0, 1, 0))
                _save_pcd(os.path.join(out_dir, f"{tag}_completed_registered.ply"),
                          _apply_T(T_reg, completed[b]), color=(0, 1, 0))
                # Input aligned by the ESTIMATED pose -> should overlap the GT cloud.
                _save_pcd(os.path.join(out_dir, f"{tag}_registered.ply"),
                          _apply_T(T_reg, intra_data_transformed[b]), color=(0, 1, 1))
            sample_idx += 1

        pbar.set_postfix_str(f"samples {sample_idx}/{total_samples}")

    # ---- persist transforms + per-sample metrics -----------------------------
    np.savez(os.path.join(out_dir, "transforms.npz"),
             T_gt=np.stack(T_gt_all), T_coarse=np.stack(T_coarse_all), T_icp=np.stack(T_icp_all))

    fieldnames = list(rows[0].keys())
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {}
    lines = []
    for k, v in metrics.items():
        if not v:
            continue
        arr = np.asarray(v, dtype=np.float64)
        summary[k] = (arr.mean(), arr.std())
        lines.append(f"{k:8s}: {arr.mean():8.3f} ± {arr.std():.3f}  (n={arr.size})")
    summary_txt = "\n".join(lines)
    with open(os.path.join(out_dir, "metrics_summary.txt"), "w") as f:
        f.write(summary_txt + "\n")
    return summary, summary_txt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target_specimen_id', type=int, default=2)
    parser.add_argument('--target_anatomy', type=str, default='fibula')
    parser.add_argument('--epoch', type=int, default=150,
                        help='checkpoint epoch to load (ckpt-<epoch>.pth)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='explicit checkpoint path (overrides --epoch lookup)')
    parser.add_argument('--n_runs', type=int, default=30,
                        help='number of evaluation batches (random disturbances)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_pcd_runs', type=int, default=10,
                        help='save point clouds for the first N runs (0 = none)')
    parser.add_argument('--passes', type=int, default=TEST_TIME['passes'],
                        help='network passes with test-time re-orientation (1 = single pass)')
    # ---- test-time DiffICP overrides -----------------------------------------
    # The training-time settings (conf diff_icp block) are kept for training; at test time
    # DiffICP aligns the US cloud itself (as in the paper) with more selective soft
    # correspondences (sigma 0.001 vs 0.005 in training) and more iterations, which are cheap
    # without a gradient graph: point-to-point ICP only slowly slides a partial sweep along
    # the shaft. Denser correspondence subsampling (> 4096 points) did not help.
    parser.add_argument('--icp_iters', type=int, default=TEST_TIME['n_iters'],
                        help='DiffICP iterations at test time (overrides conf)')
    parser.add_argument('--icp_n_points', type=int, default=TEST_TIME['n_points'],
                        help='DiffICP correspondence subsample; 0 = full resolution (no subsample)')
    parser.add_argument('--icp_src', type=str, default=TEST_TIME['icp_src'], choices=['completed', 'partial'],
                        help="what DiffICP aligns to GT: the raw 'partial' US input or the 'completed' cloud")
    parser.add_argument('--icp_sigma', type=float, default=TEST_TIME['sigma'],
                        help='DiffICP soft-correspondence width at test time (normalised units; overrides conf)')
    parser.add_argument('--icp_trim', type=float, default=TEST_TIME['trim_ratio'],
                        help='DiffICP trimmed-correspondence ratio at test time (overrides conf)')
    args = parser.parse_args()

    conf = read_confs(CONFIG_FILE)
    for specimen_id in range(1, 15):
        if args.target_specimen_id > 0 and specimen_id != args.target_specimen_id:
            continue
        for anatomy in ["fibula", "tibia"]:
            if args.target_anatomy is not None and args.target_anatomy != anatomy:
                continue

            conf.put('dataset.specimen_id', specimen_id)
            conf.put('dataset.anatomy', anatomy)

            # Locate the checkpoint for this specimen/anatomy.
            if args.ckpt is not None:
                ckpt = args.ckpt
            else:
                ckpt = os.path.join(conf['general.base_exp_dir'], f"{specimen_id}_{anatomy}",
                                    "checkpoints", "ShapeCompletion", f"ckpt-{args.epoch}.pth")
            conf.pop('ckp_file_path', None)
            if not os.path.isfile(ckpt):
                print(f"[test] no checkpoint at {ckpt}; skipping {specimen_id}_{anatomy}")
                continue
            conf.put('ckp_file_path', ckpt)
            print(f"[test] loading {ckpt}")

            # Runner_DiffICP loads the checkpoint in __init__ (via ckp_file_path).
            runner = Runner_DiffICP(conf, experiment=None)

            # Test-time DiffICP settings (no gradient graph to bound at inference).
            runner.set_icp(args.icp_iters, args.icp_n_points, args.icp_sigma, args.icp_trim)
            print(f"[test] passes={args.passes}, DiffICP: n_iters={runner.diff_icp.n_iters}, "
                  f"n_points={'full' if runner.diff_icp.n_points is None else runner.diff_icp.n_points}, "
                  f"sigma={runner.diff_icp.sigma}, trim={runner.diff_icp.trim_ratio}, icp_src={args.icp_src}")

            out_dir = os.path.join(conf['general.base_exp_dir'], f"{specimen_id}_{anatomy}",
                                   "evaluation", f"ckpt-{args.epoch}")
            summary, summary_txt = evaluate_and_save(
                runner, out_dir, n_runs=args.n_runs,
                batch_size=args.batch_size, save_pcd_runs=args.save_pcd_runs,
                icp_src=args.icp_src, passes=args.passes)

            print(f"\n[test] {specimen_id}_{anatomy}  ->  {out_dir}")
            print(summary_txt)
            n_saved = min(args.save_pcd_runs, args.n_runs)
            if n_saved > 0:
                print(f"[test] point clouds + transforms.npz + metrics.csv saved to: {out_dir}")
            else:
                print(f"[test] metrics saved to: {out_dir}  (no point clouds; --save_pcd_runs 0)")
            print()
    print("finish")


if __name__ == '__main__':
    seed_everything(1234, deterministic=False)
    main()
