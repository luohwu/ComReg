import copy

import numpy as np

import cv2
import open3d as o3d
from tqdm import  tqdm
from scipy.spatial.transform import Rotation as R



import torch

from typing import Tuple

def se3_to_qt(
    T: torch.Tensor,
    eps: float = 1e-8,
    canonicalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of SE(3) transforms to (quaternion, translation).

    Args:
        T: (..., 4, 4) homogeneous transform (or (..., 3, 4) accepted).
        eps: small number for numerical stability.
        canonicalize: if True, flip sign so qw >= 0 (removes q vs -q ambiguity).

    Returns:
        q: (..., 4) quaternion in (w, x, y, z) order, unit-normalized.
        t: (..., 3) translation vector.
    """
    if T.shape[-2:] == (3, 4):
        R = T[..., :3, :3]
        t = T[..., :3, 3]
    elif T.shape[-2:] == (4, 4):
        R = T[..., :3, :3]
        t = T[..., :3, 3]
    else:
        raise ValueError(f"Expected T with shape (...,4,4) or (...,3,4), got {tuple(T.shape)}")

    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = r00 + r11 + r22

    # Prepare output tensors
    qw = torch.empty_like(trace)
    qx = torch.empty_like(trace)
    qy = torch.empty_like(trace)
    qz = torch.empty_like(trace)

    # Case 1: trace > 0
    m0 = trace > 0.0
    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=eps)) * 2.0  # S=4*qw
    qw0 = 0.25 * s0
    qx0 = (r21 - r12) / s0
    qy0 = (r02 - r20) / s0
    qz0 = (r10 - r01) / s0

    # Case 2/3/4: trace <= 0, choose largest diagonal
    m1 = (~m0) & (r00 >= r11) & (r00 >= r22)
    s1 = torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=eps)) * 2.0  # S=4*qx
    qw1 = (r21 - r12) / s1
    qx1 = 0.25 * s1
    qy1 = (r01 + r10) / s1
    qz1 = (r02 + r20) / s1

    m2 = (~m0) & (~m1) & (r11 >= r22)
    s2 = torch.sqrt(torch.clamp(1.0 + r11 - r00 - r22, min=eps)) * 2.0  # S=4*qy
    qw2 = (r02 - r20) / s2
    qx2 = (r01 + r10) / s2
    qy2 = 0.25 * s2
    qz2 = (r12 + r21) / s2

    m3 = (~m0) & (~m1) & (~m2)
    s3 = torch.sqrt(torch.clamp(1.0 + r22 - r00 - r11, min=eps)) * 2.0  # S=4*qz
    qw3 = (r10 - r01) / s3
    qx3 = (r02 + r20) / s3
    qy3 = (r12 + r21) / s3
    qz3 = 0.25 * s3

    # Assign by masks
    qw = torch.where(m0, qw0, qw)
    qx = torch.where(m0, qx0, qx)
    qy = torch.where(m0, qy0, qy)
    qz = torch.where(m0, qz0, qz)

    qw = torch.where(m1, qw1, qw)
    qx = torch.where(m1, qx1, qx)
    qy = torch.where(m1, qy1, qy)
    qz = torch.where(m1, qz1, qz)

    qw = torch.where(m2, qw2, qw)
    qx = torch.where(m2, qx2, qx)
    qy = torch.where(m2, qy2, qy)
    qz = torch.where(m2, qz2, qz)

    qw = torch.where(m3, qw3, qw)
    qx = torch.where(m3, qx3, qx)
    qy = torch.where(m3, qy3, qy)
    qz = torch.where(m3, qz3, qz)

    q = torch.stack([qw, qx, qy, qz], dim=-1)

    # Normalize (important if R is slightly non-orthonormal)
    q = q / (q.norm(dim=-1, keepdim=True) + eps)

    # Optional: canonicalize sign so qw >= 0
    if canonicalize:
        sign = torch.where(q[..., :1] < 0, -1.0, 1.0).to(q.dtype)
        q = q * sign

    return t, q

import torch
import torch.nn as nn
import torch.nn.functional as F

def rte_rre_loss_batch(
    T_est: torch.Tensor,           # (B,4,4)
    T_gt: torch.Tensor,            # (B,4,4)
    translation_scale: float = 1.0,
    w_rte: float = 1.0,
    w_rre: float = 1.0,
    eps: float = 1e-7,
    return_per_sample: bool = False,
):
    """
    Differentiable batch loss minimizing RTE + RRE.

    Returns:
      loss (scalar)
      optionally: rte_per (B,), rre_rad_per (B,), rre_deg_per (B,)
    """
    if T_est.ndim != 3 or T_est.shape[1:] != (4, 4):
        raise ValueError(f"T_est must be (B,4,4), got {tuple(T_est.shape)}")
    if T_gt.shape != T_est.shape:
        raise ValueError(f"T_gt must match T_est shape, got {tuple(T_gt.shape)} vs {tuple(T_est.shape)}")

    # --- translations ---
    t_est = T_est[:, :3, 3] * translation_scale  # (B,3)
    t_gt  = T_gt[:,  :3, 3] * translation_scale  # (B,3)

    rte_per = torch.linalg.norm(t_est - t_gt, dim=-1)  # (B,)
    rte = rte_per.mean()

    # --- rotations ---
    R_est = T_est[:, :3, :3]  # (B,3,3)
    R_gt  = T_gt[:,  :3, :3]  # (B,3,3)

    # Relative rotation: R_rel = R_est^T R_gt
    R_rel = torch.matmul(R_est.transpose(1, 2), R_gt)  # (B,3,3)

    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # (B,)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    rre_rad_per = torch.acos(cos_theta)  # (B,)
    rre_rad = rre_rad_per.mean()
    rre_deg = rre_rad * (180.0 / torch.pi)

    # Total
    loss = w_rte * rte + w_rre * rre_deg

    if return_per_sample:
        rre_deg_per = rre_rad_per * (180.0 / torch.pi)
        return loss, rte_per, rre_rad_per, rre_deg_per
    return loss

def normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # q: (B,4)
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_sign_align(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Align q_gt sign to q_pred so dot(q_pred, q_gt) >= 0.
    This avoids penalizing equivalent rotations (q and -q).
    """
    dot = (q_pred * q_gt).sum(dim=-1, keepdim=True)  # (B,1)
    q_gt_aligned = torch.where(dot < 0, -q_gt, q_gt)
    return q_gt_aligned


def quat_geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Geodesic rotation loss on SO(3) using quaternion inner product.
    Returns mean angle in radians (or you can square it).
      angle = 2 * acos(|<q_pred, q_gt>|)
    """
    q_pred = normalize_quat(q_pred)
    q_gt = normalize_quat(q_gt)

    # sign-invariant
    dot = (q_pred * q_gt).sum(dim=-1).abs()  # (B,)
    dot = torch.clamp(dot, 0.0, 1.0 - eps)

    angle = 2.0 * torch.acos(dot)  # (B,) in radians
    return angle.mean()


def quat_cosine_loss(q_pred: torch.Tensor, q_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Very stable alternative to geodesic:
      loss = 1 - |<q_pred, q_gt>|
    """
    q_pred = normalize_quat(q_pred, eps=eps)
    q_gt = normalize_quat(q_gt, eps=eps)
    dot = (q_pred * q_gt).sum(dim=-1).abs()
    return (1.0 - dot).mean()
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def vectorToMatrix(t, rotation_vector, quat=False):
       T=np.eye(4)
       if quat:
              T[:3, :3] = quaternion_to_matrix(torch.tensor(rotation_vector)).cpu().numpy()
       else:
              r=R.from_euler('xyz', rotation_vector, degrees=True)
              T[:3, :3] = r.as_matrix()
       T[:3,3]=np.asarray(t)
       return T


def random_axis_angle_transform(angle_deg_bound=90.0, t_bound=0.5):
       """Sample a random rigid transform as a screw about a single random axis.

       - A uniformly random unit axis is drawn on the sphere.
       - Rotation is `angle` degrees about that axis, angle ~ U[-angle_deg_bound, +angle_deg_bound],
         so the total geodesic rotation is bounded by `angle_deg_bound`.
       - Translation is along the SAME axis: t = axis * s, s ~ U[-t_bound, +t_bound].

       Returns a (4, 4) float numpy matrix.
       """
       axis = np.random.randn(3)
       axis = axis / (np.linalg.norm(axis) + 1e-8)
       angle = np.random.uniform(-angle_deg_bound, angle_deg_bound)
       t_scalar = np.random.uniform(-t_bound, t_bound)
       T = np.eye(4)
       T[:3, :3] = R.from_rotvec(axis * np.radians(angle)).as_matrix()
       T[:3, 3] = axis * t_scalar
       return T


def merge_point_clouds(pcd_list, *, voxel_size: float | None = None, estimate_normals: bool = False):
    """
    Merge a list of open3d.geometry.PointCloud into one PointCloud.
    Optionally voxel-downsample and (re)estimate normals.
    """
    if len(pcd_list) == 0:
        return o3d.geometry.PointCloud()

    # Fast path: concatenate points (+ colors/normals if present consistently)
    pts = []
    cols = []
    nrm = []

    has_colors = all(p.has_colors() for p in pcd_list)
    has_normals = all(p.has_normals() for p in pcd_list)

    for p in pcd_list:
        pts.append(np.asarray(p.points))
        if has_colors:
            cols.append(np.asarray(p.colors))
        if has_normals:
            nrm.append(np.asarray(p.normals))

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(pts))

    if has_colors:
        merged.colors = o3d.utility.Vector3dVector(np.vstack(cols))
    if has_normals:
        merged.normals = o3d.utility.Vector3dVector(np.vstack(nrm))

    if voxel_size is not None and voxel_size > 0:
        merged = merged.voxel_down_sample(float(voxel_size))

    if estimate_normals:
        merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * (voxel_size or 1.0), max_nn=30)
        )
        merged.normalize_normals()

    return merged


# def vectorToMatrix(vector, quat=False):
#        T=np.eye(4)
#        if quat:
#               r=R.from_quat(vector[3:])
#        else:
#               r=R.from_euler('xyz', vector[3:], degrees=True)
#        T[:3, :3] = r.as_matrix()
#        T[:3,3]=np.asarray(vector[:3])
#        return T

def normalize_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
       """
       Normalize a point cloud by shifting its centroid to (0,0,0).

       Args:
           pcd (o3d.geometry.PointCloud): The input point cloud.

       Returns:
           o3d.geometry.PointCloud: The normalized point cloud.
       """
       # Convert point cloud to numpy array
       points = np.asarray(pcd.points)

       # Compute centroid
       centroid = points.mean(axis=0)

       # Shift points to have centroid at (0, 0, 0)
       points_normalized = points - centroid

       # Update the point cloud
       pcd_normalized = o3d.geometry.PointCloud()
       pcd_normalized.points = o3d.utility.Vector3dVector(points_normalized)

       # Copy colors if available
       if pcd.has_colors():
              pcd_normalized.colors = pcd.colors
       if pcd.has_normals():
              pcd_normalized.normals = pcd.normals

       return pcd_normalized

def compute_RTE_RRE(T1, T2,translation_scale,pcd_US_moved,CT_model_mesh):
    # Extract rotation matrices and translation vectors
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Compute Relative Translation Error (RTE)
    rte = np.linalg.norm(t1 - t2)

    # Compute Relative Rotation Error (RRE)
    R_diff = np.dot(R1.T, R2)  # Compute the relative rotation matrix
    trace = np.trace(R_diff)
    rre = np.arccos(min(max((trace - 1) / 2, -1), 1))  # Clamp the value to avoid numerical issues
    rre_degrees = np.degrees(rre)  # Convert to degree
    # print(f"t1:{t1}, t2:{t2}")
    pcd_CT=CT_model_mesh.sample_points_uniformly(1000000)
    pcd_US_registered=copy.deepcopy(pcd_US_moved).transform(T1)
    # o3d.visualization.draw_geometries([pcd_CT.paint_uniform_color((1,0,0)),pcd_US_registered.paint_uniform_color((0,1,0))])
    dis=pcd_US_registered.compute_point_cloud_distance(pcd_CT)
    dis=np.asarray(dis).mean()*translation_scale

    return rte*translation_scale, rre_degrees,dis,pcd_US_registered

def compute_RTE_RRE(T1, T2,translation_scale):
    # Extract rotation matrices and translation vectors
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Compute Relative Translation Error (RTE)
    rte = np.linalg.norm(t1 - t2)

    # Compute Relative Rotation Error (RRE)
    R_diff = np.dot(R1.T, R2)  # Compute the relative rotation matrix
    trace = np.trace(R_diff)
    rre = np.arccos(min(max((trace - 1) / 2, -1), 1))  # Clamp the value to avoid numerical issues
    rre_degrees = np.degrees(rre)  # Convert to degree
    # print(f"t1:{t1}, t2:{t2}")

    return rte*translation_scale, rre_degrees

def compute_RTE_RRE_pcds(T_est, T_gt, translation_scale, src_pcd_moved, tgt_pcd):
    # Extract rotation matrices and translation vectors
    R1, t1 = T_est[:3, :3], T_est[:3, 3]
    R2, t2 = T_gt[:3, :3], T_gt[:3, 3]

    # Compute Relative Translation Error (RTE)
    rte = np.linalg.norm(t1 - t2)

    # Compute Relative Rotation Error (RRE)
    R_diff = np.dot(R1.T, R2)  # Compute the relative rotation matrix
    trace = np.trace(R_diff)
    rre = np.arccos(min(max((trace - 1) / 2, -1), 1))  # Clamp the value to avoid numerical issues
    rre_degrees = np.degrees(rre)  # Convert to degree
    # print(f"t1:{t1}, t2:{t2}")
    pcd_US_registered=copy.deepcopy(src_pcd_moved).transform(T_est)
    dis=pcd_US_registered.compute_point_cloud_distance(tgt_pcd)
    dis=np.asarray(dis).mean()*translation_scale

    return rte*translation_scale, rre_degrees,dis,pcd_US_registered


def compute_mean_RTE_RRE_batch(T1: np.ndarray, T2: np.ndarray, translation_scale: float = 1.0):
    """
    Compute mean RTE and mean RRE over a batch of 4x4 transforms.

    Args:
      T1, T2: (B, 4, 4) numpy arrays (homogeneous transforms)
      translation_scale: multiply translations by this factor BEFORE computing RTE
                         (useful if your t is normalized / in different units)

    Returns:
      mean_rte: float   (same units as translations after scaling)
      mean_rre_deg: float  (degrees)
      rte_per_sample: (B,) array
      rre_deg_per_sample: (B,) array
    """
    T1 = np.asarray(T1)
    T2 = np.asarray(T2)

    if T1.shape != T2.shape or T1.ndim != 3 or T1.shape[1:] != (4, 4):
        raise ValueError(f"Expected T1 and T2 shape (B,4,4) and equal. Got {T1.shape} and {T2.shape}")

    # Extract rotations and translations
    R1 = T1[:, :3, :3]                         # (B,3,3)
    t1 = T1[:, :3, 3] * translation_scale      # (B,3)
    R2 = T2[:, :3, :3]                         # (B,3,3)
    t2 = T2[:, :3, 3] * translation_scale      # (B,3)

    # --- RTE: L2 distance between translations ---
    rte = np.linalg.norm(t1 - t2, axis=1)      # (B,)

    # --- RRE: angle of relative rotation ---
    # R_diff = R1^T R2 for each batch item
    R_diff = np.matmul(np.transpose(R1, (0, 2, 1)), R2)  # (B,3,3)

    trace = np.trace(R_diff, axis1=1, axis2=2)           # (B,)
    cos_theta = (trace - 1.0) / 2.0                      # (B,)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)            # numeric safety
    rre_rad = np.arccos(cos_theta)                       # (B,)
    rre_deg = np.degrees(rre_rad)                        # (B,)

    return rte,rre_deg


def invert_transformation_matrix(T):
    R = T[:3, :3]  # Extract the top-left 3x3 rotation matrix
    t = T[:3, 3]   # Extract the translation vector

    R_inv = R.T   # Inverse of rotation matrix
    t_inv = -R_inv @ t  # Inverse translation

    T_inv = np.eye(4)  # Create a 4x4 identity matrix
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def MatrixToVector(T):
       r=R.from_matrix(T[:3,:3]).as_euler('xyz',degrees=True)
       return np.array([T[0,3],T[1,3],T[2,3],r[0],r[1],r[2]])


def batch_quat_t_to_T(
    t: torch.Tensor,         # (B, 3)
    q: torch.Tensor,         # (B, 4) in (w, x, y, z)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a batch of 4x4 homogeneous transformation matrices from translation + quaternion.

    Args:
      t: (B, 3) translation
      q: (B, 4) quaternion in (w, x, y, z) convention (will be normalized)
      eps: numerical stability

    Returns:
      T: (B, 4, 4) homogeneous transform, where:
         T[:, :3, :3] = R
         T[:, :3,  3] = t
         T[:,  3,  3] = 1
    """
    if t.ndim != 2 or t.shape[-1] != 3:
        raise ValueError(f"t must have shape (B, 3), got {tuple(t.shape)}")
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"q must have shape (B, 4), got {tuple(q.shape)}")
    if t.shape[0] != q.shape[0]:
        raise ValueError(f"Batch mismatch: t has B={t.shape[0]}, q has B={q.shape[0]}")

    # Normalize quaternion
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    w, x, y, z = q.unbind(dim=-1)

    # Rotation matrix from quaternion (w, x, y, z)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    B = q.shape[0]
    R = torch.empty((B, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    # Homogeneous transform
    T = torch.zeros((B, 4, 4), device=q.device, dtype=q.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T

def sweepToPCD(sweep,temporalOff,calibrationT):
       pcd=o3d.geometry.PointCloud()
       sweep.tracking().temporalOffset=temporalOff
       sweep.tracking().calibration=calibrationT
       sweep_np=sweep.numpy()
       for i in range(sweep_np.shape[0]):
              img_imf=sweep.img(i)
              img_np=np.array((sweep_np[i]))
              rows_index, cols_index,_ = np.where(img_np > 0)
              num_points = len(cols_index)
              xyz_pixels = np.stack([cols_index, rows_index, np.zeros(num_points), np.ones(num_points)], axis=0)
              xyz_img = np.dot(img_imf.pixelToImageMatrix(), xyz_pixels)
              xyz_world = np.dot(sweep.matrix(i), xyz_img)
              xyz_world = xyz_world[:3, :]
              xyz_world = xyz_world.T
              pcd.points.extend(xyz_world)
       pcd = pcd.voxel_down_sample(voxel_size=0.1)
       return pcd

def sweepFilesToPCD(files:list,temporalOff,calibrationT,spacing=None,target_depth=[0,1],labeled=False):
       pcd=o3d.geometry.PointCloud()
       for file in files:
              sweep=imfusion.io.open(file)[0]
              sweep.tracking().temporalOffset=temporalOff
              sweep.tracking().calibration=calibrationT
              sweep_np=sweep.numpy()
              for i in range(sweep_np.shape[0]):
                     img_imf=sweep.img(i)
                     if spacing:
                            img_imf.spacing = spacing
                     img_np=np.array((sweep_np[i]))
                     non_zero_pixels=img_np[img_np>0]
                     # cv2.imshow("original image",img_np)
                     # cv2.waitKey(0)
                     # _,img_thresholded=cv2.threshold(img_np,np.percentile(non_zero_pixels,95),255,cv2.THRESH_BINARY)
                     # cv2.imshow("image",img_thresholded)
                     # cv2.waitKey(0)
                     if labeled==False:
                            pass
                            # rows_index, cols_index = np.where(img_thresholded > 0)
                     else:
                            rows_index, cols_index,_ = np.where(img_np > 0)
                     index_within_target_depth=(rows_index<target_depth[1]*img_imf.height) * ((rows_index>target_depth[0]*img_imf.height))
                     rows_index=rows_index[index_within_target_depth]
                     cols_index=cols_index[index_within_target_depth]
                     num_points = len(cols_index)
                     xyz_pixels = np.stack([cols_index, rows_index, np.zeros(num_points), np.ones(num_points)], axis=0)
                     xyz_img = np.dot(img_imf.pixelToImageMatrix(), xyz_pixels)
                     xyz_world = np.dot(sweep.matrix(i), xyz_img)
                     xyz_world = xyz_world[:3, :]
                     xyz_world = xyz_world.T
                     pcd.points.extend(xyz_world)
       pcd = pcd.voxel_down_sample(voxel_size=0.1)
       return pcd

def transform_Nx3_array(T,xyz):
       return (T[:3,:3] @ xyz[:,:3].T).T + T[:3,3]
       # xyz_homo=np.hstack([xyz,np.ones((xyz.shape[0],1))])
       # return ((T@xyz_homo.T).T)[:,:3]


def array2open3dPCD(xyz,color='yellow'):
       pcd=o3d.geometry.PointCloud()
       pcd.points=o3d.utility.Vector3dVector(xyz)
       if color=="yellow":
              pcd.paint_uniform_color([1, 0.706, 0])
       else:
              pcd.paint_uniform_color([0, 0.651, 0.929])

       return pcd


def to_o3d_pcd(x):
    """
    Convert an (N,3) numpy array or torch tensor into an Open3D PointCloud.
    Accepts float32/float64. Returns o3d.geometry.PointCloud.
    """
    import numpy as np
    import open3d as o3d

    # Torch -> numpy (CPU)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected shape (N,3), got {x.shape}")

    # Open3D expects float64 for Vector3dVector
    pts = x.astype(np.float64, copy=False)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

import math
def filter_src_by_tgt_distance_o3d(src_pcd, tgt_pcd, dis_threshold):
    """
    Open3D point-cloud version. Keeps src points whose nearest distance to tgt <= threshold.

    Returns:
        src_filtered_pcd, mask, nn_dist
    """
    if not isinstance(src_pcd, o3d.geometry.PointCloud) or not isinstance(tgt_pcd, o3d.geometry.PointCloud):
        raise TypeError("src_pcd and tgt_pcd must be open3d.geometry.PointCloud")

    src = np.asarray(src_pcd.points)
    tgt = np.asarray(tgt_pcd.points)

    if tgt.shape[0] == 0:
        mask = np.zeros((src.shape[0],), dtype=bool)
        nn_dist = np.full((src.shape[0],), np.inf, dtype=np.float64)
        return o3d.geometry.PointCloud(), mask, nn_dist

    kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
    nn_dist = np.empty((src.shape[0],), dtype=np.float64)

    for i, p in enumerate(src):
        _, idx, d2 = kdtree.search_knn_vector_3d(p, 1)
        nn_dist[i] = math.sqrt(d2[0]) if len(d2) > 0 else np.inf

    mask = nn_dist <= float(dis_threshold)

    src_filtered = o3d.geometry.PointCloud()
    src_filtered.points = o3d.utility.Vector3dVector(src[mask])
    # preserve colors/normals if present
    if src_pcd.has_colors():
        src_filtered.colors = o3d.utility.Vector3dVector(np.asarray(src_pcd.colors)[mask])
    if src_pcd.has_normals():
        src_filtered.normals = o3d.utility.Vector3dVector(np.asarray(src_pcd.normals)[mask])

    return src_filtered

if __name__=='__main__':
    print(quaternion_to_matrix(torch.randn((1,4))))


