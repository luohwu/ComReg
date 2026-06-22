# -*- coding: utf-8 -*-
"""
Memory-safe differentiable ICP refinement (soft-correspondence + weighted Procrustes).

Refines a coarse SE(3) estimate ``T_init`` that aligns ``src`` to ``tgt`` by iterating:
  1. transform src by the current T
  2. soft correspondences  w_ij = softmax_j(-||src_i - tgt_j||^2 / (2 sigma^2))
  3. soft-corresponded target  q_i = sum_j w_ij tgt_j ,  confidence  c_i = max_j w_ij
  4. delta_T = weighted_Procrustes(src, q, c)   (differentiable SVD / Kabsch)
  5. T <- delta_T @ T

Memory (the soft-correspondence tensor is O(B * n_src * n_tgt)):
  * ``n_points`` subsamples src/tgt for the correspondence step (default 1024), so
    the (B, n, n) matrix stays small. DO NOT feed full ~16k clouds here -- 16k x 16k
    per iteration would be ~1 GB and is retained across iterations in the graph.
  * ``grad_checkpoint=True`` recomputes each iteration during backward, so peak
    activation memory is ~one iteration regardless of ``n_iters`` -- this is the key
    guard for high iteration counts.
  * sigma annealing (coarse -> fine) lets a few iterations converge.

The module has NO learnable parameters; it is a differentiable geometric refiner,
so gradients flow back through ``T_init`` (into the coarse pose head) and, if the
input points carry grad, into whatever produced them. In training we typically
pass detached point clouds so only the pose path is coupled.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def apply_transform(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """T: (B,4,4), pts: (B,N,3) -> (B,N,3)."""
    return torch.matmul(pts, T[:, :3, :3].transpose(1, 2)) + T[:, :3, 3].unsqueeze(1)


def weighted_procrustes(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Differentiable weighted rigid fit mapping P -> Q.

    P, Q: (B, N, 3); w: (B, N) non-negative weights. Returns (B, 4, 4).
    """
    w = w.clamp_min(0.0)
    wn = (w / w.sum(dim=1, keepdim=True).clamp_min(eps)).unsqueeze(-1)   # (B,N,1)
    Pc = (wn * P).sum(dim=1, keepdim=True)                               # (B,1,3)
    Qc = (wn * Q).sum(dim=1, keepdim=True)
    Pp, Qp = P - Pc, Q - Qc
    H = torch.matmul((wn * Pp).transpose(1, 2), Qp)                      # (B,3,3)
    U, _, Vh = torch.linalg.svd(H)
    V, Ut = Vh.transpose(1, 2), U.transpose(1, 2)
    det = torch.det(torch.matmul(V, Ut))                                 # (B,)
    D = torch.eye(3, device=P.device, dtype=P.dtype).unsqueeze(0).repeat(P.shape[0], 1, 1)
    D[:, 2, 2] = det                                                     # reflection fix
    R = torch.matmul(torch.matmul(V, D), Ut)                            # (B,3,3)
    t = Qc.squeeze(1) - torch.matmul(R, Pc.transpose(1, 2)).squeeze(-1)  # (B,3)
    T = torch.eye(4, device=P.device, dtype=P.dtype).unsqueeze(0).repeat(P.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def _subsample(pts: torch.Tensor, n) -> torch.Tensor:
    """Random subsample (B,N,3)->(B,n,3) for the correspondence step (memory bound)."""
    N = pts.shape[1]
    if n is None or n >= N:
        return pts
    idx = torch.randperm(N, device=pts.device)[:int(n)]
    return pts[:, idx, :]


class DiffICP(nn.Module):
    def __init__(self, n_iters: int = 3, sigma: float = 0.02, sigma_init: float = None,
                 n_points: int = 1024, grad_checkpoint: bool = True,
                 hard: bool = False, trim_ratio: float = 0.0):
        super().__init__()
        self.n_iters = int(n_iters)
        self.sigma = float(sigma)
        self.sigma_init = float(sigma_init) if sigma_init else float(sigma)
        self.n_points = n_points
        self.grad_checkpoint = bool(grad_checkpoint)
        # hard=True -> standard ICP (each src point -> its NEAREST tgt point), which
        # preserves discriminative features (vs soft averaging) and tends to lock
        # rotation better on elongated/near-symmetric bones. trim_ratio rejects the
        # worst correspondences each iteration (robust/trimmed ICP), useful for
        # partial-US -> full-CT where some matches are spurious. The gradient still
        # flows through the weighted Procrustes given the (fixed) hard correspondences.
        self.hard = bool(hard)
        self.trim_ratio = float(trim_ratio)

    def _sigma_at(self, it: int) -> float:
        if self.n_iters <= 1:
            return self.sigma
        r = it / (self.n_iters - 1)                       # 0 -> 1
        return float(self.sigma_init * (self.sigma / self.sigma_init) ** r)

    def _iter(self, T, src, tgt, sigma):
        src_t = apply_transform(T, src)
        d2 = torch.cdist(src_t, tgt) ** 2                 # (B, ns, nt)
        if self.hard:
            # standard ICP: each source point -> its single nearest target point
            d2min, idx = d2.min(dim=2)                    # (B, ns)
            q = torch.gather(tgt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, ns, 3)
            if self.trim_ratio and self.trim_ratio > 0.0:
                # keep the closest (1 - trim_ratio) correspondences (trimmed ICP)
                thr = torch.quantile(d2min, 1.0 - self.trim_ratio, dim=1, keepdim=True)
                w = (d2min <= thr).to(src_t.dtype)
            else:
                w = torch.ones_like(d2min)
        else:
            w_soft = torch.softmax(-d2 / (2.0 * sigma * sigma), dim=2)
            q = torch.matmul(w_soft, tgt)                 # (B, ns, 3)
            w = w_soft.max(dim=2).values                  # (B, ns) correspondence confidence
            if self.trim_ratio and self.trim_ratio > 0.0:
                # reject the worst correspondences (by nearest-neighbor distance),
                # same robust trimming as the hard path
                d2min = d2.min(dim=2).values
                thr = torch.quantile(d2min, 1.0 - self.trim_ratio, dim=1, keepdim=True)
                w = w * (d2min <= thr).to(w.dtype)
        dT = weighted_procrustes(src_t, q, w)
        return torch.matmul(dT, T)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, T_init: torch.Tensor) -> torch.Tensor:
        src = _subsample(src, self.n_points)
        tgt = _subsample(tgt, self.n_points)
        T = T_init
        for it in range(self.n_iters):
            sigma = self._sigma_at(it)
            if self.grad_checkpoint and self.training and torch.is_grad_enabled() and T.requires_grad:
                T = checkpoint(self._iter, T, src, tgt, sigma, use_reentrant=False)
            else:
                T = self._iter(T, src, tgt, sigma)
        return T


if __name__ == "__main__":
    # correctness + checkpoint smoke test (CPU)
    torch.manual_seed(0)
    B, N = 2, 2000
    tgt = torch.randn(B, N, 3)
    # a known rotation+translation applied to tgt gives src; ICP should recover ~inverse
    from math import cos, sin, pi
    a = 0.3
    Rz = torch.tensor([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
    src = (tgt @ Rz.T) + torch.tensor([0.1, -0.05, 0.02])
    T0 = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).requires_grad_(True)

    icp = DiffICP(n_iters=6, sigma=0.05, sigma_init=0.3, n_points=1024).train()
    T = icp(src, tgt, T0)
    # align src by T, measure residual to tgt (should shrink)
    res = (apply_transform(T, src) - tgt).pow(2).sum(-1).mean()
    res.backward()
    print("residual after ICP:", float(res))
    print("grad to T_init finite:", bool(torch.isfinite(T0.grad).all()))
    print("T finite:", bool(torch.isfinite(T).all()), "shape:", tuple(T.shape))
