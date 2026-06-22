import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.converter import *


class PoseFromRebuildFeature(nn.Module):
    """
    Input:
      rebuild_feature: Tensor of shape (B, M, D)  (e.g., concat of global_feature, q, coarse_xyz)

    Output:
      t: translation vector, shape (B, 3)
      q: unit quaternion (w, x, y, z), shape (B, 4)  (normalized per batch item)
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 256,
            pooling: str = "max",  # "mean" | "max" | "meanmax"
            dropout: float = 0.0,
            eps: float = 1e-8,
    ):
        super().__init__()
        assert pooling in {"mean", "max", "meanmax"}
        self.pooling = pooling
        self.eps = eps

        pooled_dim = in_dim if pooling in {"mean", "max"} else 2 * in_dim

        # Small MLP trunk after pooling over M
        self.trunk = nn.Sequential(
            # nn.LayerNorm(pooled_dim+3),
            nn.Linear(pooled_dim + 3, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Two heads: translation and quaternion
        self.head_t = nn.Sequential(
            nn.Linear(pooled_dim + 3, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Unbounded linear translation head: a Tanh here caps |t| to the
            # normalized (-1, 1) range and has vanishing gradient near the
            # bounds, which floors RTE and stalls refinement. Plain linear lets
            # the head represent the full translation range with healthy grads.
            nn.Linear(hidden_dim, 3),
        )
        self.head_q = nn.Sequential(
            nn.Linear(pooled_dim + 3, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Raw 4-vector, normalized to a unit quaternion in forward(). A Tanh
            # here squashes the pre-norm components and kills gradients for
            # larger rotations; normalization alone is the right constraint.
            nn.Linear(hidden_dim, 4),
        )

        # Optional: initialize quaternion head to near-identity (w≈1, xyz≈0)
        # nn.init.zeros_(self.head_q.weight)
        # nn.init.zeros_(self.head_q.bias)
        # with torch.no_grad():
        #     self.head_q.bias[0] = 1.0  # w component

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, D)
        if self.pooling == "mean":
            return x.mean(dim=1)  # (B, D)
        if self.pooling == "max":
            return x.max(dim=1).values  # (B, D)
        # meanmax
        mean = x.mean(dim=1)
        mx = x.max(dim=1).values
        return torch.cat([mean, mx], dim=-1)  # (B, 2D)

    def forward(self, rebuild_feature: torch.Tensor, rebuild_feature_CT: torch.Tensor, mean_diff: torch.Tensor):
        # if rebuild_feature.ndim != 3:
        #     raise ValueError(f"Expected rebuild_feature shape (B, M, D), got {tuple(rebuild_feature.shape)}")

        # pooled = self._pool(rebuild_feature-rebuild_feature_CT)    # (B, pooled_dim)
        pooled = rebuild_feature - rebuild_feature_CT
        #pooled=torch.cat((rebuild_feature,rebuild_feature_CT),dim=-1)
        # pooled_CT=self._pool(rebuild_feature_CT)
        h = torch.cat([pooled, mean_diff], dim=-1)  # (B, hidden_dim)
        # h=self.trunk(torch.cat((pooled,mean_diff),dim=-1))

        t = self.head_t(h)  # (B, 3)

        q_raw = self.head_q(h)  # (B, 4)
        q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + self.eps)  # normalize per batch item
        T_est = batch_quat_t_to_T(t, q)
        return T_est


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    B, M, D = 4, 576, 384
    rebuild_feature = torch.randn(B, M, D)

    model = PoseFromRebuildFeature(in_dim=D, hidden_dim=576, pooling="meanmax", dropout=0.1)
    t, q = model(rebuild_feature, rebuild_feature)

    print("t:", t.shape)  # (B, 3)
    print("q:", q.shape)  # (B, 4)
    print("q norms:", q.norm(dim=-1))  # ~1
