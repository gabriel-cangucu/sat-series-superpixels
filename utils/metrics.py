import torch
import torch.nn.functional as F
from torchmetrics import Metric


def _compute_boundary_map(seg: torch.Tensor) -> torch.Tensor:
    """
    Returns a binary boundary map.
    """
    boundary = torch.zeros_like(seg, dtype=torch.bool)

    boundary[:, :, :-1] |= (seg[:, :, :-1] != seg[:, :, 1:])
    boundary[:, :-1, :] |= (seg[:, :-1, :] != seg[:, 1:, :])

    return boundary


def _dilate(boundary: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Dilates a binary boundary map by given radius using max pooling.
    """
    if radius <= 0:
        return boundary
    
    boundary = boundary.float()

    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=radius)

    return dilated.squeeze(1).bool()


class BoundaryPrecision(Metric):
    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        self.radius = radius

        self.add_state("sum_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Expects preds and target to be of shape (B, H, W) or (B, 1, H, W)
        """
        preds = preds.squeeze()
        target = target.squeeze()

        sp_boundary = _compute_boundary_map(preds)
        gt_boundary = _compute_boundary_map(target)

        gt_dilated = _dilate(gt_boundary, self.radius)    

        true_positive = (sp_boundary & gt_dilated).sum(dim=(1, 2)).float()
        predicted_positive = sp_boundary.sum(dim=(1, 2)).float()

        precision = torch.where(
            predicted_positive > 0,
            true_positive / predicted_positive,
            torch.zeros_like(true_positive) 
        )

        self.sum_score += precision.sum()
        self.total += preds.shape[0]
    
    def compute(self) -> torch.Tensor:
        return self.sum_score / self.total


class BoundaryRecall(Metric):
    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        self.radius = radius

        self.add_state("sum_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Expects preds and target to be of shape (B, H, W) or (B, 1, H, W)
        """
        preds = preds.squeeze()
        target = target.squeeze()

        sp_boundary = _compute_boundary_map(preds)
        gt_boundary = _compute_boundary_map(target)

        sp_dilated = _dilate(sp_boundary, self.radius)    

        true_positive = (gt_boundary & sp_dilated).sum(dim=(1, 2)).float()
        predicted_positive = gt_boundary.sum(dim=(1, 2)).float()

        precision = torch.where(
            predicted_positive > 0,
            true_positive / predicted_positive,
            torch.zeros_like(true_positive) 
        )

        self.sum_score += precision.sum()
        self.total += preds.shape[0]
    
    def compute(self) -> torch.Tensor:
        return self.sum_score / self.total


class AchievableSegmentationAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("sum_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Expects preds and target to be of shape (B, H, W) or (B, 1, H, W)
        """
        preds = preds.squeeze()
        target = target.squeeze()

        B, H, W = preds.shape
        N = H * W
        
        preds_u, preds_dense = torch.unique(preds, return_inverse=True)
        target_u, target_dense = torch.unique(target, return_inverse=True)
        
        max_p = preds_u.numel()
        max_t = target_u.numel()
        
        P = preds_dense.view(B, N)
        T = target_dense.view(B, N)
        
        batch_offsets = torch.arange(B, device=preds.device).view(B, 1) * max_p
        P_shifted = P + batch_offsets
        
        hash_idx = P_shifted * max_t + T
        
        counts = torch.bincount(hash_idx.view(-1), minlength=B * max_p * max_t)
        counts = counts.view(B, max_p, max_t)
        
        max_overlaps = counts.max(dim=2).values  # Shape: (B, max_p)
        correct_per_image = max_overlaps.sum(dim=1)  # Shape: (B,)
        
        asa_per_image = correct_per_image.float() / N

        self.sum_score += asa_per_image.sum()
        self.total += B
    
    def compute(self) -> torch.Tensor:
        return self.sum_score / self.total


class UndersegmentationError(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("sum_error", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Expects preds and target to be of shape (B, H, W) or (B, 1, H, W)
        """
        preds = preds.squeeze()
        target = target.squeeze()

        B, H, W = preds.shape
        N = H * W
        
        preds_u, preds_dense = torch.unique(preds, return_inverse=True)
        target_u, target_dense = torch.unique(target, return_inverse=True)
        
        max_p = preds_u.numel()
        max_t = target_u.numel()

        P = preds_dense.view(B, N)
        T = target_dense.view(B, N)
        
        batch_offsets = torch.arange(B, device=preds.device).view(B, 1) * max_p
        P_shifted = P + batch_offsets
        
        hash_idx = P_shifted * max_t + T
        
        counts = torch.bincount(hash_idx.view(-1), minlength=B * max_p * max_t)
        counts = counts.view(B, max_p, max_t)
        
        s_size = counts.sum(dim=2, keepdim=True)  # Shape: (B, max_p, 1)

        errors = torch.minimum(counts, s_size - counts)
        error_per_image = errors.sum(dim=(1, 2))  # Shape: (B,)
        
        ue_per_image = error_per_image.float() / N

        self.sum_error += ue_per_image.sum()
        self.total += B
    
    def compute(self) -> torch.Tensor:
        return self.sum_error / self.total

