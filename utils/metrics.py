import torch
import torch.nn.functional as F
from torchmetrics import Metric


def _compute_boundary_map(seg: torch.Tensor) -> torch.Tensor:
    """
    Returns binary boundary map (H, W)
    """
    boundary = torch.zeros_like(seg, dtype=torch.bool)

    boundary[:, :-1] |= (seg[:, :-1] != seg[:, 1:])
    boundary[:-1, :] |= (seg[:-1, :] != seg[1:, :])

    return boundary


def _dilate(boundary: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Dilates a binary boundary map by given radius using max pooling.
    """
    if radius <= 0:
        return boundary

    boundary = boundary.float().unsqueeze(0)  # (1,1,H,W)

    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1, padding=radius)

    return dilated.squeeze(0).squeeze(0).bool()


class BoundaryPrecision(Metric):
    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        self.radius = radius

        self.add_state("sum_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        B = preds.shape[0]

        for b in range(B):
            val = self._boundary_precision(preds[b], target[b]).cpu()

            self.sum_score += val
            self.total += 1
    
    def compute(self) -> torch.Tensor:
        return self.sum_score / self.total

    def _boundary_precision(self, superpixel: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert superpixel.shape == target.shape

        sp_boundary = _compute_boundary_map(superpixel)
        gt_boundary = _compute_boundary_map(target)

        gt_dilated = _dilate(gt_boundary, self.radius)

        true_positive = (sp_boundary & gt_dilated).sum()
        predicted_positive = sp_boundary.sum()

        if predicted_positive == 0:
            return torch.tensor(0.0, device=superpixel.device)

        return true_positive.float() / predicted_positive.float()


class BoundaryRecall(Metric):
    def __init__(self, radius: int = 1) -> None:
        super().__init__()
        self.radius = radius

        self.add_state("sum_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        B = preds.shape[0]

        for b in range(B):
            val = self._boundary_recall(preds[b], target[b]).cpu()

            self.sum_score += val
            self.total += 1
    
    def compute(self) -> torch.Tensor:
        return self.sum_score / self.total

    def _boundary_recall(self, superpixel: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sp_boundary = _compute_boundary_map(superpixel)
        gt_boundary = _compute_boundary_map(target)

        sp_dilated = _dilate(sp_boundary, self.radius)

        true_positive = (gt_boundary & sp_dilated).sum()
        actual_positive = gt_boundary.sum()

        if actual_positive == 0:
            return torch.tensor(0.0, device=superpixel.device)

        return true_positive.float() / actual_positive.float()


class AchievableSegmentationAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("sum_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        B = preds.shape[0]

        for b in range(B):
            val = self._achievable_segmentation_accuracy(preds[b], target[b]).cpu()

            self.sum_score += val
            self.total += 1
    
    def compute(self) -> torch.Tensor:
        return self.sum_score / self.total

    def _achievable_segmentation_accuracy(self, superpixel: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert superpixel.shape == target.shape
        
        superpixel_flat = superpixel.view(-1)
        target_flat = target.view(-1)

        superpixel_ids = torch.unique(superpixel_flat)

        total = 0
        for sp_id in superpixel_ids:
            mask = (superpixel_flat == sp_id)
            gt_labels = target_flat[mask]

            counts = torch.bincount(gt_labels)
            if counts.numel() > 0:
                total += counts.max()

        return total.float() / superpixel_flat.numel()


class UndersegmentationError(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state("sum_error", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        B = preds.shape[0]

        for b in range(B):
            val = self._undersegmentation_error(preds[b], target[b]).cpu()

            self.sum_error += val
            self.total += 1
    
    def compute(self) -> torch.Tensor:
        return self.sum_error / self.total

    def _undersegmentation_error(self, superpixel: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert superpixel.shape == target.shape
        
        superpixel_flat = superpixel.view(-1)
        target_flat = target.view(-1)

        superpixel_ids = torch.unique(superpixel_flat)
        total_error = torch.tensor(0., device=superpixel.device)
        
        for sp_id in superpixel_ids:
            mask = (superpixel_flat == sp_id)
            gt_labels = target_flat[mask]

            counts = torch.bincount(gt_labels)
            counts = counts[counts > 0]
            
            if counts.numel() > 0:
                s_size = mask.sum()

                error = torch.minimum(counts, s_size - counts).sum()
                total_error += error

        return total_error.float() / superpixel_flat.numel()
