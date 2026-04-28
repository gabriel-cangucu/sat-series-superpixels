import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .semantic_pos_loss import SemanticPosLoss


class MILoss(nn.Module):
    def __init__(self, pos_weight: float = 0.003, kernel_size: int = 16) -> None:
        super().__init__()

        self.semantic_pos_loss = SemanticPosLoss(
            pos_weight=pos_weight,
            kernel_size=kernel_size
        )

    def forward(self,
                probs: torch.Tensor,
                targets: torch.Tensor,
                XY_features: torch.Tensor,
                modal_probs: torch.Tensor = None,
                align: torch.Tensor = None,
                mi: torch.Tensor = None,
                regularizer: Optional[nn.Module] = None
            ) -> torch.Tensor:
        
        loss = self.semantic_pos_loss(probs, targets, XY_features)

        if regularizer:
            modal_loss = self.semantic_pos_loss(modal_probs, targets, XY_features)
            aligment_loss = self._compute_alignment_loss(align)
            sample_loss = regularizer(mi[0].squeeze(), mi[1].squeeze())

            loss += modal_loss + aligment_loss + sample_loss
        
        return loss
    
    def _compute_alignment_loss(self, content):
        contentA, contentB = content
        b, c, h, w = contentA.shape

        # B, sp, dim, grid
        contentA = F.unfold(contentA, kernel_size=16, stride=16).permute(0, 2, 1).view(b, -1, c, 16 * 16).contiguous()
        contentB = F.unfold(contentB, kernel_size=16, stride=16).permute(0, 2, 1).view(b, -1, c, 16 * 16).contiguous()
        
        contentA = torch.mean(contentA, dim=3)
        contentB = torch.mean(contentB, dim=3)
        
        Pa = F.softmax(contentA, dim=2)
        Pb = F.softmax(contentB, dim=2)
        
        kl_divergence = torch.sum(Pa * torch.log(Pa / Pb), dim=2).mean() 
        
        return kl_divergence
    
