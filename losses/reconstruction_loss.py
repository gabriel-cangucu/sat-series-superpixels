import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self, compactness: float = 1e-5) -> None:
        super().__init__()

        self.compactness = compactness

    def forward(self,
                probs: torch.Tensor,
                targets: torch.Tensor,
                XY_features: torch.Tensor,
                hard_labels: torch.Tensor
            ) -> torch.Tensor:
        b, *_ = probs.shape

        targets = self._label_to_one_hot(targets, C=50).reshape(b, 50, -1)
        recons_loss = self._reconstruct_loss_with_cross_entropy(probs, targets)

        compact_loss = self._reconstruct_loss_with_mse(probs,
                                                       XY_features.reshape(*XY_features.shape[:2], -1),
                                                       hard_labels)

        return recons_loss + (self.compactness * compact_loss)
    
    def _reconstruction(self, assignment, labels, hard_assignment=None):
        """
        Reconstruction

        Args:
            assignment: torch.Tensor
                A Tensor of shape (B, n_spixels, n_pixels)
            labels: torch.Tensor
                A Tensor of shape (B, C, n_pixels)
            hard_assignment: torch.Tensor
                A Tensor of shape (B, n_pixels)
        """
        labels = labels.permute(0, 2, 1).contiguous()

        # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
        spixel_mean = torch.bmm(assignment, labels) / (assignment.sum(2, keepdim=True) + 1e-16)
        if hard_assignment is None:
            # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
            permuted_assignment = assignment.permute(0, 2, 1).contiguous()
            # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
            reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
        else:
            # index sampling
            max_valid_idx = spixel_mean.shape[1] - 1
            hard_assignment = hard_assignment.clamp(0, max_valid_idx)

            reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
            
        return reconstructed_labels.permute(0, 2, 1).contiguous()

    def _reconstruct_loss_with_cross_entropy(self, assignment, labels, hard_assignment=None):
        """
        Reconstruction loss with cross entropy

        Args:
            assignment: torch.Tensor
                A Tensor of shape (B, n_spixels, n_pixels)
            labels: torch.Tensor
                A Tensor of shape (B, C, n_pixels)
            hard_assignment: torch.Tensor
                A Tensor of shape (B, n_pixels)
        """
        reconstructed_labels = self._reconstruction(assignment, labels, hard_assignment)
        reconstructed_labels = reconstructed_labels / (1e-16 + reconstructed_labels.sum(1, keepdim=True))
        mask = labels > 0

        return -(reconstructed_labels[mask] + 1e-16).log().mean()
    
    def _reconstruct_loss_with_mse(self, assignment, labels, hard_assignment=None):
        """
        reconstruction loss with mse

        Args:
            assignment: torch.Tensor
                A Tensor of shape (B, n_spixels, n_pixels)
            labels: torch.Tensor
                A Tensor of shape (B, C, n_pixels)
            hard_assignment: torch.Tensor
                A Tensor of shape (B, n_pixels)
        """
        reconstructed_labels = self._reconstruction(assignment, labels, hard_assignment)
        return torch.nn.functional.mse_loss(reconstructed_labels, labels)
    
    def _label_to_one_hot(self, labels: torch.Tensor, C: int = 14) -> torch.Tensor:
        # w.r.t http://jacobkimmel.github.io/pytorch_onehot/
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
        C : integer.
            number of classes in labels.

        Returns
        -------
        target : torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        b, _, h, w = labels.shape
        one_hot = torch.zeros(b, C, h, w, dtype=torch.long).cuda()
        target = one_hot.scatter_(1, labels.type(torch.long).data, 1) #require long type

        return target.type(torch.float32)